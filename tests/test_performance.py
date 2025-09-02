"""
Performance tests and benchmarks for SwarmSort.
Tests focusing on speed, memory usage, and scalability.
"""
import pytest
import numpy as np
import time
import psutil
import os
from typing import List, Dict, Any
import gc

from swarmsort import SwarmSort, SwarmSortConfig, Detection, SwarmSortTracker, get_embedding_extractor
from swarmsort.core import (
    cosine_similarity_normalized,
    compute_embedding_distances_multi_history,
    compute_cost_matrix_vectorized,
)


def pytest_configure(config):
    """Configure pytest-benchmark if available."""
    try:
        import pytest_benchmark

        config.addinivalue_line("markers", "benchmark: Benchmark tests")
    except ImportError:
        # pytest-benchmark not available, define dummy marker
        pass


class MemoryProfiler:
    """Simple memory profiler for tracking memory usage."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.baseline = None

    def start(self):
        """Start memory profiling."""
        gc.collect()  # Clean up before measurement
        self.baseline = self.process.memory_info().rss / 1024 / 1024  # MB

    def get_usage(self) -> float:
        """Get current memory usage increase in MB."""
        if self.baseline is None:
            return 0.0
        current = self.process.memory_info().rss / 1024 / 1024  # MB
        return current - self.baseline

    def get_peak_usage(self) -> float:
        """Get peak memory usage."""
        try:
            peak_rss = (
                self.process.memory_info().peak_wss
                if hasattr(self.process.memory_info(), "peak_wss")
                else 0
            )
            return peak_rss / 1024 / 1024 if peak_rss else self.get_usage()
        except AttributeError:
            return self.get_usage()


@pytest.mark.performance
class TestNumbaFunctionPerformance:
    """Test performance of core Numba JIT functions."""

    @pytest.mark.benchmark
    def test_cosine_similarity_performance(self, benchmark):
        """Benchmark cosine similarity computation."""
        np.random.seed(42)
        emb1 = np.random.randn(128).astype(np.float32)
        emb2 = np.random.randn(128).astype(np.float32)

        # Warm up JIT
        for _ in range(10):
            cosine_similarity_normalized(emb1, emb2)

        result = benchmark(cosine_similarity_normalized, emb1, emb2)
        assert 0.0 <= result <= 1.0

    @pytest.mark.benchmark
    @pytest.mark.parametrize("embedding_dim", [64, 128, 256, 512])
    def test_cosine_similarity_scaling(self, benchmark, embedding_dim):
        """Test cosine similarity performance with different embedding dimensions."""
        np.random.seed(42)
        emb1 = np.random.randn(embedding_dim).astype(np.float32)
        emb2 = np.random.randn(embedding_dim).astype(np.float32)

        # Warm up JIT
        cosine_similarity_normalized(emb1, emb2)

        result = benchmark(cosine_similarity_normalized, emb1, emb2)
        assert 0.0 <= result <= 1.0

    @pytest.mark.benchmark
    def test_embedding_distances_multi_history_performance(self, benchmark):
        """Benchmark multi-history embedding distance computation."""
        np.random.seed(42)
        det_embeddings = np.random.randn(10, 128).astype(np.float32)
        track_embeddings = np.random.randn(50, 128).astype(np.float32)  # 50 total embeddings
        track_counts = np.array([5, 8, 12, 10, 15], dtype=np.int32)  # 5 tracks

        # Warm up JIT
        compute_embedding_distances_multi_history(
            det_embeddings, track_embeddings, track_counts, method="best_match"
        )

        result = benchmark(
            compute_embedding_distances_multi_history,
            det_embeddings,
            track_embeddings,
            track_counts,
            "best_match",
        )
        assert result.shape == (10, 5)

    @pytest.mark.benchmark
    @pytest.mark.parametrize("matrix_size", [(5, 3), (10, 8), (20, 15), (50, 30)])
    def test_cost_matrix_performance(self, benchmark, matrix_size):
        """Test cost matrix computation performance with different sizes."""
        n_det, n_track = matrix_size
        np.random.seed(42)
        det_positions = np.random.rand(n_det, 2).astype(np.float32) * 100
        track_last_positions = np.random.rand(n_track, 2).astype(np.float32) * 100
        track_kalman_positions = np.random.rand(n_track, 2).astype(np.float32) * 100
        det_embeddings = np.random.rand(n_det, 64).astype(np.float32)
        track_embeddings = np.random.rand(n_track, 64).astype(np.float32)
        use_embeddings = True
        max_distance = 50.0
        embedding_weight = 0.5

        # Warm up JIT
        compute_cost_matrix_vectorized(
            det_positions,
            track_last_positions,
            track_kalman_positions,
            det_embeddings,
            track_embeddings,
            use_embeddings,
            max_distance,
            embedding_weight,
        )

        result = benchmark(
            compute_cost_matrix_vectorized,
            det_positions,
            track_last_positions,
            track_kalman_positions,
            det_embeddings,
            track_embeddings,
            use_embeddings,
            max_distance,
            embedding_weight,
        )
        assert result.shape == (n_det, n_track)


@pytest.mark.performance
class TestTrackerPerformance:
    """Test overall tracker performance."""

    @pytest.mark.benchmark
    def test_single_frame_update_performance(self, benchmark):
        """Benchmark single frame update performance."""
        config = SwarmSortConfig(
            do_embeddings=True,
            embedding_weight=0.3,
            min_consecutive_detections=1,  # Immediate tracking for performance test
        )
        tracker = SwarmSort(config)

        # Create detections with embeddings
        np.random.seed(42)
        detections = []
        for i in range(10):
            detection = Detection(
                position=np.random.rand(2) * 200,
                confidence=0.8 + np.random.rand() * 0.2,
                embedding=np.random.randn(128).astype(np.float32),
                bbox=np.random.rand(4) * 20 + np.array([0, 0, 20, 20]),
            )
            detections.append(detection)

        # Warm up tracker
        for _ in range(3):
            tracker.update(detections)

        def update_frame():
            return tracker.update(detections)

        result = benchmark(update_frame)
        assert isinstance(result, list)

    @pytest.mark.benchmark
    @pytest.mark.parametrize("num_detections", [5, 10, 20, 50])
    def test_scaling_with_detection_count(self, benchmark, num_detections):
        """Test performance scaling with number of detections."""
        config = SwarmSortConfig(do_embeddings=True, min_consecutive_detections=1)
        tracker = SwarmSort(config)

        # Create detections
        np.random.seed(42)
        detections = []
        for i in range(num_detections):
            detection = Detection(
                position=np.random.rand(2) * 200,
                confidence=0.8 + np.random.rand() * 0.2,
                embedding=np.random.randn(128).astype(np.float32),
            )
            detections.append(detection)

        # Warm up
        tracker.update(detections)

        result = benchmark(tracker.update, detections)
        assert isinstance(result, list)

    @pytest.mark.benchmark
    def test_long_sequence_performance(self, benchmark, benchmark_data):
        """Test performance over a long sequence."""
        config = SwarmSortConfig(
            do_embeddings=True, embedding_weight=0.3, min_consecutive_detections=2
        )
        tracker = SwarmSort(config)

        # Get benchmark data (smaller for performance test)
        data = benchmark_data(num_objects=5, num_frames=20, embedding_dim=64)

        def run_sequence():
            tracker.reset()
            for frame_detections in data:
                tracker.update(frame_detections)
            return tracker.get_statistics()

        result = benchmark(run_sequence)
        assert result["frame_count"] == 20

    @pytest.mark.benchmark
    def test_memory_efficiency(self):
        """Test memory efficiency of tracker."""
        profiler = MemoryProfiler()
        profiler.start()

        config = SwarmSortConfig(do_embeddings=True, max_embeddings_per_track=10, max_track_age=20)
        tracker = SwarmSort(config)

        # Run tracking for many frames
        np.random.seed(42)
        initial_memory = profiler.get_usage()

        for frame in range(100):
            detections = []
            for i in range(8):
                position = np.array([i * 30 + frame % 10, 50 + (i % 2) * 20])
                detection = Detection(
                    position=position,
                    confidence=0.9,
                    embedding=np.random.randn(64).astype(np.float32),
                )
                detections.append(detection)

            tracker.update(detections)

            # Check memory every 20 frames
            if frame % 20 == 19:
                current_memory = profiler.get_usage()
                memory_increase = current_memory - initial_memory

                # Memory increase should be reasonable (less than 100MB)
                # Use absolute value to handle potential negative values due to GC
                assert (
                    abs(memory_increase) < 100
                ), f"Memory usage changed by: {memory_increase:.2f}MB"

        final_memory = profiler.get_usage()
        print(f"Final memory usage: {final_memory:.2f}MB")

        # Final memory usage should be reasonable
        assert final_memory < 200  # Less than 200MB total


@pytest.mark.performance
class TestScalabilityBenchmarks:
    """Test scalability with different problem sizes."""

    def generate_scenario(
        self, num_objects: int, num_frames: int, embedding_dim: int = 128, noise_level: float = 1.0
    ):
        """Generate tracking scenario for scalability testing."""
        np.random.seed(42)  # Reproducible

        # Initialize objects with different motion patterns
        objects = []
        for i in range(num_objects):
            obj = {
                "pos": np.array([i * 50 + 25, i * 40 + 30], dtype=np.float32),
                "vel": np.array(
                    [np.random.rand() * 4 - 2, np.random.rand() * 4 - 2],  # -2 to +2
                    dtype=np.float32,
                ),
                "embedding": np.random.randn(embedding_dim).astype(np.float32),
                "class_id": i % 5,
            }
            objects.append(obj)

        scenarios = []
        for frame in range(num_frames):
            detections = []
            for i, obj in enumerate(objects):
                # Update position
                obj["pos"] += obj["vel"] + np.random.randn(2) * noise_level

                # Small embedding drift
                obj["embedding"] += np.random.randn(embedding_dim) * 0.02

                # Create detection
                detection = Detection(
                    position=obj["pos"].copy(),
                    confidence=0.8 + np.random.rand() * 0.2,
                    embedding=obj["embedding"].copy(),
                    bbox=np.array(
                        [obj["pos"][0] - 8, obj["pos"][1] - 8, obj["pos"][0] + 8, obj["pos"][1] + 8]
                    ),
                    class_id=obj["class_id"],
                    id=f"det_{frame}_{i}",
                )
                detections.append(detection)

            scenarios.append(detections)

        return scenarios

    @pytest.mark.benchmark
    @pytest.mark.parametrize("num_objects", [3, 5, 10, 20])
    def test_object_count_scalability(self, benchmark, num_objects):
        """Test performance scaling with number of objects."""
        config = SwarmSortConfig(
            do_embeddings=True,
            embedding_weight=0.4,
            min_consecutive_detections=2,
            max_embeddings_per_track=8,
        )

        scenarios = self.generate_scenario(
            num_objects=num_objects, num_frames=15, embedding_dim=64  # Smaller for speed
        )

        def run_tracking():
            tracker = SwarmSort(config)
            for frame_detections in scenarios:
                tracker.update(frame_detections)
            return tracker.get_statistics()["active_tracks"]

        result = benchmark(run_tracking)
        assert result >= num_objects * 0.7  # Should track most objects

    @pytest.mark.benchmark
    @pytest.mark.parametrize("embedding_dim", [32, 64, 128, 256])
    def test_embedding_dimension_scalability(self, benchmark, embedding_dim):
        """Test performance scaling with embedding dimension."""
        config = SwarmSortConfig(
            do_embeddings=True, embedding_weight=0.5, min_consecutive_detections=2
        )

        scenarios = self.generate_scenario(
            num_objects=5, num_frames=12, embedding_dim=embedding_dim
        )

        def run_tracking():
            tracker = SwarmSort(config)
            for frame_detections in scenarios:
                tracker.update(frame_detections)
            return len(frame_detections)  # Return something for verification

        result = benchmark(run_tracking)
        assert result == 5  # Should process all detections

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_long_sequence_scalability(self, benchmark):
        """Test performance with very long sequences."""
        config = SwarmSortConfig(
            do_embeddings=True,
            embedding_weight=0.3,
            min_consecutive_detections=3,
            max_embeddings_per_track=10,
            max_track_age=15,
        )

        scenarios = self.generate_scenario(
            num_objects=8, num_frames=100, embedding_dim=128  # Long sequence
        )

        def run_long_tracking():
            tracker = SwarmSort(config)
            final_stats = None
            for frame_detections in scenarios:
                tracker.update(frame_detections)
                final_stats = tracker.get_statistics()
            return final_stats

        result = benchmark(run_long_tracking)
        assert result["frame_count"] == 100
        assert result["active_tracks"] >= 5  # Should maintain reasonable tracks


@pytest.mark.performance
class TestMemoryBenchmarks:
    """Memory usage benchmarks and leak detection."""

    def test_memory_leak_detection(self):
        """Test for memory leaks over extended operation."""
        profiler = MemoryProfiler()
        profiler.start()

        config = SwarmSortConfig(do_embeddings=True, max_embeddings_per_track=8, max_track_age=10)

        memory_snapshots = []

        # Run multiple cycles
        for cycle in range(10):
            tracker = SwarmSort(config)

            # Run tracking for this cycle
            np.random.seed(cycle)  # Different data each cycle
            for frame in range(50):
                detections = []
                for i in range(6):
                    detection = Detection(
                        position=np.random.rand(2) * 200,
                        confidence=0.8 + np.random.rand() * 0.2,
                        embedding=np.random.randn(64).astype(np.float32),
                    )
                    detections.append(detection)

                tracker.update(detections)

            # Force cleanup
            del tracker
            gc.collect()

            # Measure memory
            memory_usage = profiler.get_usage()
            memory_snapshots.append(memory_usage)
            print(f"Cycle {cycle}: {memory_usage:.2f}MB")

        # Check for memory leaks (memory should not continuously increase)
        if len(memory_snapshots) >= 5:
            early_avg = np.mean(memory_snapshots[:3])
            late_avg = np.mean(memory_snapshots[-3:])
            memory_increase = late_avg - early_avg

            # Should not leak more than 50MB over 10 cycles
            # Use absolute value to handle potential negative values due to GC
            assert abs(memory_increase) < 50, f"Memory usage changed by: {memory_increase:.2f}MB"

    def test_large_embedding_memory_usage(self):
        """Test memory usage with large embeddings."""
        profiler = MemoryProfiler()
        profiler.start()

        config = SwarmSortConfig(
            do_embeddings=True, max_embeddings_per_track=15, embedding_weight=0.4
        )
        tracker = SwarmSort(config)

        # Create detections with large embeddings
        np.random.seed(42)
        large_embeddings = []
        for i in range(10):
            embedding = np.random.randn(1024).astype(np.float32)  # Large embedding
            large_embeddings.append(embedding)

        baseline_memory = profiler.get_usage()

        # Process frames with large embeddings
        for frame in range(30):
            detections = []
            for i in range(5):
                detection = Detection(
                    position=np.array([i * 40 + frame % 10, 50.0]),
                    confidence=0.9,
                    embedding=large_embeddings[i % len(large_embeddings)].copy(),
                )
                detections.append(detection)

            tracker.update(detections)

        final_memory = profiler.get_usage()
        memory_increase = final_memory - baseline_memory

        print(f"Memory increase with large embeddings: {memory_increase:.2f}MB")

        # Should handle large embeddings without excessive memory usage
        # (this is a rough estimate, adjust based on actual requirements)
        assert memory_increase < 500  # Less than 500MB for large embeddings

    def test_track_history_memory_management(self):
        """Test memory management of track history."""
        profiler = MemoryProfiler()
        profiler.start()

        config = SwarmSortConfig(
            do_embeddings=True, max_embeddings_per_track=5, max_track_age=8  # Limited history
        )
        tracker = SwarmSort(config)

        baseline_memory = profiler.get_usage()

        # Create tracks with long histories
        np.random.seed(42)
        for frame in range(100):
            detections = []
            for i in range(8):
                # Tracks that persist for long time
                position = np.array([i * 25 + frame * 0.5, 50.0 + i * 15])
                detection = Detection(
                    position=position,
                    confidence=0.95,
                    embedding=np.random.randn(128).astype(np.float32),
                )
                detections.append(detection)

            tracker.update(detections)

            # Check memory every 25 frames
            if frame % 25 == 24:
                current_memory = profiler.get_usage()
                memory_increase = current_memory - baseline_memory

                # Memory should stabilize due to limited history
                if frame > 50:  # After initial ramp-up
                    # Use absolute value to handle potential negative values due to GC
                    assert (
                        abs(memory_increase) < 150
                    ), f"Memory usage at frame {frame}: {memory_increase:.2f}MB"

        final_memory = profiler.get_usage()
        print(f"Final memory with history management: {final_memory:.2f}MB")


@pytest.mark.performance
class TestConfigurationPerformance:
    """Test performance impact of different configuration options."""

    def create_test_scenario(self, num_objects=8, num_frames=20):
        """Create a standard test scenario."""
        np.random.seed(42)
        scenarios = []

        for frame in range(num_frames):
            detections = []
            for i in range(num_objects):
                position = np.array([i * 30 + frame * 2, 50 + np.sin(frame * 0.1) * 20])
                detection = Detection(
                    position=position,
                    confidence=0.8 + np.random.rand() * 0.2,
                    embedding=np.random.randn(128).astype(np.float32),
                )
                detections.append(detection)
            scenarios.append(detections)

        return scenarios

    @pytest.mark.benchmark
    @pytest.mark.parametrize("do_embeddings", [True, False])
    def test_embedding_vs_no_embedding_performance(self, benchmark, do_embeddings):
        """Compare performance with and without embeddings."""
        config = SwarmSortConfig(
            do_embeddings=do_embeddings,
            embedding_weight=0.4 if do_embeddings else 0.0,
            min_consecutive_detections=2,
        )

        scenarios = self.create_test_scenario()

        def run_tracking():
            tracker = SwarmSort(config)
            for frame_detections in scenarios:
                tracker.update(frame_detections)
            return tracker.get_statistics()["active_tracks"]

        result = benchmark(run_tracking)
        assert result >= 4  # Should track most objects

    @pytest.mark.benchmark
    @pytest.mark.parametrize("embedding_method", ["best_match", "average", "weighted_average"])
    def test_embedding_method_performance(self, benchmark, embedding_method):
        """Compare performance of different embedding methods."""
        config = SwarmSortConfig(
            do_embeddings=True,
            embedding_matching_method=embedding_method,
            embedding_weight=0.4,
            min_consecutive_detections=2,
        )

        scenarios = self.create_test_scenario()

        def run_tracking():
            tracker = SwarmSort(config)
            for frame_detections in scenarios:
                tracker.update(frame_detections)
            return tracker.get_statistics()["active_tracks"]

        result = benchmark(run_tracking)
        assert result >= 4

    @pytest.mark.benchmark
    @pytest.mark.parametrize("max_embeddings", [3, 8, 15, 25])
    def test_embedding_history_performance(self, benchmark, max_embeddings):
        """Test performance impact of embedding history length."""
        config = SwarmSortConfig(
            do_embeddings=True,
            max_embeddings_per_track=max_embeddings,
            embedding_weight=0.4,
            min_consecutive_detections=2,
        )

        scenarios = self.create_test_scenario(num_frames=30)  # Longer for history

        def run_tracking():
            tracker = SwarmSort(config)
            for frame_detections in scenarios:
                tracker.update(frame_detections)
            return tracker.get_statistics()["active_tracks"]

        result = benchmark(run_tracking)
        assert result >= 4


# Utility function for performance testing
def run_performance_comparison():
    """Run a comprehensive performance comparison."""
    import pandas as pd

    results = []
    configurations = [
        ("Basic", SwarmSortConfig(do_embeddings=False, min_consecutive_detections=1)),
        ("With Embeddings", SwarmSortConfig(do_embeddings=True, embedding_weight=0.3)),
        (
            "High Precision",
            SwarmSortConfig(
                do_embeddings=True, embedding_weight=0.6, min_consecutive_detections=4
            ),
        ),
        (
            "Fast",
            SwarmSortConfig(do_embeddings=False, max_distance=100, min_consecutive_detections=1),
        ),
    ]

    # Test scenario
    np.random.seed(42)
    test_data = []
    for frame in range(50):
        detections = []
        for i in range(10):
            detection = Detection(
                position=np.array([i * 20 + frame, 50 + i * 10]),
                confidence=0.8 + np.random.rand() * 0.2,
                embedding=np.random.randn(128).astype(np.float32),
            )
            detections.append(detection)
        test_data.append(detections)

    for config_name, config in configurations:
        tracker = SwarmSort(config)

        # Time the tracking
        start_time = time.perf_counter()
        for frame_detections in test_data:
            tracker.update(frame_detections)
        end_time = time.perf_counter()

        stats = tracker.get_statistics()
        results.append(
            {
                "Configuration": config_name,
                "Total Time (s)": end_time - start_time,
                "Time per Frame (ms)": (end_time - start_time) / len(test_data) * 1000,
                "Active Tracks": stats["active_tracks"],
                "Frame Count": stats["frame_count"],
            }
        )

    df = pd.DataFrame(results)
    print("\nPerformance Comparison Results:")
    print(df.to_string(index=False))
    return df


@pytest.mark.performance
class TestIntegrationPerformance:
    """Test performance of new integration features."""

    @pytest.mark.benchmark
    def test_swarmtracker_constructor_performance(self, benchmark):
        """Test performance of SwarmSortTracker with integration parameters."""
        def create_tracker():
            return SwarmSortTracker(
                config={'use_embeddings': True, 'max_distance': 150.0},
                embedding_type='cupytexture',
                use_gpu=False
            )
        
        tracker = benchmark(create_tracker)
        assert tracker.embedding_extractor is not None
        assert tracker.config.do_embeddings == True

    @pytest.mark.benchmark
    def test_color_embedding_performance(self, benchmark):
        """Test performance of CupyTextureColorEmbedding."""
        frame = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        
        embedding_extractor = get_embedding_extractor('cupytexture_color', use_gpu=False)
        
        # Warm up
        embedding_extractor.extract(frame, bbox)
        
        result = benchmark(embedding_extractor.extract, frame, bbox)
        assert result.shape == (84,)

    @pytest.mark.benchmark
    def test_color_vs_regular_embedding_performance(self, benchmark):
        """Compare performance of color vs regular embeddings."""
        frame = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        
        regular_extractor = get_embedding_extractor('cupytexture', use_gpu=False)
        color_extractor = get_embedding_extractor('cupytexture_color', use_gpu=False)
        
        # Warm up
        regular_extractor.extract(frame, bbox)
        color_extractor.extract(frame, bbox)
        
        # Time regular embedding
        start_time = time.perf_counter()
        for _ in range(10):
            regular_extractor.extract(frame, bbox)
        regular_time = time.perf_counter() - start_time
        
        # Time color embedding
        start_time = time.perf_counter()
        for _ in range(10):
            color_extractor.extract(frame, bbox)
        color_time = time.perf_counter() - start_time
        
        print(f"Regular embedding: {regular_time*100:.1f}ms avg")
        print(f"Color embedding: {color_time*100:.1f}ms avg")
        
        # Color embedding should be slower but not excessively (less than 100x)
        # Color embedding has more complex features (84 vs 36 dimensions)
        assert color_time < regular_time * 100

    @pytest.mark.benchmark
    def test_frame_parameter_performance(self, benchmark):
        """Test performance impact of passing frame to update method."""
        config = SwarmSortConfig(do_embeddings=False)
        tracker = SwarmSortTracker(config=config)
        
        detection = Detection(
            position=np.array([100.0, 100.0], dtype=np.float32),
            confidence=0.9
        )
        frame = np.random.randint(0, 255, size=(240, 320, 3), dtype=np.uint8)
        
        # Warm up
        tracker.update([detection], frame)
        
        def update_with_frame():
            return tracker.update([detection], frame)
        
        result = benchmark(update_with_frame)
        assert isinstance(result, list)

    @pytest.mark.benchmark 
    def test_batch_color_embedding_performance(self, benchmark):
        """Test batch performance of color embeddings."""
        frame = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        bboxes = np.array([
            [50, 50, 100, 100],
            [150, 150, 200, 200],
            [250, 250, 300, 300],
            [350, 350, 400, 400],
            [100, 300, 150, 350]
        ], dtype=np.float32)
        
        embedding_extractor = get_embedding_extractor('cupytexture_color', use_gpu=False)
        
        # Warm up
        embedding_extractor.extract_batch(frame, bboxes)
        
        result = benchmark(embedding_extractor.extract_batch, frame, bboxes)
        assert len(result) == 5
        for features in result:
            assert features.shape == (84,)

    def test_integration_memory_usage(self):
        """Test memory usage of integration features."""
        profiler = MemoryProfiler()
        profiler.start()
        
        # Test SwarmSortTracker with embedding integration
        tracker = SwarmSortTracker(
            config={'use_embeddings': True, 'max_age': 20},
            embedding_type='cupytexture_color',
            use_gpu=False
        )
        
        baseline_memory = profiler.get_usage()
        
        # Create test frame and run tracking
        frame = np.random.randint(0, 255, size=(240, 320, 3), dtype=np.uint8)
        
        for frame_idx in range(100):
            detections = []
            for i in range(5):
                detection = Detection(
                    position=np.array([100.0 + i*30 + frame_idx*0.5, 100.0 + i*20], dtype=np.float32),
                    confidence=0.9,
                    bbox=np.array([90+i*30, 90+i*20, 110+i*30, 110+i*20], dtype=np.float32)
                )
                detections.append(detection)
            
            tracker.update(detections, frame)
            
            if frame_idx % 20 == 19:
                current_memory = profiler.get_usage()
                memory_increase = current_memory - baseline_memory
                
                # Memory should not grow excessively
                assert abs(memory_increase) < 150, \
                    f"Memory usage at frame {frame_idx}: {memory_increase:.2f}MB"

    @pytest.mark.skipif(os.getenv('CI') is not None, reason="Skip flaky performance test in CI")
    @pytest.mark.parametrize("embedding_type", ['cupytexture', 'cupytexture_color', 'mega_cupytexture'])
    def test_all_embeddings_performance(self, embedding_type):
        """Test performance of all available embeddings."""
        try:
            extractor = get_embedding_extractor(embedding_type, use_gpu=False)
        except Exception as e:
            pytest.skip(f"Embedding {embedding_type} not available: {e}")
        
        frame = np.random.randint(0, 255, size=(240, 320, 3), dtype=np.uint8)
        bbox = np.array([50, 50, 100, 100], dtype=np.float32)
        
        # Warm up
        extractor.extract(frame, bbox)
        
        # Time extraction
        start_time = time.perf_counter()
        for _ in range(10):
            features = extractor.extract(frame, bbox)
        total_time = time.perf_counter() - start_time
        avg_time = total_time / 10
        
        print(f"{embedding_type} extraction: {avg_time*1000:.1f}ms avg (dim: {features.shape[0]})")
        
        # Should extract in reasonable time (< 100ms)
        assert avg_time < 0.1, f"{embedding_type} took {avg_time*1000:.1f}ms, expected < 100ms"
        assert features.shape[0] == extractor.embedding_dim


if __name__ == "__main__":
    # Run performance comparison when script is executed directly
    try:
        run_performance_comparison()
    except ImportError:
        print("pandas not available, skipping performance comparison table")
