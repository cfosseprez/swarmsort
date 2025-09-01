"""
Pytest configuration and fixtures for SwarmSort tests.
"""
import pytest
import numpy as np
from typing import List, Generator
import sys
from pathlib import Path

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarmsort import (
    SwarmSortTracker,
    SwarmSortConfig,
    Detection,
    TrackedObject,
    SwarmSort,
    create_tracker,
    EmbeddingDistanceScaler,
)


@pytest.fixture
def default_config():
    """Default SwarmSort configuration for testing."""
    return SwarmSortConfig(
        max_distance=50.0,
        init_conf_threshold=0.8,
        min_consecutive_detections=2,
        use_embeddings=True,
        embedding_weight=0.3,
    )


@pytest.fixture
def basic_config():
    """Basic configuration without embeddings for simple tests."""
    return SwarmSortConfig(max_distance=50.0, use_embeddings=False, min_consecutive_detections=2)


@pytest.fixture
def embedding_config():
    """Configuration optimized for embedding tests."""
    return SwarmSortConfig(
        max_distance=60.0,
        use_embeddings=True,
        embedding_weight=0.5,
        embedding_matching_method="best_match",
        max_embeddings_per_track=10,
        min_consecutive_detections=2,
        embedding_scaling_min_samples=50,  # Lower for faster tests
    )


@pytest.fixture
def strict_config():
    """Strict configuration for precision tests."""
    return SwarmSortConfig(
        max_distance=30.0,
        init_conf_threshold=0.9,
        min_consecutive_detections=4,
        detection_conf_threshold=0.8,
        use_embeddings=True,
        embedding_weight=0.6,
    )


@pytest.fixture
def permissive_config():
    """Permissive configuration for stress tests."""
    return SwarmSortConfig(
        max_distance=200.0,
        init_conf_threshold=0.3,
        min_consecutive_detections=1,
        detection_conf_threshold=0.1,
        max_track_age=50,
        use_embeddings=False,
    )


@pytest.fixture
def tracker(default_config):
    """Default tracker instance."""
    return SwarmSort(default_config)


@pytest.fixture
def embedding_tracker(embedding_config):
    """Tracker configured for embedding tests."""
    return SwarmSort(embedding_config)


@pytest.fixture
def basic_tracker(basic_config):
    """Basic tracker without embeddings."""
    return SwarmSort(basic_config)


@pytest.fixture
def sample_detections():
    """Sample detections for testing."""
    return [
        Detection(position=np.array([10.0, 20.0]), confidence=0.9, id="det1"),
        Detection(position=np.array([50.0, 60.0]), confidence=0.8, id="det2"),
        Detection(position=np.array([100.0, 30.0]), confidence=0.7, id="det3"),
    ]


@pytest.fixture
def sample_detections_with_embeddings():
    """Sample detections with embeddings."""
    np.random.seed(42)  # For reproducible tests
    return [
        Detection(
            position=np.array([10.0, 20.0]),
            confidence=0.9,
            embedding=np.random.randn(64).astype(np.float32),
            bbox=np.array([5.0, 15.0, 15.0, 25.0]),
            class_id=0,
            id="det1",
        ),
        Detection(
            position=np.array([50.0, 60.0]),
            confidence=0.8,
            embedding=np.random.randn(64).astype(np.float32),
            bbox=np.array([45.0, 55.0, 55.0, 65.0]),
            class_id=1,
            id="det2",
        ),
        Detection(
            position=np.array([100.0, 30.0]),
            confidence=0.7,
            embedding=np.random.randn(64).astype(np.float32),
            bbox=np.array([95.0, 25.0, 105.0, 35.0]),
            class_id=0,
            id="det3",
        ),
    ]


@pytest.fixture
def consistent_embeddings():
    """Consistent embeddings for tracking tests."""
    np.random.seed(123)
    return {
        "emb1": np.random.randn(128).astype(np.float32),
        "emb2": np.random.randn(128).astype(np.float32),
        "emb3": np.random.randn(128).astype(np.float32),
    }


@pytest.fixture
def tracking_scenario():
    """Generator for a complete tracking scenario."""

    def _scenario(num_objects=2, num_frames=10, noise_level=0.5):
        """
        Generate a tracking scenario with moving objects.

        Args:
            num_objects: Number of objects to track
            num_frames: Number of frames to generate
            noise_level: Amount of noise in detections
        """
        np.random.seed(42)
        scenarios = []

        # Initialize object states
        objects = []
        for i in range(num_objects):
            objects.append(
                {
                    "position": np.array([20.0 + i * 40, 30.0 + i * 30], dtype=np.float32),
                    "velocity": np.array([2.0 + i, 1.0 - i * 0.5], dtype=np.float32),
                    "embedding": np.random.randn(64).astype(np.float32),
                    "class_id": i % 3,
                }
            )

        for frame in range(num_frames):
            detections = []
            for i, obj in enumerate(objects):
                # Update position with some noise
                obj["position"] += obj["velocity"] + np.random.randn(2) * noise_level

                # Add small drift to embedding
                obj["embedding"] += np.random.randn(64) * 0.05

                # Create detection
                detection = Detection(
                    position=obj["position"].copy(),
                    confidence=0.9 - np.random.rand() * 0.2,
                    embedding=obj["embedding"].copy(),
                    bbox=np.array(
                        [
                            obj["position"][0] - 5,
                            obj["position"][1] - 5,
                            obj["position"][0] + 5,
                            obj["position"][1] + 5,
                        ]
                    ),
                    class_id=obj["class_id"],
                    id=f"det_{frame}_{i}",
                )
                detections.append(detection)

            scenarios.append(detections)

        return scenarios

    return _scenario


@pytest.fixture
def benchmark_data():
    """Large dataset for benchmarking."""

    def _data(num_objects=10, num_frames=100, embedding_dim=128):
        """Generate benchmark data."""
        np.random.seed(12345)
        data = []

        # Create object templates
        templates = []
        for i in range(num_objects):
            templates.append(
                {
                    "start_pos": np.random.rand(2) * 200,
                    "velocity": (np.random.rand(2) - 0.5) * 4,
                    "embedding": np.random.randn(embedding_dim).astype(np.float32),
                }
            )

        for frame in range(num_frames):
            detections = []
            for i, template in enumerate(templates):
                # Calculate position
                position = template["start_pos"] + template["velocity"] * frame
                position += np.random.randn(2) * 0.5  # Add noise

                # Add embedding variation
                embedding = template["embedding"] + np.random.randn(embedding_dim) * 0.1

                detection = Detection(
                    position=position,
                    confidence=0.9 - np.random.rand() * 0.2,
                    embedding=embedding.astype(np.float32),
                    bbox=np.array(
                        [position[0] - 10, position[1] - 10, position[0] + 10, position[1] + 10]
                    ),
                    class_id=i % 5,
                    id=f"bench_{frame}_{i}",
                )
                detections.append(detection)

            data.append(detections)

        return data

    return _data


@pytest.fixture(scope="session")
def temp_config_file(tmp_path_factory):
    """Create a temporary configuration file."""
    temp_dir = tmp_path_factory.mktemp("config")
    config_file = temp_dir / "test_config.yaml"

    config_content = """
max_distance: 75.0
init_conf_threshold: 0.85
use_embeddings: true
embedding_weight: 0.4
embedding_matching_method: "weighted_average"
min_consecutive_detections: 3
reid_enabled: true
reid_max_distance: 120.0
"""

    config_file.write_text(config_content)
    return str(config_file)


# Custom assertions and utilities for tests
class TrackingAssertions:
    """Custom assertions for tracking tests."""

    @staticmethod
    def assert_valid_tracked_object(tracked_obj: TrackedObject):
        """Assert that a TrackedObject is valid."""
        assert isinstance(tracked_obj.id, int)
        assert tracked_obj.id > 0
        assert isinstance(tracked_obj.position, np.ndarray)
        assert tracked_obj.position.shape == (2,)
        assert 0.0 <= tracked_obj.confidence <= 1.0
        assert tracked_obj.age >= 1
        assert tracked_obj.hits >= 1
        assert tracked_obj.time_since_update >= 0

    @staticmethod
    def assert_track_continuity(tracks_sequence: List[List[TrackedObject]]):
        """Assert that tracks maintain continuity across frames."""
        if len(tracks_sequence) < 2:
            return

        # Track ID consistency
        all_ids = set()
        for frame_tracks in tracks_sequence:
            frame_ids = {track.id for track in frame_tracks}
            # No duplicate IDs in single frame
            assert len(frame_ids) == len(frame_tracks)
            all_ids.update(frame_ids)

        # Check age progression
        track_ages = {}
        for frame_idx, frame_tracks in enumerate(tracks_sequence):
            for track in frame_tracks:
                if track.id not in track_ages:
                    track_ages[track.id] = []
                track_ages[track.id].append(track.age)

        # Ages should generally increase (with some exceptions for re-ID)
        for track_id, ages in track_ages.items():
            if len(ages) > 1:
                # At least some progression should occur
                assert max(ages) >= min(ages)

    @staticmethod
    def assert_no_track_jumps(
        tracks_sequence: List[List[TrackedObject]], max_distance: float = 100.0
    ):
        """Assert that tracks don't make unrealistic jumps."""
        track_positions = {}

        for frame_tracks in tracks_sequence:
            for track in frame_tracks:
                if track.id not in track_positions:
                    track_positions[track.id] = []
                track_positions[track.id].append(track.position.copy())

        for track_id, positions in track_positions.items():
            for i in range(1, len(positions)):
                distance = np.linalg.norm(positions[i] - positions[i - 1])
                assert distance <= max_distance, f"Track {track_id} jumped {distance:.2f} pixels"


@pytest.fixture
def tracking_assertions():
    """Provide tracking assertion utilities."""
    return TrackingAssertions()


# Performance measurement utilities
@pytest.fixture
def performance_tracker():
    """Track performance metrics during tests."""
    import time

    class PerformanceTracker:
        def __init__(self):
            self.measurements = {}

        def time_operation(self, name: str):
            """Context manager for timing operations."""

            class Timer:
                def __init__(self, tracker, operation_name):
                    self.tracker = tracker
                    self.name = operation_name
                    self.start_time = None

                def __enter__(self):
                    self.start_time = time.perf_counter()
                    return self

                def __exit__(self, exc_type, exc_val, exc_tb):
                    duration = time.perf_counter() - self.start_time
                    if self.name not in self.tracker.measurements:
                        self.tracker.measurements[self.name] = []
                    self.tracker.measurements[self.name].append(duration)

            return Timer(self, name)

        def get_stats(self, name: str):
            """Get statistics for a measurement."""
            if name not in self.measurements:
                return None

            measurements = self.measurements[name]
            return {
                "count": len(measurements),
                "total": sum(measurements),
                "mean": sum(measurements) / len(measurements),
                "min": min(measurements),
                "max": max(measurements),
            }

        def print_summary(self):
            """Print performance summary."""
            print("\\nPerformance Summary:")
            print("-" * 40)
            for name, measurements in self.measurements.items():
                stats = self.get_stats(name)
                print(f"{name}:")
                print(f"  Count: {stats['count']}")
                print(f"  Mean: {stats['mean']*1000:.2f}ms")
                print(f"  Total: {stats['total']*1000:.1f}ms")
                print(f"  Range: {stats['min']*1000:.2f}ms - {stats['max']*1000:.2f}ms")

    return PerformanceTracker()


# Pytest markers for test categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance and benchmark tests")
    config.addinivalue_line("markers", "stress: Stress and edge case tests")
    config.addinivalue_line("markers", "embedding: Embedding-specific tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
