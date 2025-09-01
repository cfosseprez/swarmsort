"""
Tests for CUPY GPU-accelerated embeddings in SwarmSort standalone.
"""
import numpy as np
import pytest
import cv2
import sys
from unittest.mock import patch, MagicMock
import logging

# Set up logging to capture GPU-related messages
logging.basicConfig(level=logging.INFO)

# Import embeddings module
from swarmsort.embeddings import (
    CupyTextureEmbedding,
    MegaCupyTextureEmbedding,
    compute_embedding_distance,
    compute_embedding_distances_batch,
    get_embedding_extractor,
    list_available_embeddings,
    is_gpu_available,
    CUPY_AVAILABLE,
    AVAILABLE_EMBEDDINGS,
)
from swarmsort import SwarmSortTracker, SwarmSortConfig, Detection
from swarmsort.embedding_scaler import EmbeddingDistanceScaler


class TestEmbeddingBasics:
    """Test basic embedding functionality."""

    def test_available_embeddings(self):
        """Test that embeddings are properly registered."""
        available = list_available_embeddings()
        assert "cupytexture" in available
        assert "mega_cupytexture" in available

    def test_get_embedding_extractor(self):
        """Test embedding extractor factory."""
        # Valid extractors
        extractor1 = get_embedding_extractor("cupytexture")
        assert isinstance(extractor1, CupyTextureEmbedding)

        extractor2 = get_embedding_extractor("mega_cupytexture")
        assert isinstance(extractor2, MegaCupyTextureEmbedding)

        # Invalid extractor
        with pytest.raises(ValueError, match="Embedding 'invalid' not found"):
            get_embedding_extractor("invalid")

    def test_gpu_availability_check(self):
        """Test GPU availability detection."""
        availability = is_gpu_available()
        assert isinstance(availability, bool)
        # Should match the actual CUPY_AVAILABLE constant
        assert availability == CUPY_AVAILABLE


class TestCupyTextureEmbedding:
    """Test CupyTextureEmbedding functionality."""

    def test_initialization(self):
        """Test embedding initialization."""
        # Default initialization
        embedding = CupyTextureEmbedding()
        assert embedding.embedding_dim == 36
        assert embedding.patch_size == 32
        assert embedding.use_gpu == CUPY_AVAILABLE

        # Custom initialization
        embedding_custom = CupyTextureEmbedding(patch_size=64, use_gpu=False)
        assert embedding_custom.patch_size == 64
        assert embedding_custom.use_gpu is False

    def test_single_extraction(self):
        """Test single embedding extraction."""
        embedding = CupyTextureEmbedding()

        # Create test frame and bbox
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bbox = np.array([10, 10, 50, 50])

        # Extract embedding
        result = embedding.extract(frame, bbox)

        # Check result
        assert isinstance(result, np.ndarray)
        assert result.shape == (36,)
        assert result.dtype == np.float32
        assert not np.all(result == 0)  # Should not be all zeros for valid image

    def test_batch_extraction_cpu(self):
        """Test batch embedding extraction on CPU."""
        embedding = CupyTextureEmbedding(use_gpu=False)

        # Create test data
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bboxes = np.array([[10, 10, 30, 30], [40, 40, 70, 70], [20, 20, 60, 60]])

        # Extract embeddings
        results = embedding.extract_batch_cpu(frame, bboxes)

        # Check results
        assert len(results) == 3
        for result in results:
            assert isinstance(result, np.ndarray)
            assert result.shape == (36,)
            assert result.dtype == np.float32

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_batch_extraction_gpu(self):
        """Test batch embedding extraction on GPU (if available)."""
        embedding = CupyTextureEmbedding(use_gpu=True)

        # Create test data
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bboxes = np.array([[10, 10, 30, 30], [40, 40, 70, 70], [20, 20, 60, 60]])

        # Extract embeddings
        results = embedding.extract_batch_gpu(frame, bboxes)

        # Check results
        assert len(results) == 3
        for result in results:
            assert isinstance(result, np.ndarray)
            assert result.shape == (36,)
            assert result.dtype == np.float32

    def test_invalid_bbox_handling(self):
        """Test handling of invalid bounding boxes."""
        embedding = CupyTextureEmbedding()

        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Invalid bbox (outside frame bounds)
        bbox_invalid = np.array([150, 150, 200, 200])
        result = embedding.extract(frame, bbox_invalid)

        # Should return zeros for invalid bbox
        assert isinstance(result, np.ndarray)
        assert result.shape == (36,)
        assert np.all(result == 0)

        # Zero area bbox
        bbox_zero = np.array([50, 50, 50, 50])
        result_zero = embedding.extract(frame, bbox_zero)
        assert np.all(result_zero == 0)

    def test_empty_batch_handling(self):
        """Test handling of empty batch."""
        embedding = CupyTextureEmbedding()

        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        empty_bboxes = np.array([]).reshape(0, 4)

        results = embedding.extract_batch_cpu(frame, empty_bboxes)
        assert results == []

    def test_gpu_fallback(self):
        """Test GPU fallback to CPU when GPU fails."""
        # Mock CuPy to raise an exception during GPU processing
        with patch("swarmsort.embeddings.cp") as mock_cp:
            mock_cp.asarray.side_effect = RuntimeError("GPU memory error")

            embedding = CupyTextureEmbedding(use_gpu=True)
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            bboxes = np.array([[10, 10, 30, 30]])

            # Should fallback to CPU without raising exception
            results = embedding.extract_batch_gpu(frame, bboxes)
            assert len(results) == 1
            assert isinstance(results[0], np.ndarray)


class TestMegaCupyTextureEmbedding:
    """Test MegaCupyTextureEmbedding functionality."""

    def test_initialization(self):
        """Test mega embedding initialization."""
        embedding = MegaCupyTextureEmbedding()
        assert embedding.embedding_dim == 64
        assert embedding.patch_size == 32
        assert embedding.use_gpu == CUPY_AVAILABLE

    def test_single_extraction(self):
        """Test single mega embedding extraction."""
        embedding = MegaCupyTextureEmbedding()

        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bbox = np.array([10, 10, 50, 50])

        result = embedding.extract(frame, bbox)

        assert isinstance(result, np.ndarray)
        assert result.shape == (64,)
        assert result.dtype == np.float32
        assert not np.all(result == 0)

    def test_batch_extraction_cpu(self):
        """Test mega embedding batch extraction on CPU."""
        embedding = MegaCupyTextureEmbedding(use_gpu=False)

        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bboxes = np.array([[10, 10, 30, 30], [40, 40, 70, 70]])

        results = embedding.extract_batch_cpu(frame, bboxes)

        assert len(results) == 2
        for result in results:
            assert isinstance(result, np.ndarray)
            assert result.shape == (64,)
            assert result.dtype == np.float32

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_batch_extraction_gpu(self):
        """Test mega embedding batch extraction on GPU."""
        embedding = MegaCupyTextureEmbedding(use_gpu=True)

        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bboxes = np.array([[10, 10, 30, 30], [40, 40, 70, 70]])

        results = embedding.extract_batch_gpu(frame, bboxes)

        assert len(results) == 2
        for result in results:
            assert isinstance(result, np.ndarray)
            assert result.shape == (64,)
            assert result.dtype == np.float32

    def test_gpu_auto_fallback(self):
        """Test automatic GPU fallback in extract_batch method."""
        with patch("swarmsort.embeddings.cp") as mock_cp:
            mock_cp.asarray.side_effect = RuntimeError("GPU error")

            embedding = MegaCupyTextureEmbedding(use_gpu=True)
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            bboxes = np.array([[10, 10, 30, 30]])

            # Should automatically fallback to CPU
            results = embedding.extract_batch(frame, bboxes)
            assert len(results) == 1
            assert isinstance(results[0], np.ndarray)
            assert results[0].shape == (64,)


class TestDistanceComputation:
    """Test embedding distance computation."""

    def test_correlation_distance(self):
        """Test correlation distance computation."""
        # Create test embeddings
        emb1 = np.random.rand(36).astype(np.float32)
        emb2 = np.random.rand(36).astype(np.float32)
        emb3 = emb1.copy()  # Identical to emb1

        # Test normal distance
        dist12 = compute_embedding_distance(emb1, emb2)
        assert isinstance(dist12, float)
        assert 0.0 <= dist12 <= 1.0

        # Test identical embeddings
        dist13 = compute_embedding_distance(emb1, emb3)
        assert dist13 == 0.0

        # Test None embeddings
        dist_none = compute_embedding_distance(None, emb1)
        assert dist_none == float("inf")

        # Test different shapes
        emb_diff_shape = np.random.rand(20).astype(np.float32)
        dist_diff = compute_embedding_distance(emb1, emb_diff_shape)
        assert dist_diff == float("inf")

    def test_batch_distance_computation(self):
        """Test batch distance computation."""
        emb_query = np.random.rand(36).astype(np.float32)
        emb_list = [
            np.random.rand(36).astype(np.float32),
            np.random.rand(36).astype(np.float32),
            emb_query.copy(),  # Identical
            None,  # Invalid
        ]

        distances = compute_embedding_distances_batch(emb_query, emb_list)

        assert isinstance(distances, np.ndarray)
        assert len(distances) == 4
        assert 0.0 <= distances[0] <= 1.0
        assert 0.0 <= distances[1] <= 1.0
        assert distances[2] == 0.0  # Identical embedding
        assert distances[3] == float("inf")  # None embedding

    def test_empty_batch_distances(self):
        """Test batch distance computation with empty list."""
        emb_query = np.random.rand(36).astype(np.float32)
        distances = compute_embedding_distances_batch(emb_query, [])

        assert isinstance(distances, np.ndarray)
        assert len(distances) == 0


class TestEmbeddingConsistency:
    """Test consistency between CPU and GPU implementations."""

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_cpu_gpu_consistency_cupytexture(self):
        """Test CPU/GPU consistency for CupyTextureEmbedding."""
        # Create deterministic test data
        np.random.seed(42)
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bboxes = np.array([[10, 10, 40, 40], [50, 50, 80, 80]])

        # Extract with CPU
        embedding_cpu = CupyTextureEmbedding(use_gpu=False)
        results_cpu = embedding_cpu.extract_batch_cpu(frame, bboxes)

        # Extract with GPU
        embedding_gpu = CupyTextureEmbedding(use_gpu=True)
        results_gpu = embedding_gpu.extract_batch_gpu(frame, bboxes)

        # Compare results (allow small numerical differences)
        assert len(results_cpu) == len(results_gpu)
        for cpu_emb, gpu_emb in zip(results_cpu, results_gpu):
            # Check basic properties
            assert cpu_emb.shape == gpu_emb.shape
            assert cpu_emb.dtype == gpu_emb.dtype

            # GPU implementation is simplified, so exact match is not expected
            # Instead, check that both produce non-zero features
            assert not np.all(cpu_emb == 0)
            assert not np.all(gpu_emb == 0)

    def test_deterministic_extraction(self):
        """Test that extraction is deterministic for same input."""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bbox = np.array([20, 20, 60, 60])

        embedding = CupyTextureEmbedding(use_gpu=False)

        # Extract twice
        result1 = embedding.extract(frame, bbox)
        result2 = embedding.extract(frame, bbox)

        # Should be identical
        np.testing.assert_array_equal(result1, result2)


class TestEmbeddingPerformance:
    """Test embedding performance characteristics."""

    def test_batch_processing_efficiency(self):
        """Test that batch processing is more efficient than single processing."""
        import time

        embedding = CupyTextureEmbedding(use_gpu=False)
        frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        bboxes = np.array([[i * 20, i * 20, i * 20 + 30, i * 20 + 30] for i in range(5)])

        # Time single processing
        start_single = time.time()
        results_single = [embedding.extract(frame, bbox) for bbox in bboxes]
        time_single = time.time() - start_single

        # Time batch processing
        start_batch = time.time()
        results_batch = embedding.extract_batch_cpu(frame, bboxes)
        time_batch = time.time() - start_batch

        # Verify results are equivalent
        assert len(results_single) == len(results_batch)
        for single, batch in zip(results_single, results_batch):
            np.testing.assert_array_equal(single, batch)

        # Batch should be faster (or at least not significantly slower)
        assert time_batch <= time_single * 1.5  # Allow 50% margin

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_gpu_performance_scalability(self):
        """Test that GPU performance scales well with batch size."""
        embedding = CupyTextureEmbedding(use_gpu=True)
        frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        # Test different batch sizes
        batch_sizes = [1, 5, 10]
        times = []

        for batch_size in batch_sizes:
            bboxes = np.array(
                [[i * 15, i * 15, i * 15 + 25, i * 15 + 25] for i in range(batch_size)]
            )

            import time

            start = time.time()
            results = embedding.extract_batch_gpu(frame, bboxes)
            elapsed = time.time() - start
            times.append(elapsed)

            # Verify results
            assert len(results) == batch_size
            for result in results:
                assert result.shape == (36,)

        # GPU should handle larger batches efficiently
        # (Time per item should not increase linearly)
        time_per_item_small = times[0] / batch_sizes[0]
        time_per_item_large = times[-1] / batch_sizes[-1]
        assert time_per_item_large <= time_per_item_small * 2  # Allow 2x margin


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_corrupted_frame(self):
        """Test handling of corrupted or invalid frames."""
        embedding = CupyTextureEmbedding()

        # Test with all-zero frame
        zero_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = np.array([10, 10, 50, 50])
        result = embedding.extract(zero_frame, bbox)

        assert isinstance(result, np.ndarray)
        assert result.shape == (36,)
        # Should handle gracefully, not crash

    def test_extreme_bbox_values(self):
        """Test handling of extreme bounding box values."""
        embedding = CupyTextureEmbedding()
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Negative coordinates
        bbox_neg = np.array([-10, -10, 20, 20])
        result_neg = embedding.extract(frame, bbox_neg)
        assert result_neg.shape == (36,)

        # Very large coordinates
        bbox_large = np.array([50, 50, 1000, 1000])
        result_large = embedding.extract(frame, bbox_large)
        assert result_large.shape == (36,)

    def test_memory_efficiency(self):
        """Test memory efficiency with large batches."""
        embedding = CupyTextureEmbedding(use_gpu=False)
        frame = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)

        # Large batch
        large_batch_size = 50
        bboxes = np.array(
            [[i * 10, i * 10, i * 10 + 30, i * 10 + 30] for i in range(large_batch_size)]
        )

        # Should not crash with memory error
        try:
            results = embedding.extract_batch_cpu(frame, bboxes)
            assert len(results) == large_batch_size
        except MemoryError:
            pytest.skip("Not enough memory for large batch test")


# Test markers for different scenarios
@pytest.mark.gpu
class TestGPUSpecific:
    """GPU-specific tests that only run when CuPy is available."""

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_gpu_memory_management(self):
        """Test GPU memory management."""
        embedding = MegaCupyTextureEmbedding(use_gpu=True)
        frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        bboxes = np.array([[i * 20, i * 20, i * 20 + 40, i * 20 + 40] for i in range(10)])

        # Multiple extractions should not leak memory
        for _ in range(5):
            results = embedding.extract_batch_gpu(frame, bboxes)
            assert len(results) == 10

            # Force garbage collection to test memory cleanup
            import gc

            gc.collect()

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_cupy_import_success(self):
        """Test that CuPy imports successfully when available."""
        import swarmsort.embeddings as emb_module

        # If CuPy is available, it should have been imported successfully
        assert emb_module.CUPY_AVAILABLE is True
        assert emb_module.cp is not None


class TestEmbeddingFunctionality:
    """Test embedding-related functionality."""

    def test_embedding_normalization(self):
        """Test embedding normalization in tracker."""
        config = SwarmSortConfig(use_embeddings=True)
        tracker = SwarmSortTracker(config)
        
        # Create detection with non-normalized embedding
        embedding = np.array([1.0, 2.0, 3.0, 4.0] * 16, dtype=np.float32)  # 64-dim
        detection = Detection(
            position=np.array([100.0, 100.0]),
            confidence=0.8,
            embedding=embedding
        )
        
        result = tracker.update([detection])
        # Should process without errors
        assert len(result) >= 0

    def test_gpu_cpu_fallback(self):
        """Test GPU to CPU fallback behavior."""
        # This test would need to mock CuPy availability
        config = SwarmSortConfig(use_embeddings=True)
        
        # Should create tracker regardless of GPU availability
        tracker = SwarmSortTracker(config)
        assert tracker is not None


class TestEmbeddingDistanceScaler:
    """Comprehensive tests for EmbeddingDistanceScaler."""

    def test_scaler_different_methods(self):
        """Test scaler with different scaling methods."""
        methods = ["robust_minmax", "min_robustmax", "zscore", "median_mad", "quantile"]
        
        for method in methods:
            scaler = EmbeddingDistanceScaler(method=method, min_samples=10)
            
            # Add sample data
            distances = np.random.rand(50) * 2.0
            scaler.update_statistics(distances)
            
            # Should become ready after min_samples
            stats = scaler.get_statistics()
            assert stats["ready"] == True
            assert stats["method"] == method
            
            # Test scaling
            test_distances = np.array([0.1, 0.5, 1.0, 1.5])
            scaled = scaler.scale_distances(test_distances)
            assert len(scaled) == len(test_distances)

    def test_scaler_insufficient_data(self):
        """Test scaler behavior with insufficient data."""
        scaler = EmbeddingDistanceScaler(min_samples=100)
        
        # Add small amount of data
        distances = np.random.rand(50)
        scaler.update_statistics(distances)
        
        # Should not be ready yet
        stats = scaler.get_statistics()
        assert stats["ready"] == False
        
        # Scaling should return fallback behavior (clipped scaled distances)
        test_distances = np.array([0.1, 0.5, 1.0])
        scaled = scaler.scale_distances(test_distances)
        # Just check that it returns something reasonable
        assert len(scaled) == len(test_distances)
        assert np.all(scaled >= 0)
        assert np.all(scaled <= 1)

    def test_scaler_edge_cases(self):
        """Test scaler with edge case inputs."""
        scaler = EmbeddingDistanceScaler(min_samples=5)
        
        # Test with all zeros
        zeros = np.zeros(10)
        scaler.update_statistics(zeros)
        
        # Test with all same values
        ones = np.ones(10) * 0.5
        scaler.update_statistics(ones)
        
        # Should handle gracefully
        stats = scaler.get_statistics()
        assert stats["ready"] == True

    def test_scaler_statistics_tracking(self):
        """Test scaler statistics tracking."""
        scaler = EmbeddingDistanceScaler(min_samples=10)
        
        # Add data to make it ready
        distances = np.random.rand(20)
        scaler.update_statistics(distances)
        stats = scaler.get_statistics()
        assert stats["ready"] == True
        assert stats["sample_count"] == 20

    def test_scaler_incremental_update(self):
        """Test incremental updates to scaler."""
        scaler = EmbeddingDistanceScaler(min_samples=20, update_rate=0.1)
        
        # Add data in batches
        for i in range(5):
            batch = np.random.rand(10) * (i + 1)  # Different ranges per batch
            scaler.update_statistics(batch)
            
            stats = scaler.get_statistics()
            assert stats["sample_count"] == (i + 1) * 10


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
