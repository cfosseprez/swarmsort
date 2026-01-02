"""
Tests for Cost Computation Module

This module validates the cost computation functions including:
- Spatial distance computation
- Embedding distance computation
- Cost matrix construction
- Covariance estimation
- Probabilistic cost matrix
- Deduplication mask computation
- Freeze flags computation
"""

import numpy as np
import pytest
from src.swarmsort.cost_computation import (
    estimate_track_covariances,
    cosine_similarity_normalized,
    compute_embedding_distances_with_method,
    compute_cost_matrix_vectorized,
    compute_probabilistic_cost_matrix_vectorized,
    compute_freeze_flags_vectorized,
    compute_deduplication_mask,
)


class TestCovarianceEstimation:
    """Tests for track covariance estimation."""

    def test_isotropic_covariance_for_stationary_tracks(self):
        """Test that stationary tracks get isotropic covariance."""
        # Zero velocity tracks
        track_velocities = np.array([[0.0, 0.0], [0.01, 0.0]], dtype=np.float32)

        covariances = estimate_track_covariances(
            track_velocities, base_variance=5.0, velocity_scale=2.0
        )

        # Should be isotropic (diagonal with equal values)
        assert covariances.shape == (2, 2, 2)

        # First track: exactly zero velocity
        np.testing.assert_allclose(covariances[0, 0, 0], 5.0, rtol=1e-5)
        np.testing.assert_allclose(covariances[0, 1, 1], 5.0, rtol=1e-5)
        np.testing.assert_allclose(covariances[0, 0, 1], 0.0, atol=1e-5)

    def test_anisotropic_covariance_for_moving_tracks(self):
        """Test that moving tracks get anisotropic covariance."""
        # Track moving in x direction
        track_velocities = np.array([[10.0, 0.0]], dtype=np.float32)

        covariances = estimate_track_covariances(
            track_velocities, base_variance=5.0, velocity_scale=2.0
        )

        # Variance along velocity (x) should be higher
        assert covariances[0, 0, 0] > covariances[0, 1, 1]

    def test_covariance_positive_definite(self):
        """Test that all covariances are positive semi-definite."""
        np.random.seed(42)
        track_velocities = np.random.randn(10, 2).astype(np.float32) * 5

        covariances = estimate_track_covariances(track_velocities)

        for i in range(10):
            cov = covariances[i]
            # Check symmetric
            np.testing.assert_allclose(cov, cov.T, rtol=1e-5)
            # Check positive semi-definite (eigenvalues >= 0)
            eigenvalues = np.linalg.eigvalsh(cov)
            assert np.all(eigenvalues >= -1e-6)  # Allow small numerical errors

    def test_covariance_scales_with_parameters(self):
        """Test that covariance scales properly with parameters."""
        track_velocities = np.array([[5.0, 5.0]], dtype=np.float32)

        # Double base variance should double the covariance
        cov1 = estimate_track_covariances(track_velocities, base_variance=5.0)
        cov2 = estimate_track_covariances(track_velocities, base_variance=10.0)

        # The ratio should be approximately 2 for the base component
        assert cov2[0, 0, 0] > cov1[0, 0, 0]


class TestEmbeddingDistance:
    """Tests for embedding distance computation."""

    def test_cosine_similarity_identical_vectors(self):
        """Test that identical vectors have zero distance."""
        emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        distance = cosine_similarity_normalized(emb, emb)
        np.testing.assert_allclose(distance, 0.0, atol=1e-5)

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test that orthogonal vectors have distance 0.5."""
        emb1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        distance = cosine_similarity_normalized(emb1, emb2)
        np.testing.assert_allclose(distance, 0.5, atol=1e-5)

    def test_cosine_similarity_opposite_vectors(self):
        """Test that opposite vectors have distance 1.0."""
        emb1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        distance = cosine_similarity_normalized(emb1, emb2)
        np.testing.assert_allclose(distance, 1.0, atol=1e-5)

    def test_cosine_similarity_range(self):
        """Test that cosine similarity distance is in [0, 1]."""
        np.random.seed(42)
        for _ in range(100):
            emb1 = np.random.randn(128).astype(np.float32)
            emb2 = np.random.randn(128).astype(np.float32)
            emb1 = emb1 / np.linalg.norm(emb1)
            emb2 = emb2 / np.linalg.norm(emb2)

            distance = cosine_similarity_normalized(emb1, emb2)
            assert 0.0 <= distance <= 1.0 + 1e-5

    def test_embedding_distances_with_method_average(self):
        """Test batch embedding distance computation with average method."""
        # 2 detections, 3 tracks with 1 embedding each
        det_embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float32)

        # 1D flattened track embeddings (3 tracks, 1 embedding each, 4 dims)
        # All embeddings concatenated: track0_emb0 | track1_emb0 | track2_emb0
        track_embeddings_flat = np.array([
            1.0, 0.0, 0.0, 0.0,  # Track 0 - same as det 0
            0.0, 1.0, 0.0, 0.0,  # Track 1 - same as det 1
            0.0, 0.0, 1.0, 0.0,  # Track 2 - different
        ], dtype=np.float32)

        # Each track has 1 embedding
        track_counts = np.array([1, 1, 1], dtype=np.int32)

        # Method: 1 = average
        distances = compute_embedding_distances_with_method(
            det_embeddings, track_embeddings_flat, track_counts, method=1
        )

        assert distances.shape == (2, 3)
        # det 0 should match track 0 well (distance ~ 0)
        assert distances[0, 0] < distances[0, 2]
        # det 1 should match track 1 well (distance ~ 0)
        assert distances[1, 1] < distances[1, 2]


class TestCostMatrixComputation:
    """Tests for cost matrix computation."""

    def test_cost_matrix_basic_spatial(self):
        """Test basic spatial cost matrix without embeddings."""
        det_positions = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
        track_positions = np.array([[1.0, 0.0], [11.0, 0.0]], dtype=np.float32)
        track_last_positions = track_positions.copy()
        track_misses = np.array([0, 0], dtype=np.int32)
        embedding_distances = np.zeros((2, 2), dtype=np.float32)

        cost_matrix = compute_cost_matrix_vectorized(
            det_positions, track_positions, track_last_positions,
            track_misses, embedding_distances,
            embedding_weight=0.0, max_distance=50.0, do_embeddings=False
        )

        assert cost_matrix.shape == (2, 2)
        # det 0 should be close to track 0 (distance 1)
        np.testing.assert_allclose(cost_matrix[0, 0], 1.0, atol=0.1)
        # det 1 should be close to track 1 (distance 1)
        np.testing.assert_allclose(cost_matrix[1, 1], 1.0, atol=0.1)

    def test_cost_matrix_max_distance_gating(self):
        """Test that distances beyond max_distance are set to infinity."""
        det_positions = np.array([[0.0, 0.0]], dtype=np.float32)
        track_positions = np.array([[100.0, 0.0], [10.0, 0.0]], dtype=np.float32)
        track_last_positions = track_positions.copy()
        track_misses = np.array([0, 0], dtype=np.int32)
        embedding_distances = np.zeros((1, 2), dtype=np.float32)

        cost_matrix = compute_cost_matrix_vectorized(
            det_positions, track_positions, track_last_positions,
            track_misses, embedding_distances,
            embedding_weight=0.0, max_distance=50.0, do_embeddings=False
        )

        # Track at 100 pixels should be gated out
        assert np.isinf(cost_matrix[0, 0])
        # Track at 10 pixels should be valid
        assert not np.isinf(cost_matrix[0, 1])

    def test_cost_matrix_embedding_weight(self):
        """Test that embedding weight properly blends spatial and appearance."""
        det_positions = np.array([[0.0, 0.0]], dtype=np.float32)
        track_positions = np.array([[10.0, 0.0], [10.0, 0.0]], dtype=np.float32)
        track_last_positions = track_positions.copy()
        track_misses = np.array([0, 0], dtype=np.int32)

        # Track 0 has good embedding match, track 1 has poor
        embedding_distances = np.array([[0.1, 0.9]], dtype=np.float32)

        cost_matrix = compute_cost_matrix_vectorized(
            det_positions, track_positions, track_last_positions,
            track_misses, embedding_distances,
            embedding_weight=0.5, max_distance=50.0, do_embeddings=True
        )

        # With embedding weight 0.5, track 0 should have lower cost
        assert cost_matrix[0, 0] < cost_matrix[0, 1]

    def test_cost_matrix_miss_threshold_behavior(self):
        """Test that miss threshold switches to last detection position."""
        det_positions = np.array([[0.0, 0.0]], dtype=np.float32)
        # Predicted position far away
        track_positions = np.array([[100.0, 0.0]], dtype=np.float32)
        # Last detection position close
        track_last_positions = np.array([[5.0, 0.0]], dtype=np.float32)
        # High misses - should use last detection position
        track_misses = np.array([10], dtype=np.int32)
        embedding_distances = np.zeros((1, 1), dtype=np.float32)

        cost_matrix = compute_cost_matrix_vectorized(
            det_positions, track_positions, track_last_positions,
            track_misses, embedding_distances,
            embedding_weight=0.0, max_distance=50.0, do_embeddings=False,
            miss_threshold=3
        )

        # Should use last_detection (5 px) not predicted (100 px)
        assert cost_matrix[0, 0] < 50.0  # Should be close to 5, not 100


class TestProbabilisticCostMatrix:
    """Tests for probabilistic cost matrix computation."""

    def test_probabilistic_cost_basic(self):
        """Test basic probabilistic cost computation."""
        det_positions = np.array([[0.0, 0.0]], dtype=np.float32)
        track_positions = np.array([[5.0, 0.0]], dtype=np.float32)
        track_last_positions = track_positions.copy()
        track_misses = np.array([0], dtype=np.int32)
        track_covariances = np.array([[[10.0, 0.0], [0.0, 10.0]]], dtype=np.float32)
        embedding_distances = np.zeros((1, 1), dtype=np.float32)

        cost_matrix = compute_probabilistic_cost_matrix_vectorized(
            det_positions, track_positions, track_last_positions,
            track_misses, track_covariances, embedding_distances,
            embedding_weight=0.0, max_distance=50.0, do_embeddings=False
        )

        assert cost_matrix.shape == (1, 1)
        assert not np.isinf(cost_matrix[0, 0])
        assert cost_matrix[0, 0] >= 0

    def test_probabilistic_cost_uncertainty_increases_tolerance(self):
        """Test that higher uncertainty allows matches at greater distances."""
        det_positions = np.array([[0.0, 0.0]], dtype=np.float32)
        track_positions = np.array([[30.0, 0.0]], dtype=np.float32)
        track_last_positions = track_positions.copy()
        track_misses = np.array([0], dtype=np.int32)
        embedding_distances = np.zeros((1, 1), dtype=np.float32)

        # Low uncertainty covariance
        low_cov = np.array([[[5.0, 0.0], [0.0, 5.0]]], dtype=np.float32)
        # High uncertainty covariance
        high_cov = np.array([[[50.0, 0.0], [0.0, 50.0]]], dtype=np.float32)

        cost_low = compute_probabilistic_cost_matrix_vectorized(
            det_positions, track_positions, track_last_positions,
            track_misses, low_cov, embedding_distances,
            embedding_weight=0.0, max_distance=100.0, do_embeddings=False
        )

        cost_high = compute_probabilistic_cost_matrix_vectorized(
            det_positions, track_positions, track_last_positions,
            track_misses, high_cov, embedding_distances,
            embedding_weight=0.0, max_distance=100.0, do_embeddings=False
        )

        # Higher uncertainty should result in lower cost for same distance
        assert cost_high[0, 0] < cost_low[0, 0]


class TestDeduplicationMask:
    """Tests for detection deduplication."""

    def test_deduplication_keeps_highest_confidence(self):
        """Test that deduplication keeps highest confidence detection."""
        positions = np.array([
            [0.0, 0.0],
            [1.0, 0.0],  # Very close to first
            [100.0, 0.0]  # Far away
        ], dtype=np.float32)
        confidences = np.array([0.5, 0.9, 0.7], dtype=np.float32)

        mask = compute_deduplication_mask(positions, confidences, dedup_distance=5.0)

        # Detection 1 (conf 0.9) should be kept, detection 0 (conf 0.5) removed
        assert mask[1] == True  # Highest confidence in cluster
        assert mask[0] == False  # Lower confidence, close to 1
        assert mask[2] == True  # Far away, kept

    def test_deduplication_respects_distance_threshold(self):
        """Test that detections beyond threshold are kept."""
        positions = np.array([
            [0.0, 0.0],
            [10.0, 0.0],  # At threshold
        ], dtype=np.float32)
        confidences = np.array([0.9, 0.5], dtype=np.float32)

        mask = compute_deduplication_mask(positions, confidences, dedup_distance=5.0)

        # Both should be kept since distance is beyond threshold
        assert mask[0] == True
        assert mask[1] == True

    def test_deduplication_empty_input(self):
        """Test deduplication with empty input."""
        positions = np.zeros((0, 2), dtype=np.float32)
        confidences = np.zeros(0, dtype=np.float32)

        mask = compute_deduplication_mask(positions, confidences, dedup_distance=5.0)
        assert len(mask) == 0

    def test_deduplication_uses_squared_distance(self):
        """Test that deduplication correctly uses squared distance internally."""
        # This is a regression test for the optimization
        positions = np.array([
            [0.0, 0.0],
            [3.0, 4.0],  # Distance is exactly 5
        ], dtype=np.float32)
        confidences = np.array([0.9, 0.5], dtype=np.float32)

        # At exactly threshold distance
        mask = compute_deduplication_mask(positions, confidences, dedup_distance=5.0)
        # Detection at exactly 5 pixels should NOT be suppressed (< not <=)
        assert mask[1] == True


class TestFreezeFlags:
    """Tests for embedding freeze flag computation."""

    def test_freeze_flags_close_tracks(self):
        """Test that close tracks trigger freeze."""
        track_positions = np.array([
            [0.0, 0.0],
            [3.0, 0.0],  # Close to first
            [100.0, 0.0]  # Far away
        ], dtype=np.float32)

        flags = compute_freeze_flags_vectorized(track_positions, safety_distance=10.0)

        # Tracks 0 and 1 should be frozen
        assert flags[0] == True
        assert flags[1] == True
        # Track 2 should not be frozen
        assert flags[2] == False

    def test_freeze_flags_empty_input(self):
        """Test freeze flags with empty input."""
        track_positions = np.zeros((0, 2), dtype=np.float32)
        flags = compute_freeze_flags_vectorized(track_positions, safety_distance=10.0)
        assert len(flags) == 0

    def test_freeze_flags_single_track(self):
        """Test freeze flags with single track."""
        track_positions = np.array([[0.0, 0.0]], dtype=np.float32)
        flags = compute_freeze_flags_vectorized(track_positions, safety_distance=10.0)
        assert flags[0] == False  # No other track to freeze with


class TestCostMatrixEdgeCases:
    """Edge case tests for cost computation."""

    def test_empty_detections(self):
        """Test cost matrix with no detections."""
        det_positions = np.zeros((0, 2), dtype=np.float32)
        track_positions = np.array([[0.0, 0.0]], dtype=np.float32)
        track_last_positions = track_positions.copy()
        track_misses = np.array([0], dtype=np.int32)
        embedding_distances = np.zeros((0, 1), dtype=np.float32)

        cost_matrix = compute_cost_matrix_vectorized(
            det_positions, track_positions, track_last_positions,
            track_misses, embedding_distances,
            embedding_weight=0.0, max_distance=50.0, do_embeddings=False
        )

        assert cost_matrix.shape == (0, 1)

    def test_empty_tracks(self):
        """Test cost matrix with no tracks."""
        det_positions = np.array([[0.0, 0.0]], dtype=np.float32)
        track_positions = np.zeros((0, 2), dtype=np.float32)
        track_last_positions = track_positions.copy()
        track_misses = np.zeros(0, dtype=np.int32)
        embedding_distances = np.zeros((1, 0), dtype=np.float32)

        cost_matrix = compute_cost_matrix_vectorized(
            det_positions, track_positions, track_last_positions,
            track_misses, embedding_distances,
            embedding_weight=0.0, max_distance=50.0, do_embeddings=False
        )

        assert cost_matrix.shape == (1, 0)

    def test_large_scale_performance(self):
        """Test that cost computation handles large matrices efficiently."""
        np.random.seed(42)
        n_dets = 100
        n_tracks = 100

        det_positions = np.random.rand(n_dets, 2).astype(np.float32) * 1000
        track_positions = np.random.rand(n_tracks, 2).astype(np.float32) * 1000
        track_last_positions = track_positions.copy()
        track_misses = np.zeros(n_tracks, dtype=np.int32)
        embedding_distances = np.random.rand(n_dets, n_tracks).astype(np.float32)

        # Should complete without error
        cost_matrix = compute_cost_matrix_vectorized(
            det_positions, track_positions, track_last_positions,
            track_misses, embedding_distances,
            embedding_weight=0.3, max_distance=200.0, do_embeddings=True
        )

        assert cost_matrix.shape == (n_dets, n_tracks)
        # Should have some valid (non-infinite) entries
        assert np.sum(~np.isinf(cost_matrix)) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
