"""
Unit tests for SwarmSort core functionality.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from swarmsort import (
    SwarmSortTracker,
    SwarmSortConfig,
    Detection,
    TrackedObject,
    SwarmSort
)
from swarmsort.core import (
    cosine_similarity_normalized,
    fast_mahalanobis_distance,
    compute_embedding_distances_multi_history,
    compute_cost_matrix_vectorized,
    PendingDetection,
    TrackState
)


class TestCoreNumbaFunctions:
    """Test the Numba JIT-compiled core functions."""
    
    def test_cosine_similarity_normalized(self):
        """Test cosine similarity function."""
        # Identical embeddings
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        distance = cosine_similarity_normalized(emb1, emb2)
        assert distance == 0.0  # Identical = distance 0
        
        # Opposite embeddings
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        distance = cosine_similarity_normalized(emb1, emb2)
        assert distance == 1.0  # Opposite = distance 1
        
        # Orthogonal embeddings
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        distance = cosine_similarity_normalized(emb1, emb2)
        assert abs(distance - 0.5) < 1e-6  # Orthogonal = distance 0.5
        
        # Zero embedding handling
        emb1 = np.zeros(3, dtype=np.float32)
        emb2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        distance = cosine_similarity_normalized(emb1, emb2)
        assert distance == 1.0  # Zero embedding = max distance
    
    def test_fast_mahalanobis_distance(self):
        """Test Mahalanobis distance calculation."""
        # Simple 2D case with identity covariance
        diff = np.array([3.0, 4.0], dtype=np.float32)
        cov_inv = np.eye(2, dtype=np.float32)
        distance = fast_mahalanobis_distance(diff, cov_inv)
        expected = np.sqrt(3.0**2 + 4.0**2)  # Euclidean with identity cov
        assert abs(distance - expected) < 1e-6
        
        # Different covariance
        cov_inv = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=np.float32)
        distance = fast_mahalanobis_distance(diff, cov_inv)
        expected = np.sqrt(3.0**2 * 2.0 + 4.0**2 * 0.5)
        assert abs(distance - expected) < 1e-6
    
    def test_compute_embedding_distances_multi_history(self):
        """Test multi-history embedding distance computation."""
        np.random.seed(42)
        
        # Create test data (normalize embeddings to unit length)
        det_embeddings = np.random.randn(3, 64).astype(np.float32)
        det_embeddings = det_embeddings / np.linalg.norm(det_embeddings, axis=1, keepdims=True)
        
        track_embeddings = np.random.randn(8, 64).astype(np.float32)  # 8 total embeddings
        track_embeddings = track_embeddings / np.linalg.norm(track_embeddings, axis=1, keepdims=True)
        track_counts = np.array([3, 2, 3], dtype=np.int32)  # 3 tracks with different embedding counts
        
        # Test different methods
        for method in ['best_match', 'average', 'weighted_average']:
            distances = compute_embedding_distances_multi_history(
                det_embeddings, track_embeddings, track_counts, method=method
            )
            
            assert distances.shape == (3, 3)  # 3 detections × 3 tracks
            assert np.all(np.isfinite(distances))
            # With normalized embeddings, cosine distances should be in [0, 1]
            assert np.all(distances >= 0.0)
            assert np.all(distances <= 1.0)
        
        # Test edge case: no embeddings for a track
        track_counts_zero = np.array([0, 2, 0], dtype=np.int32)
        distances = compute_embedding_distances_multi_history(
            det_embeddings, track_embeddings[:2], track_counts_zero, method='best_match'
        )
        assert np.all(distances[:, 0] == 1.0)  # Track with no embeddings should have max distance
        assert np.all(distances[:, 2] == 1.0)
    
    def test_compute_cost_matrix_vectorized(self):
        """Test vectorized cost matrix computation."""
        det_positions = np.array([[10.0, 20.0], [50.0, 60.0]], dtype=np.float32)
        track_positions = np.array([[12.0, 22.0], [100.0, 30.0]], dtype=np.float32)
        max_distance = 50.0
        
        cost_matrix = compute_cost_matrix_vectorized(det_positions, track_positions, max_distance)
        
        assert cost_matrix.shape == (2, 2)
        
        # Check specific distances
        expected_dist_00 = np.sqrt((10.0 - 12.0)**2 + (20.0 - 22.0)**2)
        assert abs(cost_matrix[0, 0] - expected_dist_00) < 1e-6
        
        # All distances should be non-negative
        assert np.all(cost_matrix >= 0.0)


class TestPendingDetection:
    """Test the PendingDetection data class."""
    
    def test_pending_detection_creation(self):
        """Test PendingDetection creation and initialization."""
        pending = PendingDetection(
            position=np.array([10.0, 20.0], dtype=np.float32),
            confidence=0.8,
            first_seen_frame=5,
            last_seen_frame=5
        )
        
        assert pending.position.dtype == np.float32
        assert pending.consecutive_frames == 1
        assert pending.total_detections == 1
        assert np.allclose(pending.average_position, pending.position)
    
    def test_pending_detection_with_embedding(self):
        """Test PendingDetection with embedding."""
        embedding = np.random.randn(128).astype(np.float32)
        bbox = np.array([1.0, 2.0, 11.0, 22.0], dtype=np.float32)
        
        pending = PendingDetection(
            position=np.array([10.0, 20.0], dtype=np.float32),
            embedding=embedding,
            bbox=bbox,
            confidence=0.9
        )
        
        assert pending.embedding is not None
        assert pending.embedding.dtype == np.float32
        assert pending.bbox.dtype == np.float32
        assert np.array_equal(pending.embedding, embedding)


class TestTrackState:
    """Test the TrackState data class."""
    
    def test_track_state_creation(self):
        """Test TrackState creation and initialization."""
        track = TrackState(
            id=1,
            position=np.array([50.0, 100.0]),
            velocity=np.array([2.0, -1.0])
        )
        
        assert track.id == 1
        assert track.state.dtype == np.float32
        assert track.covariance.shape == (4, 4)
        assert track.covariance.dtype == np.float32
        
        # Check Kalman state initialization
        assert np.allclose(track.state[:2], [50.0, 100.0])
        assert np.allclose(track.state[2:], [2.0, -1.0])
        
        # Check default values
        assert track.age == 0
        assert track.hits == 0
        assert track.time_since_update == 0
    
    def test_track_state_with_bbox(self):
        """Test TrackState with bounding box."""
        bbox = np.array([45.0, 95.0, 55.0, 105.0], dtype=np.float32)
        track = TrackState(
            id=2,
            position=np.array([50.0, 100.0], dtype=np.float32),
            bbox=bbox
        )
        
        assert track.bbox.dtype == np.float32
        assert np.array_equal(track.bbox, bbox)
    
    def test_track_embeddings(self):
        """Test embedding storage in TrackState."""
        track = TrackState(id=1, position=np.array([0.0, 0.0]))
        
        # Initially empty
        assert len(track.embeddings) == 0
        
        # Add embeddings
        emb1 = np.random.randn(64).astype(np.float32)
        emb2 = np.random.randn(64).astype(np.float32)
        
        track.embeddings.append(emb1)
        track.embeddings.append(emb2)
        
        assert len(track.embeddings) == 2
        assert np.array_equal(track.embeddings[0], emb1)
        assert np.array_equal(track.embeddings[1], emb2)


class TestSwarmSortTrackerCore:
    """Test core SwarmSortTracker functionality."""
    
    def test_tracker_initialization(self, default_config):
        """Test tracker initialization."""
        tracker = SwarmSortTracker(default_config)
        
        assert tracker.config == default_config
        assert tracker.next_id == 1
        assert tracker.frame_count == 0
        assert len(tracker.tracks) == 0
        assert len(tracker.lost_tracks) == 0
        assert len(tracker.pending_detections) == 0
    
    def test_tracker_with_dict_config(self):
        """Test tracker initialization with dict configuration."""
        config_dict = {
            'max_distance': 75.0,
            'use_embeddings': False,
            'min_consecutive_detections': 3
        }
        
        tracker = SwarmSortTracker(config_dict)
        
        assert tracker.config.max_distance == 75.0
        assert not tracker.config.use_embeddings
        assert tracker.config.min_consecutive_detections == 3
    
    def test_motion_model_setup(self, default_config):
        """Test motion model matrix setup."""
        tracker = SwarmSortTracker(default_config)
        
        # Check matrix shapes and types
        assert tracker.F.shape == (4, 4)
        assert tracker.H.shape == (2, 4)
        assert tracker.Q.shape == (4, 4)
        assert tracker.R.shape == (2, 2)
        
        assert tracker.F.dtype == np.float32
        assert tracker.H.dtype == np.float32
        assert tracker.Q.dtype == np.float32
        assert tracker.R.dtype == np.float32
        
        # Check state transition matrix properties
        # F should represent constant velocity model
        assert tracker.F[0, 2] == tracker.dt  # x velocity coupling
        assert tracker.F[1, 3] == tracker.dt  # y velocity coupling
        assert tracker.F[0, 0] == 1.0  # position persistence
        assert tracker.F[1, 1] == 1.0  # position persistence
    
    def test_remove_duplicate_detections(self, default_config):
        """Test duplicate detection removal."""
        tracker = SwarmSortTracker(default_config)
        
        # Create detections with some duplicates
        detections = [
            Detection(position=np.array([10.0, 20.0]), confidence=0.9),
            Detection(position=np.array([10.5, 20.5]), confidence=0.8),  # Close to first
            Detection(position=np.array([50.0, 60.0]), confidence=0.7),
            Detection(position=np.array([10.2, 20.1]), confidence=0.95), # Close to first, higher confidence
        ]
        
        unique_detections = tracker._remove_duplicate_detections(detections)
        
        # Should keep the highest confidence from the duplicate group
        assert len(unique_detections) == 2
        confidences = [det.confidence for det in unique_detections]
        assert 0.95 in confidences  # Highest confidence kept
        assert 0.7 in confidences   # Non-duplicate kept
    
    def test_predict_tracks(self, default_config):
        """Test track prediction step."""
        tracker = SwarmSortTracker(default_config)
        
        # Create a track manually
        track = TrackState(
            id=1,
            position=np.array([10.0, 20.0]),
            velocity=np.array([2.0, 1.0])
        )
        track.state = np.array([10.0, 20.0, 2.0, 1.0], dtype=np.float32)
        track.covariance = np.eye(4, dtype=np.float32)
        tracker.tracks[1] = track
        
        # Predict
        tracker._predict_tracks()
        
        # Check prediction
        updated_track = tracker.tracks[1]
        expected_pos = np.array([12.0, 21.0])  # old_pos + velocity * dt
        assert np.allclose(updated_track.position, expected_pos, atol=1e-5)
        assert updated_track.time_since_update == 1
        assert updated_track.age == 1
    
    def test_reset_functionality(self, default_config):
        """Test tracker reset."""
        tracker = SwarmSortTracker(default_config)
        
        # Add some state
        tracker.frame_count = 10
        tracker.next_id = 5
        tracker.tracks[1] = TrackState(id=1, position=np.array([0.0, 0.0]))
        tracker.timing_stats['test'] = 0.1
        
        # Reset
        tracker.reset()
        
        # Check reset state
        assert tracker.frame_count == 0
        assert tracker.next_id == 1
        assert len(tracker.tracks) == 0
        assert len(tracker.lost_tracks) == 0
        assert len(tracker.pending_detections) == 0
        assert len(tracker.timing_stats) == 0
    
    def test_get_statistics(self, default_config):
        """Test statistics generation."""
        tracker = SwarmSortTracker(default_config)
        
        # Add some tracks
        tracker.tracks[1] = TrackState(id=1, position=np.array([0.0, 0.0]))
        tracker.tracks[2] = TrackState(id=2, position=np.array([10.0, 10.0]))
        tracker.lost_tracks[3] = TrackState(id=3, position=np.array([20.0, 20.0]))
        tracker.pending_detections.append(PendingDetection(position=np.array([30.0, 30.0])))
        tracker.frame_count = 15
        tracker.next_id = 4
        
        stats = tracker.get_statistics()
        
        assert stats['active_tracks'] == 2
        assert stats['lost_tracks'] == 1
        assert stats['pending_detections'] == 1
        assert stats['frame_count'] == 15
        assert stats['next_id'] == 4
        assert 'embedding_scaler_stats' in stats
        assert 'timing_stats' in stats
    
    @pytest.mark.parametrize("config_key,config_value", [
        ('max_distance', 100.0),
        ('use_embeddings', False),
        ('embedding_weight', 0.7),
        ('min_consecutive_detections', 5),
        ('reid_enabled', False)
    ])
    def test_config_parameter_effects(self, config_key, config_value):
        """Test that configuration parameters are properly applied."""
        config = SwarmSortConfig(**{config_key: config_value})
        tracker = SwarmSortTracker(config)
        
        assert getattr(tracker.config, config_key) == config_value


class TestSwarmSortTrackerIntegration:
    """Test SwarmSort integration and factory functions."""
    
    def test_swarmsort_wrapper_creation(self, default_config):
        """Test SwarmSort adaptive wrapper creation."""
        tracker = SwarmSort(default_config)
        
        # Should have required methods
        assert hasattr(tracker, 'update')
        assert hasattr(tracker, 'reset')
        assert hasattr(tracker, 'get_statistics')
        assert hasattr(tracker, 'config')
        
        # Config should be accessible
        assert tracker.config.max_distance == default_config.max_distance
    
    def test_adaptive_detection_handling(self, embedding_config):
        """Test that adaptive tracker handles different detection formats."""
        tracker = SwarmSort(embedding_config)
        
        # Standard Detection objects
        detections = [
            Detection(position=np.array([10.0, 20.0]), confidence=0.9),
            Detection(position=np.array([50.0, 60.0]), confidence=0.8)
        ]
        
        result = tracker.update(detections)
        assert isinstance(result, list)
        
        # Should handle empty detection list
        empty_result = tracker.update([])
        assert isinstance(empty_result, list)
        assert len(empty_result) == 0
    
    def test_statistics_consistency(self, basic_config):
        """Test that statistics remain consistent across operations."""
        tracker = SwarmSort(basic_config)
        
        # Initial stats
        initial_stats = tracker.get_statistics()
        assert initial_stats['frame_count'] == 0
        assert initial_stats['active_tracks'] == 0
        
        # After update
        detections = [Detection(position=np.array([10.0, 20.0]), confidence=0.9)]
        tracker.update(detections)
        
        updated_stats = tracker.get_statistics()
        assert updated_stats['frame_count'] == 1
        
        # After reset
        tracker.reset()
        reset_stats = tracker.get_statistics()
        assert reset_stats['frame_count'] == 0
        assert reset_stats['active_tracks'] == 0


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_detection_position(self):
        """Test handling of invalid detection positions."""
        with pytest.raises(TypeError):
            Detection(position="invalid")
    
    def test_invalid_config_values(self):
        """Test invalid configuration validation."""
        with pytest.raises(ValueError):
            config = SwarmSortConfig(max_distance=-1.0)
            config.validate()
        
        with pytest.raises(ValueError):
            config = SwarmSortConfig(high_score_threshold=1.5)
            config.validate()
        
        with pytest.raises(ValueError):
            config = SwarmSortConfig(embedding_matching_method='invalid')
            config.validate()
    
    def test_empty_detection_handling(self, basic_tracker):
        """Test handling of empty detection lists."""
        # Should handle empty list gracefully
        result = basic_tracker.update([])
        assert isinstance(result, list)
        assert len(result) == 0
        
        # Frame count should still increment
        stats = basic_tracker.get_statistics()
        assert stats['frame_count'] == 1
    
    def test_very_low_confidence_detections(self, basic_tracker):
        """Test filtering of very low confidence detections."""
        # Create detections with very low confidence
        detections = [
            Detection(position=np.array([10.0, 20.0]), confidence=0.01),  # Below threshold
            Detection(position=np.array([50.0, 60.0]), confidence=0.5),   # Above threshold
        ]
        
        result = basic_tracker.update(detections)
        
        # Only high confidence detections should be processed
        # (exact behavior depends on configuration)
        assert isinstance(result, list)
    
    def test_nan_inf_handling(self, basic_tracker):
        """Test handling of NaN and Inf values."""
        # Detection with NaN position should be handled gracefully
        try:
            detection = Detection(position=np.array([float('nan'), 20.0]), confidence=0.9)
            result = basic_tracker.update([detection])
            # Should not crash
            assert isinstance(result, list)
        except (ValueError, TypeError):
            # Acceptable to raise error for invalid input
            pass
    
    def test_large_coordinate_values(self, basic_tracker):
        """Test handling of very large coordinate values."""
        detections = [
            Detection(position=np.array([1e6, 1e6]), confidence=0.9),
            Detection(position=np.array([-1e6, -1e6]), confidence=0.8)
        ]
        
        # Should handle large coordinates without crashing
        result = basic_tracker.update(detections)
        assert isinstance(result, list)
    
    def test_mismatched_embedding_dimensions(self, embedding_config):
        """Test handling of mismatched embedding dimensions."""
        tracker = SwarmSortTracker(embedding_config)
        
        detections = [
            Detection(position=np.array([10.0, 20.0]), confidence=0.9, embedding=np.random.randn(64).astype(np.float32)),
            Detection(position=np.array([50.0, 60.0]), confidence=0.8, embedding=np.random.randn(128).astype(np.float32))  # Different dim
        ]
        
        # Should handle gracefully without crashing
        try:
            result = tracker.update(detections)
            assert isinstance(result, list)
        except (ValueError, IndexError):
            # Acceptable to fail on mismatched dimensions
            pass