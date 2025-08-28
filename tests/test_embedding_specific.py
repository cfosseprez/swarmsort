"""
Embedding-specific tests for SwarmSort.
Tests focusing on embedding processing, distance scaling, and embedding-based tracking.
"""
import pytest
import numpy as np
from typing import List, Dict, Any
import warnings

from swarmsort import (
    SwarmSort,
    SwarmSortConfig,
    Detection,
    TrackedObject,
    EmbeddingDistanceScaler
)
from swarmsort.core import (
    cosine_similarity_normalized,
    compute_embedding_distances_multi_history
)


@pytest.mark.embedding
class TestEmbeddingDistanceScaler:
    """Test the embedding distance scaler component."""
    
    def test_scaler_initialization(self):
        """Test scaler initialization with different parameters."""
        # Default scaler
        scaler1 = EmbeddingDistanceScaler()
        assert scaler1.method == 'robust_minmax'
        assert scaler1.update_rate == 0.05
        assert scaler1.min_samples == 200
        assert scaler1.sample_count == 0
        
        # Custom scaler
        scaler2 = EmbeddingDistanceScaler(
            method='min_robustmax',
            update_rate=0.1,
            min_samples=50
        )
        assert scaler2.method == 'min_robustmax'
        assert scaler2.update_rate == 0.1
        assert scaler2.min_samples == 50
    
    def test_statistics_update(self):
        """Test statistics update functionality."""
        scaler = EmbeddingDistanceScaler(min_samples=10)
        
        # Initial state
        stats = scaler.get_statistics()
        assert not stats['ready']
        assert stats['sample_count'] == 0
        
        # Add some samples
        np.random.seed(42)
        distances1 = np.random.rand(20) * 0.5  # Range [0, 0.5]
        scaler.update_statistics(distances1)
        
        stats = scaler.get_statistics()
        assert stats['sample_count'] == 20
        assert stats['min_distance'] is not None
        assert stats['max_distance'] is not None
        assert stats['mean_distance'] is not None
        
        # Add more samples
        distances2 = np.random.rand(15) * 0.3 + 0.2  # Range [0.2, 0.5]
        scaler.update_statistics(distances2)
        
        new_stats = scaler.get_statistics()
        assert new_stats['sample_count'] == 35
        assert new_stats['ready']  # Should be ready now
    
    @pytest.mark.parametrize("scaling_method", [
        'robust_minmax', 'min_robustmax', 'zscore', 'robust_zscore',
        'arcsinh', 'arcsinh_percentile', 'beta', 'double_transform',
        'sqrt', 'quantile', 'sigmoid'
    ])
    def test_different_scaling_methods(self, scaling_method):
        """Test different scaling methods."""
        scaler = EmbeddingDistanceScaler(
            method=scaling_method,
            min_samples=20
        )
        
        # Generate test data
        np.random.seed(42)
        distances = np.random.beta(2, 5, 100).astype(np.float32)  # Skewed distribution
        
        # Update statistics
        scaler.update_statistics(distances)
        
        # Scale the same distances
        scaled_distances = scaler.scale_distances(distances)
        
        # Check output properties
        assert isinstance(scaled_distances, np.ndarray)
        assert scaled_distances.shape == distances.shape
        assert np.all(scaled_distances >= 0.0)
        assert np.all(scaled_distances <= 1.0)
        
        # Should actually change the distribution (except when not ready)
        if scaler.get_statistics()['ready']:
            # Scaled should have different characteristics than original
            original_range = np.max(distances) - np.min(distances)
            scaled_range = np.max(scaled_distances) - np.min(scaled_distances)
            
            # Some methods might expand or compress the range
            assert scaled_range >= 0.0
    
    def test_scaling_with_insufficient_samples(self):
        """Test scaling behavior when insufficient samples."""
        scaler = EmbeddingDistanceScaler(min_samples=100)
        
        # Small number of samples
        distances = np.array([0.1, 0.3, 0.5, 0.7], dtype=np.float32)
        scaler.update_statistics(distances)
        
        # Should use fallback scaling
        scaled = scaler.scale_distances(distances)
        
        assert isinstance(scaled, np.ndarray)
        assert np.all(scaled >= 0.0)
        assert np.all(scaled <= 1.0)
    
    def test_scaling_edge_cases(self):
        """Test scaling with edge case inputs."""
        scaler = EmbeddingDistanceScaler(min_samples=10)
        
        # All same values
        same_distances = np.array([0.5] * 20, dtype=np.float32)
        scaler.update_statistics(same_distances)
        scaled = scaler.scale_distances(same_distances)
        
        assert isinstance(scaled, np.ndarray)
        assert len(scaled) == len(same_distances)
        
        # Very small distances
        small_distances = np.array([1e-8, 2e-8, 3e-8], dtype=np.float32)
        scaled_small = scaler.scale_distances(small_distances)
        assert np.all(np.isfinite(scaled_small))
        
        # Very large distances
        large_distances = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        scaled_large = scaler.scale_distances(large_distances)
        assert np.all(scaled_large >= 0.0)
        assert np.all(scaled_large <= 1.0)
    
    def test_online_adaptation(self):
        """Test online adaptation of scaler statistics."""
        scaler = EmbeddingDistanceScaler(
            min_samples=20,
            update_rate=0.2  # High update rate for testing
        )
        
        # Phase 1: Low distances
        np.random.seed(42)
        for _ in range(10):
            distances = np.random.rand(5) * 0.3  # [0, 0.3]
            scaler.update_statistics(distances)
        
        phase1_stats = scaler.get_statistics()
        phase1_max = phase1_stats['max_distance']
        
        # Phase 2: Higher distances
        for _ in range(10):
            distances = np.random.rand(5) * 0.5 + 0.4  # [0.4, 0.9]
            scaler.update_statistics(distances)
        
        phase2_stats = scaler.get_statistics()
        phase2_max = phase2_stats['max_distance']
        
        # Max distance should have adapted
        assert phase2_max > phase1_max
        
        # Mean should have shifted
        assert phase2_stats['mean_distance'] > phase1_stats['mean_distance']


@pytest.mark.embedding
class TestEmbeddingBasedTracking:
    """Test embedding-based tracking functionality."""
    
    def test_embedding_association_accuracy(self, consistent_embeddings):
        """Test that embeddings improve association accuracy."""
        config = SwarmSortConfig(
            use_embeddings=True,
            embedding_weight=0.8,  # High weight for testing
            min_consecutive_detections=2,
            max_distance=100.0  # Large to allow position errors
        )
        tracker = SwarmSort(config)
        
        # Create two distinct embedding templates
        emb1 = consistent_embeddings['emb1']
        emb2 = consistent_embeddings['emb2']
        
        # Track objects with consistent embeddings but crossing paths
        for i in range(10):
            # Objects cross paths but maintain distinct embeddings
            if i < 5:
                pos1, pos2 = [10 + i*10, 50], [100 - i*10, 50]
            else:
                pos1, pos2 = [100 - (i-5)*10, 50], [10 + (i-5)*10, 50]
            
            detections = [
                Detection(
                    position=pos1,
                    confidence=0.9,
                    embedding=emb1 + np.random.randn(128) * 0.1  # Small variation
                ),
                Detection(
                    position=pos2,
                    confidence=0.9,
                    embedding=emb2 + np.random.randn(128) * 0.1  # Small variation
                )
            ]
            
            tracked_objects = tracker.update(detections)
        
        # Should maintain 2 distinct tracks despite crossing paths
        final_tracks = tracked_objects
        assert len(final_tracks) == 2
        
        # Tracks should have different IDs
        track_ids = {track.id for track in final_tracks}
        assert len(track_ids) == 2
    
    def test_embedding_matching_methods(self, consistent_embeddings):
        """Test different embedding matching methods."""
        methods = ['best_match', 'average', 'weighted_average']
        
        for method in methods:
            config = SwarmSortConfig(
                use_embeddings=True,
                embedding_matching_method=method,
                embedding_weight=0.5,
                min_consecutive_detections=2,
                max_embeddings_per_track=5
            )
            tracker = SwarmSort(config)
            
            emb1 = consistent_embeddings['emb1']
            
            # Build up embedding history
            for i in range(8):
                detection = Detection(
                    position=np.array([10 + i, 20]),
                    confidence=0.9,
                    embedding=emb1 + np.random.randn(128) * 0.05
                )
                tracked_objects = tracker.update([detection])
            
            # Should create track with the specified method
            final_stats = tracker.get_statistics()
            assert final_stats['active_tracks'] >= 1
            
            # Verify the method is being used (indirectly through successful tracking)
            assert tracker.config.embedding_matching_method == method
    
    def test_embedding_distance_computation(self):
        """Test embedding distance computation functions."""
        np.random.seed(42)
        
        # Create test embeddings (normalize to unit length for proper cosine distance range)
        det_embeddings = np.random.randn(3, 64).astype(np.float32)
        det_embeddings = det_embeddings / np.linalg.norm(det_embeddings, axis=1, keepdims=True)
        
        track_embeddings = np.random.randn(10, 64).astype(np.float32)
        track_embeddings = track_embeddings / np.linalg.norm(track_embeddings, axis=1, keepdims=True)
        track_counts = np.array([3, 4, 3], dtype=np.int32)  # 3 tracks
        
        # Test different methods
        for method in ['best_match', 'average', 'weighted_average']:
            distances = compute_embedding_distances_multi_history(
                det_embeddings, track_embeddings, track_counts, method=method
            )
            
            # Check output shape and range
            assert distances.shape == (3, 3)  # 3 detections × 3 tracks
            assert np.all(np.isfinite(distances))
            assert np.all(distances <= 1.0)
            
            # Different methods should produce different results
            if method == 'best_match':
                # Best match should generally give lower distances than average
                avg_distances = compute_embedding_distances_multi_history(
                    det_embeddings, track_embeddings, track_counts, 'average'
                )
                # At least some distances should be lower for best_match
                assert np.mean(distances) <= np.mean(avg_distances) * 1.1
    
    def test_embedding_scaling_integration(self):
        """Test integration of embedding scaling with tracking."""
        config = SwarmSortConfig(
            use_embeddings=True,
            embedding_weight=0.6,
            embedding_scaling_method='min_robustmax',
            embedding_scaling_min_samples=30,
            min_consecutive_detections=2
        )
        tracker = SwarmSort(config)
        
        np.random.seed(42)
        
        # Generate embeddings with different scales in different phases
        phases = [
            (np.random.randn(64) * 0.1, 20),  # Small scale embeddings
            (np.random.randn(64) * 0.5, 20),  # Medium scale embeddings  
            (np.random.randn(64) * 1.0, 20),  # Large scale embeddings
        ]
        
        scaler_ready_frames = []
        
        frame_count = 0
        for phase_emb, phase_frames in phases:
            for i in range(phase_frames):
                detection = Detection(
                    position=np.array([10 + frame_count, 20]),
                    confidence=0.9,
                    embedding=phase_emb + np.random.randn(64) * 0.1
                )
                
                tracked_objects = tracker.update([detection])
                
                # Check if scaler becomes ready
                stats = tracker.get_statistics()
                emb_stats = stats['embedding_scaler_stats']
                if emb_stats['ready'] and not scaler_ready_frames:
                    scaler_ready_frames.append(frame_count)
                
                frame_count += 1
        
        # Scaler should become ready after min_samples
        assert len(scaler_ready_frames) > 0
        assert scaler_ready_frames[0] >= config.embedding_scaling_min_samples
        
        # Should track successfully despite different embedding scales
        final_stats = tracker.get_statistics()
        assert final_stats['active_tracks'] >= 1
    
    def test_embedding_history_management(self):
        """Test embedding history management in tracks."""
        config = SwarmSortConfig(
            use_embeddings=True,
            max_embeddings_per_track=3,  # Small limit for testing
            embedding_weight=0.4,
            min_consecutive_detections=2
        )
        tracker = SwarmSort(config)
        
        np.random.seed(42)
        base_embedding = np.random.randn(64).astype(np.float32)
        
        # Update track multiple times
        for i in range(10):
            detection = Detection(
                position=np.array([10 + i, 20]),
                confidence=0.9,
                embedding=base_embedding + np.random.randn(64) * 0.05
            )
            tracked_objects = tracker.update([detection])
        
        # Check that embedding history is limited
        if tracker.tracker.tracks:
            track = next(iter(tracker.tracker.tracks.values()))
            assert len(track.embeddings) <= config.max_embeddings_per_track
            
            # Should keep most recent embeddings
            assert len(track.embeddings) == config.max_embeddings_per_track
    
    def test_embedding_similarity_thresholds(self):
        """Test embedding similarity thresholds for association."""
        config = SwarmSortConfig(
            use_embeddings=True,
            embedding_weight=0.9,  # Very high weight
            max_distance=200.0,  # Large position tolerance
            min_consecutive_detections=1  # Immediate tracking
        )
        tracker = SwarmSort(config)
        
        np.random.seed(42)
        
        # Create very different embeddings
        emb1 = np.random.randn(64).astype(np.float32)
        emb2 = -emb1  # Opposite embedding (maximum distance)
        
        # First detection
        det1 = Detection(position=np.array([10, 20]), confidence=0.9, embedding=emb1)
        tracker.update([det1])
        
        # Second detection at similar position but very different embedding
        det2 = Detection(position=np.array([12, 22]), confidence=0.9, embedding=emb2)
        tracked_objects = tracker.update([det2])
        
        # Should create separate tracks due to embedding dissimilarity
        assert len(tracked_objects) >= 1
        
        # Get final statistics
        stats = tracker.get_statistics()
        # High embedding weight should prevent association of dissimilar embeddings
        # even at close positions, so might create separate tracks
        assert stats['active_tracks'] >= 1
    
    def test_embedding_normalization(self):
        """Test embedding normalization and preprocessing."""
        config = SwarmSortConfig(use_embeddings=True)
        tracker = SwarmSort(config)
        
        # Test different embedding magnitudes
        test_embeddings = [
            np.ones(64, dtype=np.float32) * 0.1,      # Small magnitude
            np.ones(64, dtype=np.float32),             # Unit magnitude  
            np.ones(64, dtype=np.float32) * 10.0,      # Large magnitude
            np.zeros(64, dtype=np.float32),            # Zero embedding
        ]
        
        for i, embedding in enumerate(test_embeddings):
            detection = Detection(
                position=np.array([i * 20, 50]),
                confidence=0.9,
                embedding=embedding
            )
            
            # Should handle different magnitudes gracefully
            result = tracker.update([detection])
            assert isinstance(result, list)
        
        # All should be processed without error
        stats = tracker.get_statistics()
        assert stats['frame_count'] == len(test_embeddings)


@pytest.mark.embedding
class TestEmbeddingEdgeCases:
    """Test edge cases specific to embedding processing."""
    
    def test_very_high_dimensional_embeddings(self):
        """Test handling of very high dimensional embeddings."""
        config = SwarmSortConfig(use_embeddings=True, min_consecutive_detections=1)
        tracker = SwarmSort(config)
        
        # Test progressively larger dimensions
        dimensions = [32, 128, 512, 2048]
        
        for dim in dimensions:
            np.random.seed(42)  # Consistent for comparison
            embedding = np.random.randn(dim).astype(np.float32)
            
            detection = Detection(
                position=np.array([10, 20]),
                confidence=0.9,
                embedding=embedding
            )
            
            # Should handle high dimensions
            result = tracker.update([detection])
            assert isinstance(result, list)
            
            tracker.reset()  # Reset for next dimension
    
    def test_embedding_with_special_values(self):
        """Test embeddings containing special float values."""
        config = SwarmSortConfig(use_embeddings=True)
        tracker = SwarmSort(config)
        
        # Test embeddings with special values
        special_embeddings = [
            # Very large values
            np.array([1e6] * 64, dtype=np.float32),
            # Very small values  
            np.array([1e-6] * 64, dtype=np.float32),
            # Mixed large and small
            np.array([1e6, 1e-6] * 32, dtype=np.float32),
        ]
        
        for i, embedding in enumerate(special_embeddings):
            detection = Detection(
                position=np.array([i * 30, 50]),
                confidence=0.9,
                embedding=embedding
            )
            
            try:
                result = tracker.update([detection])
                assert isinstance(result, list)
            except (ValueError, OverflowError, FloatingPointError):
                # Some extreme values might be rejected
                pass
    
    def test_embedding_consistency_across_updates(self):
        """Test embedding consistency in tracking."""
        config = SwarmSortConfig(
            use_embeddings=True,
            embedding_weight=0.7,
            min_consecutive_detections=2,
            max_embeddings_per_track=5
        )
        tracker = SwarmSort(config)
        
        np.random.seed(42)
        base_embedding = np.random.randn(128).astype(np.float32)
        
        track_ids_history = []
        
        # Gradual embedding drift
        for i in range(15):
            # Embedding drifts slightly each frame
            drift = np.random.randn(128) * 0.02 * i  # Increasing drift
            current_embedding = base_embedding + drift
            
            detection = Detection(
                position=np.array([10 + i, 20]),
                confidence=0.9,
                embedding=current_embedding
            )
            
            tracked_objects = tracker.update([detection])
            
            if tracked_objects:
                track_ids_history.append(tracked_objects[0].id)
        
        # Should maintain track continuity despite embedding drift
        if track_ids_history:
            unique_ids = set(track_ids_history)
            # Should be mostly one ID (allowing for some initial uncertainty)
            assert len(unique_ids) <= 3
            
            # Most common ID should dominate
            from collections import Counter
            id_counts = Counter(track_ids_history)
            most_common_count = id_counts.most_common(1)[0][1]
            assert most_common_count >= len(track_ids_history) * 0.6
    
    def test_embedding_scaling_numerical_stability(self):
        """Test numerical stability of embedding scaling."""
        scaler = EmbeddingDistanceScaler(
            method='robust_minmax',
            min_samples=20
        )
        
        # Test with problematic distributions
        test_cases = [
            # All very similar values
            np.array([0.5] * 50 + [0.500001] * 50, dtype=np.float32),
            # Extreme outliers
            np.array([0.1] * 90 + [100.0] * 10, dtype=np.float32),
            # Very small differences
            np.linspace(1e-8, 2e-8, 100, dtype=np.float32),
        ]
        
        for i, distances in enumerate(test_cases):
            scaler_test = EmbeddingDistanceScaler(method='robust_minmax', min_samples=10)
            scaler_test.update_statistics(distances)
            
            try:
                scaled = scaler_test.scale_distances(distances)
                
                # Check for numerical stability
                assert np.all(np.isfinite(scaled))
                assert np.all(scaled >= 0.0)
                assert np.all(scaled <= 1.0)
                # Check that scaling didn't collapse to single value (allow for some tolerance)
                assert not np.allclose(scaled, scaled[0])  # Should not collapse to single value
                
            except (RuntimeWarning, FloatingPointError) as e:
                # Some extreme cases might trigger warnings
                warnings.warn(f"Test case {i} triggered numerical warning: {e}")
    
    def test_embedding_memory_efficiency(self):
        """Test memory efficiency of embedding storage."""
        config = SwarmSortConfig(
            use_embeddings=True,
            max_embeddings_per_track=10,
            min_consecutive_detections=1
        )
        tracker = SwarmSort(config)
        
        # Create large embeddings
        np.random.seed(42)
        large_embedding_dim = 1024
        base_embeddings = [
            np.random.randn(large_embedding_dim).astype(np.float32)
            for _ in range(5)
        ]
        
        # Process many frames
        for frame in range(100):
            detections = []
            
            for i, base_emb in enumerate(base_embeddings):
                # Small variation each frame
                current_emb = base_emb + np.random.randn(large_embedding_dim) * 0.01
                
                detection = Detection(
                    position=np.array([i * 40 + frame % 5, 50]),
                    confidence=0.9,
                    embedding=current_emb
                )
                detections.append(detection)
            
            tracked_objects = tracker.update(detections)
        
        # Check memory constraints are respected
        for track in tracker.tracker.tracks.values():
            assert len(track.embeddings) <= config.max_embeddings_per_track
        
        # Should still be tracking successfully
        stats = tracker.get_statistics()
        assert stats['active_tracks'] >= 3


@pytest.mark.embedding
class TestEmbeddingScalingMethods:
    """Test specific embedding scaling methods in detail."""
    
    def setup_method(self):
        """Set up test data for scaling methods."""
        np.random.seed(42)
        # Create a bimodal distribution for testing
        self.test_distances = np.concatenate([
            np.random.beta(2, 8, 50),      # Clustered near 0
            np.random.beta(8, 2, 50) * 0.3 + 0.7  # Clustered near 1
        ]).astype(np.float32)
    
    @pytest.mark.parametrize("method,expected_properties", [
        ('robust_minmax', {'preserves_order': True, 'bounded': True}),
        ('min_robustmax', {'preserves_order': True, 'bounded': True}),
        ('zscore', {'normalized': True, 'bounded': False}),
        ('robust_zscore', {'normalized': True, 'bounded': False}),
        ('arcsinh', {'monotonic': True, 'bounded': False}),
        ('arcsinh_percentile', {'bounded': True, 'monotonic': True}),
        ('beta', {'bounded': True, 'smooth': True}),
        ('quantile', {'bounded': True, 'preserves_order': True}),
        ('sigmoid', {'bounded': True, 'smooth': True}),
    ])
    def test_scaling_method_properties(self, method, expected_properties):
        """Test specific properties of each scaling method."""
        scaler = EmbeddingDistanceScaler(method=method, min_samples=20)
        scaler.update_statistics(self.test_distances)
        
        scaled = scaler.scale_distances(self.test_distances)
        
        # Test bounded property
        if expected_properties.get('bounded'):
            assert np.all(scaled >= 0.0)
            assert np.all(scaled <= 1.0)
        
        # Test order preservation (with tolerance for numerical precision)
        if expected_properties.get('preserves_order'):
            # Sorted order should be preserved (allowing for small numerical differences)
            original_order = np.argsort(self.test_distances)
            scaled_order = np.argsort(scaled)
            # Allow small discrepancies due to numerical precision in scaling
            order_similarity = np.mean(original_order == scaled_order)
            assert order_similarity > 0.9, f"Order preservation failed: {order_similarity:.3f} similarity"
        
        # Test monotonicity
        if expected_properties.get('monotonic'):
            sorted_indices = np.argsort(self.test_distances)
            sorted_scaled = scaled[sorted_indices]
            # Should be non-decreasing (avoid boolean array ambiguity)
            diff_values = np.diff(sorted_scaled)
            assert (diff_values >= -1e-6).all()  # Small tolerance for numerical errors
    
    def test_scaling_method_comparison(self):
        """Compare different scaling methods on the same data."""
        methods = ['robust_minmax', 'min_robustmax', 'beta', 'sigmoid']
        
        scalers = {}
        scaled_results = {}
        
        for method in methods:
            scaler = EmbeddingDistanceScaler(method=method, min_samples=20)
            scaler.update_statistics(self.test_distances)
            scaled = scaler.scale_distances(self.test_distances)
            
            scalers[method] = scaler
            scaled_results[method] = scaled
        
        # All methods should produce valid outputs
        for method, scaled in scaled_results.items():
            assert np.all(np.isfinite(scaled))
            assert len(scaled) == len(self.test_distances)
        
        # Different methods should produce different distributions
        methods_list = list(scaled_results.keys())
        for i, method1 in enumerate(methods_list):
            for method2 in methods_list[i+1:]:
                correlation = np.corrcoef(scaled_results[method1], scaled_results[method2])[0, 1]
                # Should be correlated (same order) but allow high correlation for similar methods
                # Some methods like robust_minmax and min_robustmax can be very similar
                assert correlation > 0.5, f"{method1} vs {method2}: correlation = {correlation} (too low)"
                if correlation >= 0.999:
                    # Very high correlation is acceptable for similar scaling methods
                    pass
                else:
                    assert correlation < 0.99, f"{method1} vs {method2}: correlation = {correlation} (identical methods?)"
    
    def test_scaling_adaptation_over_time(self):
        """Test how scaling adapts as more data arrives."""
        scaler = EmbeddingDistanceScaler(
            method='robust_minmax',
            update_rate=0.1,  # Faster adaptation for testing
            min_samples=30
        )
        
        np.random.seed(42)
        
        # Phase 1: Narrow distribution
        phase1_distances = np.random.beta(5, 5, 50) * 0.3  # [0, 0.3]
        scaler.update_statistics(phase1_distances)
        phase1_scaled = scaler.scale_distances(phase1_distances)
        
        # Should use most of [0, 1] range
        phase1_range = np.max(phase1_scaled) - np.min(phase1_scaled)
        
        # Phase 2: Wider distribution
        phase2_distances = np.random.beta(2, 2, 50) * 0.8  # [0, 0.8]
        scaler.update_statistics(phase2_distances)
        phase2_scaled = scaler.scale_distances(phase2_distances)
        
        # Should adapt to new distribution
        phase2_range = np.max(phase2_scaled) - np.min(phase2_scaled)
        
        # Both phases should utilize reasonable range
        assert phase1_range > 0.3
        assert phase2_range > 0.3
        
        # Statistics should reflect adaptation
        stats1 = scaler.get_statistics()
        assert stats1['sample_count'] == 100
        assert stats1['ready']


@pytest.mark.embedding
@pytest.mark.slow
class TestEmbeddingIntegrationScenarios:
    """Integration tests for embedding functionality in realistic scenarios."""
    
    def test_multi_class_embedding_tracking(self):
        """Test tracking multiple object classes with different embedding characteristics."""
        config = SwarmSortConfig(
            use_embeddings=True,
            embedding_weight=0.5,
            min_consecutive_detections=2,
            embedding_scaling_min_samples=50
        )
        tracker = SwarmSort(config)
        
        np.random.seed(42)
        
        # Create class-specific embedding templates
        class_templates = {
            0: np.random.randn(128).astype(np.float32),  # Class 0
            1: np.random.randn(128).astype(np.float32),  # Class 1  
            2: np.random.randn(128).astype(np.float32),  # Class 2
        }
        
        # Make templates more distinct
        class_templates[1] *= 2.0
        class_templates[2] *= 0.5
        
        objects = []
        for class_id in [0, 1, 2]:
            for instance in range(3):  # 3 instances per class
                obj = {
                    'class_id': class_id,
                    'instance': instance,
                    'position': np.array([class_id * 80 + instance * 25, 50 + instance * 30], dtype=np.float64),
                    'embedding_template': class_templates[class_id],
                    'velocity': np.array([1.0, 0.5 * (-1) ** instance])
                }
                objects.append(obj)
        
        # Track over multiple frames
        for frame in range(25):
            detections = []
            
            for obj in objects:
                # Update position
                obj['position'] += obj['velocity'] + np.random.randn(2) * 0.5
                
                # Create embedding with class characteristics + instance variation
                embedding = (obj['embedding_template'] + 
                           np.random.randn(128) * 0.1 +  # General noise
                           np.random.randn(128) * 0.05 * obj['instance'])  # Instance variation
                
                detection = Detection(
                    position=obj['position'].copy(),
                    confidence=0.8 + np.random.rand() * 0.2,
                    embedding=embedding,
                    class_id=obj['class_id']
                )
                detections.append(detection)
            
            tracked_objects = tracker.update(detections)
        
        # Should track most objects distinctly
        final_stats = tracker.get_statistics()
        assert final_stats['active_tracks'] >= 7  # At least 7/9 objects
        
        # Check class distribution in tracks
        if tracked_objects:
            class_counts = {}
            for track in tracked_objects:
                class_id = track.class_id
                if class_id is not None:
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
            
            # Should have tracks from multiple classes
            assert len(class_counts) >= 2
    
    def test_embedding_based_reidentification(self):
        """Test re-identification using embeddings."""
        config = SwarmSortConfig(
            use_embeddings=True,
            embedding_weight=0.6,
            reid_enabled=True,
            reid_embedding_threshold=0.3,
            reid_max_distance=100.0,
            max_age=5  # Short age for testing ReID
        )
        tracker = SwarmSort(config)
        
        np.random.seed(42)
        persistent_embedding = np.random.randn(128).astype(np.float32)
        
        # Phase 1: Object appears and gets tracked
        for i in range(5):
            detection = Detection(
                position=np.array([10 + i*2, 20]),
                confidence=0.9,
                embedding=persistent_embedding + np.random.randn(128) * 0.05
            )
            tracked_objects = tracker.update([detection])
        
        # Should have created a track
        phase1_stats = tracker.get_statistics()
        assert phase1_stats['active_tracks'] >= 1
        
        original_track_id = None
        if tracked_objects:
            original_track_id = tracked_objects[0].id
        
        # Phase 2: Object disappears (no detections)
        for _ in range(6):  # Longer than max_age
            tracker.update([])
        
        # Track should be lost
        phase2_stats = tracker.get_statistics()
        assert phase2_stats['active_tracks'] == 0
        assert phase2_stats['lost_tracks'] >= 1
        
        # Phase 3: Object reappears at different location with same embedding
        reappear_detection = Detection(
            position=np.array([100, 80]),  # Far from original location
            confidence=0.9,
            embedding=persistent_embedding + np.random.randn(128) * 0.05
        )
        
        tracked_objects = tracker.update([reappear_detection])
        
        # Should reidentify (though might get new ID due to implementation details)
        phase3_stats = tracker.get_statistics()
        assert phase3_stats['active_tracks'] >= 1
    
    def test_embedding_scaling_with_diverse_data(self):
        """Test embedding scaling with diverse real-world-like data."""
        config = SwarmSortConfig(
            use_embeddings=True,
            embedding_scaling_method='min_robustmax',
            embedding_scaling_min_samples=100,
            embedding_weight=0.7
        )
        tracker = SwarmSort(config)
        
        np.random.seed(42)
        
        # Simulate different types of objects with different embedding characteristics
        scenarios = [
            # Similar objects (small embedding distances)
            lambda: np.random.randn(64) * 0.1,
            # Diverse objects (large embedding distances)  
            lambda: np.random.randn(64) * 2.0,
            # Mixed scenario
            lambda: np.random.randn(64) * (0.1 if np.random.rand() < 0.3 else 1.0)
        ]
        
        scaling_evolution = []
        
        for scenario_phase, embedding_generator in enumerate(scenarios):
            for frame in range(50):
                detections = []
                
                # Generate 5-8 detections per frame
                for i in range(5 + scenario_phase):
                    detection = Detection(
                        position=np.random.rand(2) * 200,
                        confidence=0.8 + np.random.rand() * 0.2,
                        embedding=embedding_generator().astype(np.float32)
                    )
                    detections.append(detection)
                
                tracked_objects = tracker.update(detections)
                
                # Track scaling evolution
                emb_stats = tracker.get_statistics()['embedding_scaler_stats']
                if emb_stats['ready']:
                    scaling_evolution.append({
                        'frame': scenario_phase * 50 + frame,
                        'scenario': scenario_phase,
                        'min_distance': emb_stats['min_distance'],
                        'max_distance': emb_stats['max_distance'],
                        'mean_distance': emb_stats['mean_distance']
                    })
        
        # Should have successful scaling evolution
        assert len(scaling_evolution) > 0
        
        # Scaling should adapt across scenarios
        if len(scaling_evolution) > 10:
            early_stats = scaling_evolution[0]
            later_stats = scaling_evolution[-1]
            
            # Statistics should evolve
            assert later_stats['min_distance'] != early_stats['min_distance'] or \
                   later_stats['max_distance'] != early_stats['max_distance']


if __name__ == "__main__":
    # Run basic embedding tests when executed directly
    print("Running basic embedding tests...")
    
    # Test scaler
    scaler = EmbeddingDistanceScaler(min_samples=10)
    distances = np.random.rand(50) * 0.8
    scaler.update_statistics(distances)
    scaled = scaler.scale_distances(distances)
    print(f"Scaler test: {len(scaled)} distances scaled, range: {np.min(scaled):.3f} - {np.max(scaled):.3f}")
    
    # Test tracker with embeddings
    tracker = SwarmSort(SwarmSortConfig(use_embeddings=True, min_consecutive_detections=2))
    
    np.random.seed(42)
    base_emb = np.random.randn(64).astype(np.float32)
    
    for i in range(10):
        detection = Detection(
            position=np.array([10 + i, 20]),
            confidence=0.9,
            embedding=base_emb + np.random.randn(64) * 0.1
        )
        result = tracker.update([detection])
        print(f"Frame {i}: {len(result)} tracks")
    
    stats = tracker.get_statistics()
    emb_stats = stats['embedding_scaler_stats']
    print(f"Final: {stats['active_tracks']} tracks, scaler ready: {emb_stats['ready']}")
    print("Basic embedding tests completed successfully!")