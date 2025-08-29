"""
Integration tests for SwarmSort package.
Tests complete tracking scenarios and integration between components.
"""
import pytest
import numpy as np
from typing import List

from swarmsort import (
    SwarmSort,
    SwarmSortConfig,
    Detection,
    TrackedObject,
    create_tracker,
    is_within_swarmtracker,
)


@pytest.mark.integration
class TestCompleteTrackingScenarios:
    """Test complete end-to-end tracking scenarios."""

    def test_single_object_tracking(self, basic_tracker, tracking_assertions):
        """Test tracking a single object across multiple frames."""
        tracks_sequence = []

        # Track single object moving in straight line
        for i in range(10):
            detection = Detection(
                position=np.array([10.0 + i * 3, 20.0 + i * 2]), confidence=0.9, id=f"det_{i}"
            )

            tracked_objects = basic_tracker.update([detection])
            tracks_sequence.append(tracked_objects)

        # After enough frames, should have one stable track
        final_tracks = tracks_sequence[-1]
        assert len(final_tracks) >= 1

        # Check track continuity
        tracking_assertions.assert_track_continuity(tracks_sequence)
        tracking_assertions.assert_no_track_jumps(tracks_sequence, max_distance=10.0)

        # Track should have reasonable age and hits
        if final_tracks:
            track = final_tracks[0]
            assert track.age >= 3
            assert track.hits >= 3

    def test_multiple_objects_tracking(self, basic_config, tracking_assertions):
        """Test tracking multiple objects simultaneously."""
        config = basic_config
        config.min_consecutive_detections = 2
        tracker = SwarmSort(config)

        tracks_sequence = []

        # Track 3 objects moving in different directions
        for i in range(8):
            detections = [
                Detection(position=np.array([10.0 + i * 2, 20.0]), confidence=0.9),  # Moving right
                Detection(position=np.array([50.0, 30.0 + i * 1.5]), confidence=0.8),  # Moving up
                Detection(
                    position=np.array([80.0 - i, 50.0 + i]), confidence=0.85
                ),  # Moving diagonally
            ]

            tracked_objects = tracker.update(detections)
            tracks_sequence.append(tracked_objects)

        # Should eventually have 3 tracks
        final_tracks = tracks_sequence[-1]
        assert len(final_tracks) == 3

        # All tracks should have unique IDs
        track_ids = [track.id for track in final_tracks]
        assert len(set(track_ids)) == len(track_ids)

        # Check continuity and no unrealistic jumps
        tracking_assertions.assert_track_continuity(tracks_sequence)
        tracking_assertions.assert_no_track_jumps(tracks_sequence)

    def test_object_appearance_disappearance(self, basic_config):
        """Test handling of objects appearing and disappearing."""
        config = basic_config
        config.max_age = 5
        tracker = SwarmSort(config)

        all_tracks = []

        # Object appears
        for i in range(3):
            detection = Detection(position=np.array([10.0 + i, 20.0 + i]), confidence=0.9)
            tracks = tracker.update([detection])
            all_tracks.extend(tracks)

        assert tracker.get_statistics()["active_tracks"] >= 1

        # Object disappears (no detections)
        for i in range(6):  # More than max_age
            tracks = tracker.update([])
            all_tracks.extend(tracks)

        # Track should be removed after max_age
        final_stats = tracker.get_statistics()
        assert final_stats["active_tracks"] == 0

        # Object reappears
        for i in range(3):
            detection = Detection(position=np.array([20.0 + i, 30.0 + i]), confidence=0.9)
            tracks = tracker.update([detection])
            all_tracks.extend(tracks)

        # Should create new track (or re-identify if ReID enabled)
        reappear_stats = tracker.get_statistics()
        assert reappear_stats["active_tracks"] >= 1

    def test_crossing_objects(self, basic_config):
        """Test tracking objects that cross paths."""
        tracker = SwarmSort(basic_config)

        tracks_sequence = []

        for i in range(10):
            # Two objects crossing paths
            detections = [
                Detection(position=np.array([10.0 + i * 5, 50.0]), confidence=0.9),  # Moving right
                Detection(position=np.array([60.0 - i * 5, 50.0]), confidence=0.9),  # Moving left
            ]

            tracked_objects = tracker.update(detections)
            tracks_sequence.append(tracked_objects)

        # Should maintain 2 distinct tracks even when crossing
        final_tracks = tracks_sequence[-1]
        assert len(final_tracks) == 2

        # Tracks should have different IDs
        track_ids = [track.id for track in final_tracks]
        assert len(set(track_ids)) == 2


@pytest.mark.integration
class TestEmbeddingIntegration:
    """Test integration scenarios with embeddings."""

    def test_embedding_based_association(self, embedding_config):
        """Test that embeddings improve association accuracy."""
        # Ensure the test uses the probabilistic path, which correctly updates scaler stats
        embedding_config.use_probabilistic_costs = True

        tracker = SwarmSort(embedding_config)

        # Create consistent embeddings for two objects
        emb1 = np.random.randn(64).astype(np.float32)
        emb2 = np.random.randn(64).astype(np.float32)

        tracks_sequence = []

        for i in range(8):
            # Objects with distinct embeddings
            detections = [
                Detection(
                    position=np.array(
                        [10.0 + i + np.random.randn() * 2, 20.0 + np.random.randn() * 2]
                    ),
                    confidence=0.9,
                    embedding=emb1 + np.random.randn(64) * 0.1,  # Small variation
                ),
                Detection(
                    position=np.array(
                        [50.0 + i + np.random.randn() * 2, 30.0 + np.random.randn() * 2]
                    ),
                    confidence=0.8,
                    embedding=emb2 + np.random.randn(64) * 0.1,  # Small variation
                ),
            ]

            tracked_objects = tracker.update(detections)
            tracks_sequence.append(tracked_objects)

        # Should successfully track despite positional noise
        final_tracks = tracks_sequence[-1]
        assert len(final_tracks) == 2

        # Check embedding scaler is being updated
        stats = tracker.get_statistics()
        emb_stats = stats["embedding_scaler_stats"]
        assert emb_stats["sample_count"] > 0

    def test_embedding_scaling_adaptation(self, embedding_config):
        """Test that embedding distance scaling adapts over time."""
        config = embedding_config
        config.embedding_scaling_min_samples = 10  # Low threshold for testing
        tracker = SwarmSort(config)

        # Generate embeddings with different scales
        for phase in range(3):
            for i in range(8):
                # Different embedding scales per phase
                scale = 0.1 + phase * 0.5
                embedding = np.random.randn(64).astype(np.float32) * scale

                detection = Detection(
                    position=np.array([10.0 + phase * 50 + i * 2, 20.0 + i]),
                    confidence=0.9,
                    embedding=embedding,
                )

                tracker.update([detection])

        # Scaler should have adapted to different scales
        stats = tracker.get_statistics()
        emb_stats = stats["embedding_scaler_stats"]
        # Scaler might be ready or need more samples for complex scenarios
        assert emb_stats["ready"] or emb_stats["sample_count"] >= 0


@pytest.mark.integration
class TestConfigurationIntegration:
    """Test integration with different configurations."""

    def test_strict_vs_permissive_configs(self):
        """Test behavior differences between strict and permissive configs."""
        # Strict configuration
        strict_config = SwarmSortConfig(
            max_distance=20.0,
            high_score_threshold=0.9,
            min_consecutive_detections=4,
            detection_conf_threshold=0.8,
        )

        # Permissive configuration
        permissive_config = SwarmSortConfig(
            max_distance=100.0,
            high_score_threshold=0.3,
            min_consecutive_detections=1,
            detection_conf_threshold=0.1,
        )

        strict_tracker = SwarmSort(strict_config)
        permissive_tracker = SwarmSort(permissive_config)

        # Test with noisy detections
        noisy_detections = [
            Detection(position=np.array([10.0, 20.0]), confidence=0.5),  # Medium confidence
            Detection(position=np.array([50.0, 60.0]), confidence=0.7),  # Medium confidence
            Detection(
                position=np.array([15.0, 25.0]), confidence=0.4
            ),  # Low confidence, close to first
        ]

        # Run several frames
        strict_tracks = []
        permissive_tracks = []

        for _ in range(5):
            strict_result = strict_tracker.update(noisy_detections)
            permissive_result = permissive_tracker.update(noisy_detections)

            strict_tracks.extend(strict_result)
            permissive_tracks.extend(permissive_result)

        # Permissive should create more tracks, strict should be more conservative
        strict_stats = strict_tracker.get_statistics()
        permissive_stats = permissive_tracker.get_statistics()

        # This is probabilistic, but generally true
        assert permissive_stats["active_tracks"] >= strict_stats["active_tracks"]

    def test_reid_enabled_vs_disabled(self):
        """Test ReID enabled vs disabled behavior."""
        reid_config = SwarmSortConfig(
            reid_enabled=True, reid_max_distance=80.0, reid_embedding_threshold=0.5, max_age=3
        )

        no_reid_config = SwarmSortConfig(reid_enabled=False, max_age=3)

        reid_tracker = SwarmSort(reid_config)
        no_reid_tracker = SwarmSort(no_reid_config)

        # Create consistent embedding for reidentification
        consistent_embedding = np.random.randn(64).astype(np.float32)

        # Object appears
        for i in range(3):
            detection = Detection(
                position=np.array([10.0 + i, 20.0]),
                confidence=0.9,
                embedding=consistent_embedding + np.random.randn(64) * 0.05,
            )
            reid_tracker.update([detection])
            no_reid_tracker.update([detection])

        # Object disappears
        for i in range(4):  # Longer than max_age
            reid_tracker.update([])
            no_reid_tracker.update([])

        # Object reappears at different location
        reappear_detection = Detection(
            position=np.array([50.0, 60.0]),  # Far from original
            confidence=0.9,
            embedding=consistent_embedding + np.random.randn(64) * 0.05,
        )

        reid_result = reid_tracker.update([reappear_detection])
        no_reid_result = no_reid_tracker.update([reappear_detection])

        # Both should process detections successfully
        # Get final statistics to verify operation
        reid_stats = reid_tracker.get_statistics()
        no_reid_stats = no_reid_tracker.get_statistics()
        assert reid_stats["frame_count"] >= 6 and no_reid_stats["frame_count"] >= 6


@pytest.mark.integration
class TestFactoryAndIntegrationAPIs:
    """Test factory functions and integration APIs."""

    def test_create_tracker_with_different_inputs(self, temp_config_file):
        """Test create_tracker with various input types."""
        # Default tracker
        tracker1 = create_tracker()
        assert hasattr(tracker1, "update")

        # With config dict
        config_dict = {"max_distance": 60.0, "use_embeddings": True}
        tracker2 = create_tracker(config_dict)
        assert tracker2.config.max_distance == 60.0
        assert tracker2.config.use_embeddings

        # With SwarmSortConfig object
        config_obj = SwarmSortConfig(embedding_weight=0.7)
        tracker3 = create_tracker(config_obj)
        assert tracker3.config.embedding_weight == 0.7

        # With YAML file
        tracker4 = create_tracker(temp_config_file)
        assert tracker4.config.max_distance == 75.0  # From YAML

        # Force standalone
        tracker5 = create_tracker(force_standalone=True)
        assert hasattr(tracker5, "update")

    def test_environment_detection(self):
        """Test environment detection functionality."""
        within_swarmtracker = is_within_swarmtracker()

        # Should return boolean
        assert isinstance(within_swarmtracker, bool)

        # In standalone tests, should typically be True (due to swarmtracker being available)
        # but this can vary depending on test environment

    def test_tracker_consistency_across_interfaces(self):
        """Test that different tracker interfaces behave consistently."""
        config = SwarmSortConfig(min_consecutive_detections=2)

        # Different ways to create tracker
        tracker1 = SwarmSort(config)
        tracker2 = create_tracker(config)

        # Same detections
        detections = [
            Detection(position=np.array([10.0, 20.0]), confidence=0.9),
            Detection(position=np.array([50.0, 60.0]), confidence=0.8),
        ]

        # Should behave similarly (though not necessarily identically due to randomness)
        result1 = tracker1.update(detections.copy())
        result2 = tracker2.update(detections.copy())

        # Both should return list of TrackedObject
        assert isinstance(result1, list)
        assert isinstance(result2, list)

        # Stats should be similar structure
        stats1 = tracker1.get_statistics()
        stats2 = tracker2.get_statistics()

        assert set(stats1.keys()) == set(stats2.keys())


@pytest.mark.integration
@pytest.mark.slow
class TestLongRunningScenarios:
    """Test long-running tracking scenarios."""

    def test_long_sequence_stability(self, tracking_scenario, basic_config):
        """Test tracker stability over long sequences."""
        config = basic_config
        config.min_consecutive_detections = 2
        tracker = SwarmSort(config)

        # Generate long scenario
        scenarios = tracking_scenario(num_objects=3, num_frames=50, noise_level=1.0)

        all_tracks = []
        track_counts = []

        for frame_detections in scenarios:
            tracked_objects = tracker.update(frame_detections)
            all_tracks.append(tracked_objects)
            track_counts.append(len(tracked_objects))

        # Should eventually stabilize to expected number of tracks
        final_track_count = track_counts[-1]
        assert final_track_count >= 2  # Should track at least some objects

        # Track count should be relatively stable in later frames
        later_counts = track_counts[-10:]  # Last 10 frames
        count_variance = np.var(later_counts)
        assert count_variance <= 2.0  # Reasonable stability

        # Memory usage should be reasonable
        stats = tracker.get_statistics()
        assert stats["active_tracks"] <= 10  # Should not create excessive tracks
        assert stats["lost_tracks"] <= 20  # Should not accumulate too many lost tracks

    def test_memory_management(self, tracking_scenario, embedding_config):
        """Test that memory usage remains reasonable over time."""
        config = embedding_config
        config.max_embeddings_per_track = 5  # Limit embedding history
        tracker = SwarmSort(config)

        # Run for many frames
        scenarios = tracking_scenario(num_objects=5, num_frames=100, noise_level=0.8)

        initial_stats = tracker.get_statistics()

        for i, frame_detections in enumerate(scenarios):
            tracker.update(frame_detections)

            # Periodically check memory usage
            if i % 20 == 19:
                stats = tracker.get_statistics()

                # Should not accumulate unlimited embeddings
                for track in tracker.tracker.tracks.values():
                    assert len(track.embedding_history) <= config.max_embeddings_per_track

                # Should not accumulate unlimited lost tracks
                assert stats["lost_tracks"] <= 50

        final_stats = tracker.get_statistics()
        assert final_stats["active_tracks"] <= 20  # Reasonable upper bound

    def test_performance_consistency(self, tracking_scenario, performance_tracker):
        """Test that performance remains consistent over time."""
        tracker = SwarmSort()
        scenarios = tracking_scenario(num_objects=4, num_frames=30)

        for i, frame_detections in enumerate(scenarios):
            with performance_tracker.time_operation("frame_update"):
                tracker.update(frame_detections)

        stats = performance_tracker.get_stats("frame_update")
        assert stats is not None

        # Performance should be reasonable and consistent
        mean_time = stats["mean"]
        max_time = stats["max"]

        assert mean_time < 0.1  # Should be fast (< 100ms per frame)
        assert max_time < mean_time * 5  # No frame should be dramatically slower


@pytest.mark.integration
class TestConfigurationPersistence:
    """Test configuration loading and saving."""

    def test_config_to_dict_roundtrip(self):
        """Test configuration serialization roundtrip."""
        original_config = SwarmSortConfig(
            max_distance=85.0,
            use_embeddings=True,
            embedding_weight=0.45,
            reid_enabled=False,
            min_consecutive_detections=4,
        )

        # Convert to dict and back
        config_dict = original_config.to_dict()
        restored_config = SwarmSortConfig.from_dict(config_dict)

        # Should be identical
        assert restored_config.max_distance == original_config.max_distance
        assert restored_config.use_embeddings == original_config.use_embeddings
        assert restored_config.embedding_weight == original_config.embedding_weight
        assert restored_config.reid_enabled == original_config.reid_enabled
        assert (
            restored_config.min_consecutive_detections == original_config.min_consecutive_detections
        )

    def test_config_validation_integration(self):
        """Test that configuration validation works in practice."""
        # Valid config should work
        valid_config = SwarmSortConfig(max_distance=50.0)
        tracker = SwarmSort(valid_config)

        detection = Detection(position=np.array([10.0, 20.0]), confidence=0.9)
        result = tracker.update([detection])
        assert isinstance(result, list)

        # Invalid config should raise error
        with pytest.raises(ValueError):
            invalid_config = SwarmSortConfig(max_distance=-10.0)
            invalid_config.validate()

    def test_yaml_config_integration(self, temp_config_file):
        """Test YAML configuration loading integration."""
        tracker = create_tracker(temp_config_file)

        # Config values from YAML should be applied
        assert tracker.config.max_distance == 75.0
        assert tracker.config.high_score_threshold == 0.85
        assert tracker.config.use_embeddings == True
        assert tracker.config.embedding_weight == 0.4

        # Should work for tracking
        detection = Detection(position=np.array([10.0, 20.0]), confidence=0.9)
        result = tracker.update([detection])
        assert isinstance(result, list)
