"""
Stress tests and edge case tests for SwarmSort.
Tests for robustness, error handling, and extreme scenarios.
"""
import pytest
import numpy as np
import warnings
from typing import List, Dict, Any
import gc

from swarmsort import SwarmSort, SwarmSortConfig, Detection, TrackedObject, SwarmSortTracker


@pytest.mark.stress
class TestEdgeCaseInputs:
    """Test handling of edge case inputs and invalid data."""

    def test_empty_detection_sequences(self):
        """Test handling of empty detection sequences."""
        tracker = SwarmSort()

        # Multiple empty frames
        for _ in range(10):
            result = tracker.update([])
            assert isinstance(result, list)
            assert len(result) == 0

        stats = tracker.get_statistics()
        assert stats["frame_count"] == 10
        assert stats["active_tracks"] == 0

    def test_single_detection_sequences(self):
        """Test handling of single detection per frame."""
        tracker = SwarmSort(SwarmSortConfig(min_consecutive_detections=2))

        # Single detection moving
        for i in range(10):
            detection = Detection(position=np.array([10.0 + i, 20.0 + i * 0.5]), confidence=0.9)
            result = tracker.update([detection])
            assert isinstance(result, list)

        # Should eventually create a track
        final_stats = tracker.get_statistics()
        assert final_stats["active_tracks"] >= 1

    def test_very_low_confidence_detections(self):
        """Test handling of very low confidence detections."""
        config = SwarmSortConfig(detection_conf_threshold=0.1)  # Very low threshold
        tracker = SwarmSort(config)

        low_conf_detections = [
            Detection(position=np.array([10.0, 20.0]), confidence=0.05),  # Below threshold
            Detection(position=np.array([50.0, 60.0]), confidence=0.15),  # Above threshold
            Detection(position=np.array([100.0, 30.0]), confidence=0.02),  # Below threshold
        ]

        result = tracker.update(low_conf_detections)
        # Should filter out very low confidence detections
        assert isinstance(result, list)

    def test_extreme_coordinate_values(self):
        """Test handling of extreme coordinate values."""
        tracker = SwarmSort()

        extreme_detections = [
            Detection(position=np.array([1e6, 1e6]), confidence=0.9),  # Very large
            Detection(position=np.array([-1e6, -1e6]), confidence=0.9),  # Very negative
            Detection(position=np.array([1e-6, 1e-6]), confidence=0.9),  # Very small
            Detection(position=np.array([0.0, 0.0]), confidence=0.9),  # Zero
        ]

        # Should handle without crashing
        result = tracker.update(extreme_detections)
        assert isinstance(result, list)

        # Test statistics still work
        stats = tracker.get_statistics()
        assert isinstance(stats, dict)

    def test_invalid_bbox_values(self):
        """Test handling of invalid bounding box values."""
        tracker = SwarmSort()

        # Various invalid bboxes
        invalid_bbox_detections = [
            Detection(
                position=np.array([10.0, 20.0]), confidence=0.8, bbox=np.array([0, 0, -10, -10])
            ),  # Negative size
            Detection(
                position=np.array([30.0, 40.0]), confidence=0.9, bbox=np.array([100, 100, 50, 50])
            ),  # x1 > x2, y1 > y2
            Detection(
                position=np.array([50.0, 60.0]), confidence=0.7, bbox=np.array([np.inf, 0, 10, 10])
            ),  # Infinity
        ]

        # Should handle gracefully
        try:
            result = tracker.update(invalid_bbox_detections)
            assert isinstance(result, list)
        except (ValueError, TypeError):
            # Acceptable to reject invalid input
            pass

    def test_nan_and_inf_positions(self):
        """Test handling of NaN and infinity in positions."""
        tracker = SwarmSort()

        # Detections with problematic values
        problematic_detections = [
            Detection(position=np.array([10.0, 20.0]), confidence=0.9),  # Normal
        ]

        # Test NaN positions
        try:
            nan_detection = Detection(position=np.array([float("nan"), 20.0]), confidence=0.9)
            result = tracker.update([nan_detection])
            # If it doesn't crash, it should return a list
            assert isinstance(result, list)
        except (ValueError, TypeError):
            # Acceptable to reject NaN input
            pass

        # Test infinity positions
        try:
            inf_detection = Detection(position=np.array([float("inf"), 20.0]), confidence=0.9)
            result = tracker.update([inf_detection])
            assert isinstance(result, list)
        except (ValueError, TypeError, OverflowError):
            # Acceptable to reject infinity input
            pass

    def test_mismatched_embedding_dimensions(self):
        """Test handling of mismatched embedding dimensions."""
        config = SwarmSortConfig(use_embeddings=True)
        tracker = SwarmSort(config)

        detections_mixed_dims = [
            Detection(
                position=np.array([10.0, 20.0]),
                embedding=np.random.randn(64).astype(np.float32),  # 64-dim
                confidence=0.9,
            ),
            Detection(
                position=np.array([50.0, 60.0]),
                embedding=np.random.randn(128).astype(np.float32),  # 128-dim
                confidence=0.9,
            ),
        ]

        # Should handle gracefully (either by ignoring embeddings or normalizing)
        try:
            result = tracker.update(detections_mixed_dims)
            assert isinstance(result, list)
        except (ValueError, IndexError):
            # Acceptable to fail on mismatched dimensions
            pass

    def test_zero_and_negative_confidences(self):
        """Test handling of zero and negative confidence values."""
        tracker = SwarmSort(SwarmSortConfig(detection_conf_threshold=0.0))

        edge_confidence_detections = [
            Detection(position=np.array([10.0, 20.0]), confidence=0.0),  # Zero confidence
            Detection(position=np.array([50.0, 60.0]), confidence=-0.5),  # Negative confidence
            Detection(position=np.array([100.0, 30.0]), confidence=1.5),  # Above 1.0
        ]

        # Should handle edge cases
        result = tracker.update(edge_confidence_detections)
        assert isinstance(result, list)


@pytest.mark.stress
class TestHighLoadScenarios:
    """Test tracker behavior under high load conditions."""

    def test_many_objects_scenario(self):
        """Test tracking many objects simultaneously."""
        config = SwarmSortConfig(max_distance=50.0, min_consecutive_detections=2, max_age=15)
        tracker = SwarmSort(config)

        num_objects = 100  # Very high number

        # Create many objects in a grid
        for frame in range(20):
            detections = []
            grid_size = int(np.ceil(np.sqrt(num_objects)))

            for i in range(num_objects):
                row = i // grid_size
                col = i % grid_size

                # Add small motion
                position = np.array(
                    [
                        col * 40 + frame * 0.5 + np.random.randn() * 0.5,
                        row * 40 + frame * 0.3 + np.random.randn() * 0.5,
                    ]
                )

                detection = Detection(
                    position=position,
                    confidence=0.8 + np.random.rand() * 0.2,
                    id=f"obj_{i}_frame_{frame}",
                )
                detections.append(detection)

            result = tracker.update(detections)
            assert isinstance(result, list)

            # Memory and performance should remain reasonable
            if frame % 5 == 4:
                stats = tracker.get_statistics()
                # Should not create excessive tracks
                assert stats["active_tracks"] <= num_objects * 1.2

        # Final check
        final_stats = tracker.get_statistics()
        assert final_stats["active_tracks"] >= num_objects * 0.5  # Should track at least half

    def test_rapid_appearance_disappearance(self):
        """Test objects appearing and disappearing rapidly."""
        config = SwarmSortConfig(max_age=3, min_consecutive_detections=2)
        tracker = SwarmSort(config)

        object_pool = []
        for i in range(20):
            object_pool.append(
                {"id": i, "position": np.array([i * 15, 50]), "active": False, "frames_active": 0}
            )

        for frame in range(50):
            detections = []

            # Randomly activate/deactivate objects
            for obj in object_pool:
                if obj["active"]:
                    obj["frames_active"] += 1
                    # Random disappearance
                    if np.random.rand() < 0.2 or obj["frames_active"] > 8:
                        obj["active"] = False
                        obj["frames_active"] = 0
                else:
                    # Random appearance
                    if np.random.rand() < 0.15:
                        obj["active"] = True
                        obj["position"] = np.array([np.random.rand() * 300, np.random.rand() * 200])

                if obj["active"]:
                    # Small movement
                    obj["position"] += np.random.randn(2) * 2

                    detection = Detection(
                        position=obj["position"].copy(),
                        confidence=0.9,
                        id=f"obj_{obj['id']}_frame_{frame}",
                    )
                    detections.append(detection)

            result = tracker.update(detections)
            assert isinstance(result, list)

        # Should handle the chaos without crashing
        stats = tracker.get_statistics()
        assert stats["frame_count"] == 50

    def test_dense_object_clusters(self):
        """Test tracking objects in very dense clusters."""
        config = SwarmSortConfig(
            duplicate_detection_threshold=5.0,  # Small threshold for dense clusters
            max_distance=15.0,
            min_consecutive_detections=2,
        )
        tracker = SwarmSort(config)

        # Create dense clusters
        cluster_centers = [(50, 50), (150, 50), (100, 150)]

        for frame in range(15):
            detections = []

            for center_x, center_y in cluster_centers:
                # 10 objects per cluster in small area
                for i in range(10):
                    # Very small area around center
                    angle = i * 2 * np.pi / 10 + frame * 0.1
                    radius = 3 + np.random.rand() * 2  # Very close objects

                    position = np.array(
                        [center_x + radius * np.cos(angle), center_y + radius * np.sin(angle)]
                    )

                    detection = Detection(
                        position=position,
                        confidence=0.8 + np.random.rand() * 0.2,
                        id=f"cluster_{center_x}_{center_y}_obj_{i}_frame_{frame}",
                    )
                    detections.append(detection)

            result = tracker.update(detections)
            assert isinstance(result, list)

        # Should handle dense clusters reasonably
        stats = tracker.get_statistics()
        assert stats["active_tracks"] >= 10  # Should create some tracks despite density

    def test_very_noisy_detections(self):
        """Test tracking with very noisy detections."""
        config = SwarmSortConfig(
            max_distance=80.0, detection_conf_threshold=0.3  # Larger distance to handle noise
        )
        tracker = SwarmSort(config)

        # Create underlying true objects
        true_objects = []
        for i in range(5):
            true_objects.append(
                {
                    "position": np.array([i * 60 + 50, 100], dtype=np.float64),
                    "velocity": np.array([1.0, 0.5]),
                }
            )

        for frame in range(25):
            detections = []

            # Add true detections with noise
            for obj in true_objects:
                obj["position"] += obj["velocity"]

                # Very noisy position
                noisy_position = obj["position"] + np.random.randn(2) * 15  # High noise
                detection = Detection(
                    position=noisy_position, confidence=0.6 + np.random.rand() * 0.4
                )
                detections.append(detection)

            # Add false positive detections
            for _ in range(np.random.poisson(8)):  # Random false positives
                false_pos = Detection(
                    position=np.random.rand(2) * 400,
                    confidence=0.3 + np.random.rand() * 0.3,  # Lower confidence
                )
                detections.append(false_pos)

            result = tracker.update(detections)
            assert isinstance(result, list)

        # Should track true objects despite noise
        stats = tracker.get_statistics()
        assert stats["active_tracks"] >= 2  # Should track at least some true objects

    @pytest.mark.slow
    def test_very_long_sequence(self):
        """Test tracking over very long sequences."""
        config = SwarmSortConfig(max_age=20, max_embeddings_per_track=8)  # Limit memory usage
        tracker = SwarmSort(config)

        # Track 6 objects for 500 frames
        num_frames = 500
        num_objects = 6

        objects = []
        for i in range(num_objects):
            objects.append(
                {
                    "position": np.array([i * 50 + 25, 50], dtype=np.float64),
                    "velocity": np.array([np.random.rand() * 2 - 1, np.random.rand() * 2 - 1])
                    * 0.5,
                    "embedding": np.random.randn(64).astype(np.float32)
                    if config.use_embeddings
                    else None,
                }
            )

        track_count_history = []

        for frame in range(num_frames):
            detections = []

            for obj in objects:
                # Update position
                obj["position"] += obj["velocity"]

                # Occasionally change direction
                if np.random.rand() < 0.02:
                    obj["velocity"] = np.random.randn(2) * 0.5

                # Sometimes miss detection
                if np.random.rand() < 0.05:
                    continue

                # Update embedding slightly
                if obj["embedding"] is not None:
                    obj["embedding"] += np.random.randn(64) * 0.01

                detection = Detection(
                    position=obj["position"] + np.random.randn(2) * 0.5,  # Small noise
                    confidence=0.8 + np.random.rand() * 0.2,
                    embedding=obj["embedding"].copy() if obj["embedding"] is not None else None,
                )
                detections.append(detection)

            result = tracker.update(detections)
            assert isinstance(result, list)

            track_count_history.append(len(result))

            # Periodic checks
            if frame % 100 == 99:
                stats = tracker.get_statistics()
                # Should maintain reasonable track count
                assert stats["active_tracks"] <= num_objects * 2
                assert stats["lost_tracks"] <= 50  # Don't accumulate too many lost tracks

        # Final validation
        final_stats = tracker.get_statistics()
        assert final_stats["frame_count"] == num_frames
        assert final_stats["active_tracks"] >= num_objects * 0.5  # Should track most objects

        # Track count should be relatively stable in later frames
        later_counts = track_count_history[-50:]  # Last 50 frames
        if later_counts:
            count_variance = np.var(later_counts)
            assert count_variance <= 4.0  # Reasonable stability


@pytest.mark.stress
class TestResourceLimitTests:
    """Test behavior under resource constraints."""

    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure."""
        config = SwarmSortConfig(
            max_embeddings_per_track=3,  # Very limited
            max_age=5,  # Short age to force cleanup
            use_embeddings=True,
        )
        tracker = SwarmSort(config)

        # Create scenario that would use lots of memory
        large_embeddings = []
        for _ in range(20):
            large_embeddings.append(np.random.randn(512).astype(np.float32))  # Large embeddings

        for frame in range(100):
            detections = []

            # Many objects with large embeddings
            for i in range(15):
                detection = Detection(
                    position=np.array([i * 20 + frame % 10, 50 + i * 5]),
                    confidence=0.9,
                    embedding=large_embeddings[i % len(large_embeddings)].copy(),
                )
                detections.append(detection)

            result = tracker.update(detections)
            assert isinstance(result, list)

        # Verify memory constraints are respected
        for track in tracker.tracker.tracks.values():
            assert len(track.embedding_history) <= config.max_embeddings_per_track

        stats = tracker.get_statistics()
        assert stats["active_tracks"] <= 30  # Should not create excessive tracks

    def test_extreme_configuration_values(self):
        """Test with extreme configuration values."""
        # Very permissive config
        permissive_config = SwarmSortConfig(
            max_distance=1000.0,
            init_conf_threshold=0.01,
            min_consecutive_detections=1,
            detection_conf_threshold=0.001,
            max_track_age=1000,
            max_embeddings_per_track=100,
        )

        permissive_tracker = SwarmSort(permissive_config)

        # Very strict config
        strict_config = SwarmSortConfig(
            max_distance=1.0,
            init_conf_threshold=0.999,
            min_consecutive_detections=20,
            detection_conf_threshold=0.999,
            max_track_age=1,
            max_embeddings_per_track=1,
        )

        strict_tracker = SwarmSort(strict_config)

        # Test both with same data
        detections = [
            Detection(position=np.array([10.0, 20.0]), confidence=0.5),
            Detection(position=np.array([50.0, 60.0]), confidence=0.7),
            Detection(position=np.array([12.0, 21.0]), confidence=0.6),  # Close to first
        ]

        # Both should handle extreme configs without crashing
        permissive_result = permissive_tracker.update(detections)
        strict_result = strict_tracker.update(detections)

        assert isinstance(permissive_result, list)
        assert isinstance(strict_result, list)

        # Permissive should likely create more tracks
        permissive_stats = permissive_tracker.get_statistics()
        strict_stats = strict_tracker.get_statistics()

        # Both should have valid statistics
        assert isinstance(permissive_stats, dict)
        assert isinstance(strict_stats, dict)

    def test_concurrent_access_simulation(self):
        """Simulate concurrent access patterns (single-threaded simulation)."""
        tracker = SwarmSort()

        # Simulate multiple threads accessing tracker
        # (Note: SwarmSort is not thread-safe, but we test for graceful degradation)

        detection_sets = []
        for i in range(10):
            detections = [
                Detection(position=np.array([i * 20 + j * 5, 50]), confidence=0.9) for j in range(3)
            ]
            detection_sets.append(detections)

        # Rapidly switch between different detection sets
        for round_num in range(50):
            detection_set = detection_sets[round_num % len(detection_sets)]

            # Simulate rapid consecutive calls
            for _ in range(5):
                result = tracker.update(detection_set)
                assert isinstance(result, list)

            # Check statistics access
            stats = tracker.get_statistics()
            assert isinstance(stats, dict)

        # Should maintain consistent state
        final_stats = tracker.get_statistics()
        assert final_stats["frame_count"] == 50 * 5  # 250 frames total


@pytest.mark.stress
class TestErrorRecoveryTests:
    """Test error recovery and robustness."""

    def test_recovery_from_numerical_issues(self):
        """Test recovery from numerical instability."""
        tracker = SwarmSort(SwarmSortConfig(use_embeddings=True))

        # Normal operation first
        normal_detections = [
            Detection(
                position=np.array([10.0, 20.0]),
                confidence=0.9,
                embedding=np.random.randn(64).astype(np.float32),
            )
        ]

        for _ in range(5):
            tracker.update(normal_detections)

        # Create problematic scenarios
        problematic_scenarios = [
            # Very similar embeddings (potential numerical issues)
            [
                Detection(
                    position=np.array([10.0, 20.0]),
                    confidence=0.9,
                    embedding=np.ones(64, dtype=np.float32) * 0.1,
                )
                for _ in range(5)
            ],
            # Zero embeddings
            [
                Detection(
                    position=np.array([10.0, 20.0]),
                    confidence=0.9,
                    embedding=np.zeros(64, dtype=np.float32),
                )
            ],
            # Very large embeddings
            [
                Detection(
                    position=np.array([10.0, 20.0]),
                    confidence=0.9,
                    embedding=np.ones(64, dtype=np.float32) * 1000,
                )
            ],
        ]

        for scenario in problematic_scenarios:
            try:
                result = tracker.update(scenario)
                assert isinstance(result, list)
            except (ValueError, RuntimeError, FloatingPointError):
                # Some numerical issues might be unavoidable
                pass

        # Should recover and continue normal operation
        result = tracker.update(normal_detections)
        assert isinstance(result, list)

        stats = tracker.get_statistics()
        assert isinstance(stats, dict)

    def test_graceful_degradation_with_corrupted_state(self):
        """Test graceful degradation when internal state is corrupted."""
        tracker = SwarmSort()

        # Build up some state
        for i in range(10):
            detection = Detection(position=np.array([10.0 + i, 20.0]), confidence=0.9)
            tracker.update([detection])

        # Simulate state corruption (in a controlled way)
        original_tracks = tracker.tracker.tracks.copy()

        # Try with empty tracks (simulated corruption)
        tracker.tracker.tracks.clear()

        # Should handle gracefully
        detection = Detection(position=np.array([50.0, 60.0]), confidence=0.9)
        result = tracker.update([detection])
        assert isinstance(result, list)

        # Restore and continue
        tracker.tracker.tracks = original_tracks
        result = tracker.update([detection])
        assert isinstance(result, list)

    def test_configuration_consistency_checks(self):
        """Test configuration consistency and validation."""
        # Test various invalid configurations
        invalid_configs = [
            {"max_distance": -10.0},
            {"init_conf_threshold": 2.0},
            {"embedding_weight": -1.0},
            {"min_consecutive_detections": 0},
            {"max_track_age": 0},
            {"detection_conf_threshold": 1.5},
        ]

        for invalid_config in invalid_configs:
            with pytest.raises(ValueError):
                config = SwarmSortConfig(**invalid_config)
                config.validate()

        # Test edge case but valid configurations
        edge_configs = [
            {"max_distance": 1e-6},  # Very small
            {"init_conf_threshold": 0.0},  # Minimum
            {"embedding_weight": 0.0},  # No embeddings
            {"min_consecutive_detections": 1},  # Immediate
            {"max_age": 1},  # Very short
            {"detection_conf_threshold": 0.0},  # Accept all
        ]

        for edge_config in edge_configs:
            config = SwarmSortConfig(**edge_config)
            config.validate()  # Should not raise

            # Should be usable
            tracker = SwarmSort(config)
            detection = Detection(position=np.array([10.0, 20.0]), confidence=0.5)
            result = tracker.update([detection])
            assert isinstance(result, list)


@pytest.mark.stress
class TestIntegrationStressTests:
    """Integration stress tests combining multiple challenging conditions."""

    @pytest.mark.slow
    def test_comprehensive_stress_scenario(self):
        """Comprehensive stress test combining multiple challenging conditions."""
        config = SwarmSortConfig(
            use_embeddings=True,
            max_distance=60.0,
            embedding_weight=0.4,
            max_embeddings_per_track=5,
            max_age=10,
            reid_enabled=True,
        )
        tracker = SwarmSort(config)

        # Complex scenario combining:
        # - Many objects
        # - Occlusions (objects disappearing/reappearing)
        # - Noise
        # - Crossing trajectories
        # - Clustering and separation

        num_objects = 25
        num_frames = 100

        # Initialize objects with different behaviors
        objects = []
        for i in range(num_objects):
            behavior = i % 4  # 4 different behavior types

            if behavior == 0:  # Linear motion
                obj = {
                    "type": "linear",
                    "position": np.array([i * 10, 50], dtype=np.float64),
                    "velocity": np.array([1.0, 0.5]),
                    "embedding": np.random.randn(64).astype(np.float32),
                    "active_probability": 0.95,
                }
            elif behavior == 1:  # Circular motion
                obj = {
                    "type": "circular",
                    "center": np.array([100 + i * 15, 100], dtype=np.float64),
                    "radius": 20 + i,
                    "angle": 0,
                    "angular_velocity": 0.1,
                    "embedding": np.random.randn(64).astype(np.float32),
                    "active_probability": 0.9,
                }
            elif behavior == 2:  # Random walk
                obj = {
                    "type": "random",
                    "position": np.array([200 + i * 8, 150], dtype=np.float64),
                    "velocity": np.array([0, 0], dtype=np.float64),
                    "embedding": np.random.randn(64).astype(np.float32),
                    "active_probability": 0.85,
                }
            else:  # Intermittent (appears/disappears)
                obj = {
                    "type": "intermittent",
                    "position": np.array([300 + i * 5, 200], dtype=np.float64),
                    "velocity": np.array([0.5, -0.3]),
                    "embedding": np.random.randn(64).astype(np.float32),
                    "active_probability": 0.7,
                    "phase": i,
                }

            objects.append(obj)

        successful_frames = 0
        track_count_history = []

        for frame in range(num_frames):
            detections = []

            for i, obj in enumerate(objects):
                # Update object state
                if obj["type"] == "linear":
                    obj["position"] += obj["velocity"] + np.random.randn(2) * 0.5
                elif obj["type"] == "circular":
                    obj["angle"] += obj["angular_velocity"]
                    obj["position"] = (
                        obj["center"]
                        + obj["radius"] * np.array([np.cos(obj["angle"]), np.sin(obj["angle"])])
                        + np.random.randn(2) * 0.3
                    )
                elif obj["type"] == "random":
                    obj["velocity"] = obj["velocity"] * 0.9 + np.random.randn(2) * 0.8
                    obj["position"] += obj["velocity"]
                elif obj["type"] == "intermittent":
                    if (frame + obj["phase"]) % 10 < 6:  # Active 60% of time
                        obj["position"] += obj["velocity"] + np.random.randn(2) * 0.2
                    else:
                        continue  # Skip this detection

                # Determine if object is detected this frame
                if np.random.rand() > obj["active_probability"]:
                    continue

                # Add some embedding drift
                obj["embedding"] += np.random.randn(64) * 0.02

                # Create detection with noise
                noisy_position = obj["position"] + np.random.randn(2) * 1.0
                confidence = 0.6 + np.random.rand() * 0.4

                detection = Detection(
                    position=noisy_position,
                    confidence=confidence,
                    embedding=obj["embedding"] + np.random.randn(64) * 0.05,
                    id=f"obj_{i}_frame_{frame}",
                )
                detections.append(detection)

            # Add false positive detections
            for _ in range(np.random.poisson(2)):
                false_pos = Detection(
                    position=np.random.rand(2) * 400,
                    confidence=0.3 + np.random.rand() * 0.4,
                    embedding=np.random.randn(64).astype(np.float32),
                )
                detections.append(false_pos)

            try:
                result = tracker.update(detections)
                assert isinstance(result, list)
                successful_frames += 1
                track_count_history.append(len(result))

                # Periodic validation
                if frame % 20 == 19:
                    stats = tracker.get_statistics()
                    assert stats["active_tracks"] <= num_objects * 1.5  # Reasonable upper bound
                    assert stats["frame_count"] == frame + 1

            except Exception as e:
                # Log the error but continue (stress test shouldn't crash)
                print(f"Frame {frame} failed with error: {e}")
                if successful_frames < frame * 0.8:  # If too many failures
                    raise

        # Final validation
        final_stats = tracker.get_statistics()

        # Should have processed most frames successfully
        assert successful_frames >= num_frames * 0.9

        # Should track a reasonable number of objects
        if track_count_history:
            final_track_count = track_count_history[-1]
            assert final_track_count >= num_objects * 0.3  # At least 30% of objects
            assert final_track_count <= num_objects * 2  # Not too many false tracks

        # Memory usage should be reasonable
        assert final_stats["lost_tracks"] <= 100  # Don't accumulate too many lost tracks
        assert final_stats["pending_detections"] <= 50  # Don't accumulate too many pending

        print(f"Stress test completed: {successful_frames}/{num_frames} frames successful")
        print(f"Final active tracks: {final_stats['active_tracks']}")
        print(f"Final lost tracks: {final_stats['lost_tracks']}")


if __name__ == "__main__":
    # Run a quick stress test when executed directly
    print("Running basic stress test...")

    tracker = SwarmSort()

    # Quick stress test
    for frame in range(50):
        detections = []
        for i in range(20):  # 20 objects
            detection = Detection(
                position=np.random.rand(2) * 300, confidence=0.7 + np.random.rand() * 0.3
            )
            detections.append(detection)

        result = tracker.update(detections)
        print(f"Frame {frame}: {len(result)} tracks")

    final_stats = tracker.get_statistics()
    print(f"Final stats: {final_stats['active_tracks']} active tracks")
    print("Basic stress test completed successfully!")
