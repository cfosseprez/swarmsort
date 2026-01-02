#!/usr/bin/env python
"""
Comprehensive Test Suite for SwarmSort Tracker

This module provides a comprehensive test suite that validates all major
functionality of the SwarmSort tracking system in an integrated manner.
"""

import os
os.environ['LOGURU_LEVEL'] = 'ERROR'

import numpy as np
import pytest
from src.swarmsort import SwarmSortTracker, SwarmSortConfig, Detection


class TestBasicTracking:
    """Test basic tracking functionality."""

    def test_single_object_tracking(self):
        """Test tracking a single object across multiple frames."""
        config = SwarmSortConfig(
            min_consecutive_detections=1,
            max_distance=50.0
        )
        tracker = SwarmSortTracker(config)

        # Track object moving linearly
        track_id = None
        for frame in range(10):
            pos = np.array([10.0 + frame * 5, 20.0], dtype=np.float32)
            det = Detection(position=pos, confidence=0.9)

            tracks = tracker.update([det])

            # With min_consecutive_detections=1, track is created immediately
            assert len(tracks) == 1
            if track_id is None:
                track_id = tracks[0].id
            else:
                assert tracks[0].id == track_id  # Same track ID

    def test_multiple_object_tracking(self):
        """Test tracking multiple objects simultaneously."""
        config = SwarmSortConfig(
            min_consecutive_detections=1,
            max_distance=50.0
        )
        tracker = SwarmSortTracker(config)

        # Track two objects
        track_ids = set()
        for frame in range(5):
            detections = [
                Detection(
                    position=np.array([20.0 + frame * 2, 30.0], dtype=np.float32),
                    confidence=0.9
                ),
                Detection(
                    position=np.array([80.0 - frame * 2, 70.0], dtype=np.float32),
                    confidence=0.8
                )
            ]

            tracks = tracker.update(detections)

            if frame > 0:  # After initialization
                assert len(tracks) == 2
                for track in tracks:
                    track_ids.add(track.id)

        assert len(track_ids) == 2  # Two unique track IDs

    def test_track_loss_and_recovery(self):
        """Test that tracks are maintained during brief occlusions."""
        config = SwarmSortConfig(
            min_consecutive_detections=1,
            max_track_age=3,  # Allow 3 frames of missing
            max_distance=50.0
        )
        tracker = SwarmSortTracker(config)

        # Establish track
        original_id = None
        for frame in range(3):
            det = Detection(
                position=np.array([50.0, 50.0], dtype=np.float32),
                confidence=0.9
            )
            tracks = tracker.update([det])
            if len(tracks) > 0:
                original_id = tracks[0].id

        # Lose track for 2 frames
        for _ in range(2):
            tracks = tracker.update([])
            assert len(tracks) == 0  # No visible tracks

        # Recover track
        det = Detection(
            position=np.array([55.0, 50.0], dtype=np.float32),
            confidence=0.9
        )
        tracks = tracker.update([det])

        assert len(tracks) == 1
        assert tracks[0].id == original_id  # Same track recovered


class TestEmbeddingTracking:
    """Test embedding-based tracking functionality."""

    def test_embedding_matching(self):
        """Test that embeddings improve tracking accuracy."""
        config = SwarmSortConfig(
            do_embeddings=True,
            embedding_weight=0.5,
            min_consecutive_detections=1,
            max_distance=100.0
        )
        tracker = SwarmSortTracker(config)

        # Create consistent embedding for object
        base_emb = np.random.randn(128).astype(np.float32)
        base_emb /= np.linalg.norm(base_emb)

        # Track with consistent embedding
        track_id = None
        for frame in range(5):
            # Add small noise to embedding
            emb = base_emb + np.random.randn(128) * 0.05
            emb = emb.astype(np.float32)
            emb /= np.linalg.norm(emb)

            det = Detection(
                position=np.array([50.0 + frame * 10, 50.0], dtype=np.float32),
                confidence=0.9,
                embedding=emb
            )

            tracks = tracker.update([det])

            if len(tracks) > 0:
                if track_id is None:
                    track_id = tracks[0].id
                else:
                    assert tracks[0].id == track_id  # Maintains same ID

    def test_reid_functionality(self):
        """Test re-identification of lost tracks."""
        config = SwarmSortConfig(
            do_embeddings=True,
            reid_enabled=True,
            reid_min_frames_lost=2,
            reid_max_distance=100.0,
            min_consecutive_detections=1,
            max_track_age=5
        )
        tracker = SwarmSortTracker(config)

        # Create unique embedding
        base_emb = np.random.randn(128).astype(np.float32)
        base_emb /= np.linalg.norm(base_emb)

        # Establish track
        original_id = None
        for frame in range(3):
            emb = base_emb + np.random.randn(128) * 0.02
            emb = emb.astype(np.float32)
            emb /= np.linalg.norm(emb)

            det = Detection(
                position=np.array([50.0, 50.0], dtype=np.float32),
                confidence=0.9,
                embedding=emb
            )
            tracks = tracker.update([det])
            if len(tracks) > 0:
                original_id = tracks[0].id

        # Lose track for reid_min_frames_lost + 1
        for _ in range(3):
            tracker.update([])

        # Re-detect with similar embedding
        emb = base_emb + np.random.randn(128) * 0.05
        emb = emb.astype(np.float32)
        emb /= np.linalg.norm(emb)

        det = Detection(
            position=np.array([60.0, 55.0], dtype=np.float32),
            confidence=0.9,
            embedding=emb
        )
        tracks = tracker.update([det])

        if tracks:
            # Check if ReID worked (same ID)
            assert tracks[0].id == original_id


class TestAdvancedFeatures:
    """Test advanced tracking features."""

    def test_deduplication(self):
        """Test that duplicate detections are merged."""
        config = SwarmSortConfig(
            do_embeddings=True,
            deduplication_distance=15.0,
            min_consecutive_detections=1
        )
        tracker = SwarmSortTracker(config)

        # Create duplicate detections (overlapping)
        detections = [
            Detection(
                position=np.array([50.0, 50.0], dtype=np.float32),
                confidence=0.95,
                embedding=np.random.randn(128).astype(np.float32)
            ),
            Detection(
                position=np.array([52.0, 51.0], dtype=np.float32),  # Very close
                confidence=0.90,
                embedding=np.random.randn(128).astype(np.float32)
            ),
            Detection(
                position=np.array([100.0, 100.0], dtype=np.float32),  # Far away
                confidence=0.85,
                embedding=np.random.randn(128).astype(np.float32)
            )
        ]

        # Update twice to establish tracks
        tracker.update(detections)
        tracks = tracker.update(detections)

        # Should have 2 tracks (duplicates merged)
        assert len(tracks) == 2

    def test_collision_detection(self):
        """Test that collision detection is enabled and tracks objects during close proximity."""
        config = SwarmSortConfig(
            do_embeddings=True,
            collision_freeze_embeddings=True,
            collision_safety_distance=30.0,
            min_consecutive_detections=1
        )
        tracker = SwarmSortTracker(config)

        # Track two objects approaching each other
        track_count_during_collision = 0
        for frame in range(15):
            distance = 100 - frame * 6  # Start at 100, decrease

            detections = [
                Detection(
                    position=np.array([50.0, 50.0], dtype=np.float32),
                    confidence=0.9,
                    embedding=np.random.randn(128).astype(np.float32)
                ),
                Detection(
                    position=np.array([50.0 + distance, 50.0], dtype=np.float32),
                    confidence=0.9,
                    embedding=np.random.randn(128).astype(np.float32)
                )
            ]

            tracks = tracker.update(detections)

            # During collision (distance < safety), should still maintain 2 tracks
            if distance < config.collision_safety_distance and len(tracks) == 2:
                track_count_during_collision += 1

        # Should maintain both tracks during collision frames
        assert track_count_during_collision > 0

    def test_kalman_filtering(self):
        """Test that Kalman filtering maintains consistent tracking."""
        config = SwarmSortConfig(
            kalman_type="simple",
            min_consecutive_detections=1
        )
        tracker = SwarmSortTracker(config)

        # Track object with noisy measurements
        tracked_positions = []
        np.random.seed(42)  # Fixed seed for reproducibility

        for frame in range(20):
            # True position (linear motion)
            true_pos = np.array([10.0 + frame * 5, 50.0], dtype=np.float32)

            # Add measurement noise
            noise = np.random.randn(2).astype(np.float32) * 3
            measured_pos = true_pos + noise

            det = Detection(position=measured_pos, confidence=0.9)
            tracks = tracker.update([det])

            if len(tracks) > 0:
                tracked_positions.append(tracks[0].position.copy())

        # Should maintain single track throughout
        assert len(tracked_positions) >= 15  # Track established and maintained

        # Tracked positions should roughly follow the true trajectory
        if len(tracked_positions) > 5:
            first_pos = tracked_positions[0]
            last_pos = tracked_positions[-1]
            # Object should have moved in positive X direction
            assert last_pos[0] > first_pos[0]


class TestConfigurationModes:
    """Test different configuration modes."""

    def test_assignment_strategies(self):
        """Test different assignment strategies."""
        strategies = ["hungarian", "greedy", "hybrid"]

        for strategy in strategies:
            config = SwarmSortConfig(
                assignment_strategy=strategy,
                min_consecutive_detections=1,
                max_distance=100.0  # Large enough to not reject associations
            )
            tracker = SwarmSortTracker(config)

            # Simple tracking scenario with well-separated objects
            detections = [
                Detection(
                    position=np.array([i * 50.0, 50.0], dtype=np.float32),
                    confidence=0.9
                )
                for i in range(3)
            ]

            # Run multiple frames to establish tracks
            for _ in range(3):
                tracks = tracker.update(detections)

            # All strategies should be able to track objects
            assert len(tracks) >= 2, f"Strategy {strategy} should track at least 2 objects"

    def test_embedding_methods(self):
        """Test different embedding matching methods."""
        methods = ["last", "average", "weighted_average", "best_match"]

        for method in methods:
            config = SwarmSortConfig(
                do_embeddings=True,
                embedding_matching_method=method,
                min_consecutive_detections=1
            )
            tracker = SwarmSortTracker(config)

            # Track with embeddings
            base_emb = np.random.randn(128).astype(np.float32)
            base_emb /= np.linalg.norm(base_emb)

            for frame in range(5):
                emb = base_emb + np.random.randn(128) * 0.05
                emb = emb.astype(np.float32)
                emb /= np.linalg.norm(emb)

                det = Detection(
                    position=np.array([50.0 + frame * 2, 50.0], dtype=np.float32),
                    confidence=0.9,
                    embedding=emb
                )
                tracks = tracker.update([det])

            # Should maintain single track with all methods
            assert len(tracker._tracks) == 1


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("COMPREHENSIVE SWAMSORT TRACKER TEST SUITE")
    print("=" * 60)

    test_classes = [
        TestBasicTracking,
        TestEmbeddingTracking,
        TestAdvancedFeatures,
        TestConfigurationModes
    ]

    total_passed = 0
    total_failed = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 40)

        test_instance = test_class()
        test_methods = [m for m in dir(test_instance) if m.startswith("test_")]

        for method_name in test_methods:
            test_method = getattr(test_instance, method_name)
            try:
                test_method()
                print(f"  [OK] {method_name}")
                total_passed += 1
            except AssertionError as e:
                print(f"  [FAIL] {method_name}: {e}")
                total_failed += 1
            except Exception as e:
                print(f"  [ERROR] {method_name}: Unexpected error - {e}")
                total_failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {total_passed} passed, {total_failed} failed")
    print("=" * 60)

    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)