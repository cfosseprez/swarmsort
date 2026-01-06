"""
Tests for Assignment Algorithms Module

This module validates the assignment algorithms including:
- Greedy assignment algorithm
- Hungarian (optimal) assignment algorithm
- Hybrid assignment strategy
- Cost threshold handling
- Assignment result validation
"""

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from src.swarmsort.assignment import (
    numba_greedy_assignment,
    hungarian_assignment_wrapper,
    hybrid_assignment,
)

# Create aliases for compatibility
greedy_assignment = numba_greedy_assignment
hungarian_assignment = hungarian_assignment_wrapper


class TestGreedyAssignment:
    """Test suite for greedy assignment algorithm."""

    def test_greedy_assignment_basic(self):
        """Test basic greedy assignment with clear best matches."""
        cost_matrix = np.array(
            [
                [0.5, 10.0, 10.0],  # Det 0 clearly matches Track 0
                [10.0, 0.3, 10.0],  # Det 1 clearly matches Track 1
                [10.0, 10.0, 0.7],  # Det 2 clearly matches Track 2
            ],
            dtype=np.float32,
        )

        matches, unmatched_dets, unmatched_tracks = greedy_assignment(cost_matrix, max_distance=5.0)

        # Should match all optimally since each has a clear best match
        expected_matches = [(0, 0), (1, 1), (2, 2)]
        assert len(matches) == 3
        for match in expected_matches:
            assert match in matches

        assert len(unmatched_dets) == 0
        assert len(unmatched_tracks) == 0

    def test_greedy_assignment_threshold(self):
        """Test that greedy assignment respects distance threshold."""
        cost_matrix = np.array(
            [
                [0.5, 15.0],  # Det 0 matches Track 0 (under threshold)
                [12.0, 8.0],  # Det 1 matches Track 1 (8.0 < 10.0 threshold)
            ],
            dtype=np.float32,
        )

        matches, unmatched_dets, unmatched_tracks = greedy_assignment(cost_matrix, max_distance=10.0)

        # Both detections should match (0->0 at 0.5, 1->1 at 8.0)
        assert len(matches) == 2
        # Check both matches are correct
        assert any((m[0] == 0 and m[1] == 0) for m in matches)
        assert any((m[0] == 1 and m[1] == 1) for m in matches)

        # All matched, so no unmatched
        assert len(unmatched_tracks) == 0
        assert len(unmatched_dets) == 0

    def test_greedy_assignment_competition(self):
        """Test greedy behavior when multiple detections compete for same track."""
        cost_matrix = np.array(
            [
                [0.5, 10.0],  # Det 0 wants Track 0
                [0.3, 10.0],  # Det 1 also wants Track 0 (lower cost)
                [10.0, 0.8],  # Det 2 wants Track 1
            ],
            dtype=np.float32,
        )

        matches, unmatched_dets, unmatched_tracks = greedy_assignment(cost_matrix, max_distance=15.0)

        # Greedy should assign Det 1 to Track 0 (lowest cost)
        # Then Det 2 to Track 1
        # Det 0 is unmatched
        assert (1, 0) in matches
        assert (2, 1) in matches
        assert 0 in unmatched_dets

    def test_greedy_assignment_empty_inputs(self):
        """Test greedy assignment with empty cost matrix."""
        # Empty cost matrix
        cost_matrix = np.array([], dtype=np.float32).reshape(0, 0)
        matches, unmatched_dets, unmatched_tracks = greedy_assignment(cost_matrix, max_distance=10.0)

        assert len(matches) == 0
        assert len(unmatched_dets) == 0
        assert len(unmatched_tracks) == 0

        # No detections
        cost_matrix = np.array([], dtype=np.float32).reshape(0, 3)
        matches, unmatched_dets, unmatched_tracks = greedy_assignment(cost_matrix, max_distance=10.0)

        assert len(matches) == 0
        assert len(unmatched_dets) == 0
        np.testing.assert_array_equal(unmatched_tracks, [0, 1, 2])


class TestHungarianAssignment:
    """Test suite for Hungarian (optimal) assignment algorithm."""

    def test_hungarian_assignment_optimal(self):
        """Test that Hungarian finds optimal assignment."""
        cost_matrix = np.array(
            [
                [4.0, 1.0, 3.0],
                [2.0, 0.0, 5.0],
                [3.0, 2.0, 2.0],
            ],
            dtype=np.float32,
        )

        matches, unmatched_dets, unmatched_tracks = hungarian_assignment(cost_matrix, max_distance=10.0)

        # Hungarian should find global optimum
        # Det 0 -> Track 1 (cost 1)
        # Det 1 -> Track 0 (cost 2)
        # Det 2 -> Track 2 (cost 2)
        # Total cost: 5 (optimal)

        assert len(matches) == 3
        assert (0, 1) in matches
        assert (1, 0) in matches
        assert (2, 2) in matches

    def test_hungarian_assignment_threshold(self):
        """Test Hungarian assignment with distance threshold."""
        cost_matrix = np.array(
            [
                [1.0, 20.0],
                [2.0, 3.0],
                [25.0, 4.0],
            ],
            dtype=np.float32,
        )

        matches, unmatched_dets, unmatched_tracks = hungarian_assignment(cost_matrix, max_distance=10.0)

        # Only matches under threshold should be kept
        assert (0, 0) in matches  # Cost 1.0
        assert (1, 1) in matches  # Cost 3.0

        # Det 2 unmatched (all costs exceed threshold)
        assert 2 in unmatched_dets

    def test_hungarian_vs_greedy_optimality(self):
        """Test that Hungarian finds better solution than greedy in complex case."""
        # Case where greedy makes suboptimal choice
        cost_matrix = np.array(
            [
                [2.0, 4.0, 6.0],
                [3.0, 1.0, 5.0],
                [4.0, 3.0, 1.0],
            ],
            dtype=np.float32,
        )

        greedy_matches, _, _ = greedy_assignment(cost_matrix, max_distance=10.0)
        hungarian_matches, _, _ = hungarian_assignment(cost_matrix, max_distance=10.0)

        # Calculate total costs
        greedy_cost = sum(cost_matrix[d, t] for d, t in greedy_matches)
        hungarian_cost = sum(cost_matrix[d, t] for d, t in hungarian_matches)

        # Hungarian should find equal or better solution
        assert hungarian_cost <= greedy_cost

        # Hungarian should find optimal: (0,0), (1,1), (2,2) with cost 2+1+1=4
        assert hungarian_cost == 4.0

    def test_hungarian_rectangular_matrix(self):
        """Test Hungarian with non-square cost matrix."""
        # More detections than tracks
        cost_matrix = np.array(
            [[1.0, 2.0], [3.0, 1.0], [2.0, 4.0], [5.0, 3.0]], dtype=np.float32  # 4 detections, 2 tracks
        )

        matches, unmatched_dets, unmatched_tracks = hungarian_assignment(cost_matrix, max_distance=10.0)

        # Should match optimally
        assert len(matches) == 2  # Can only match 2 (limited by tracks)
        assert len(unmatched_dets) == 2  # 2 detections unmatched


class TestGreedyAssignmentPerformance:
    """Test suite for greedy assignment with various matrix sizes."""

    def test_greedy_various_matrix_sizes(self):
        """Test that greedy assignment handles various matrix sizes correctly."""
        sizes = [(10, 10), (50, 20), (20, 50), (100, 100)]

        for n_det, n_track in sizes:
            np.random.seed(42)  # For reproducibility
            cost_matrix = np.random.rand(n_det, n_track).astype(np.float32) * 10

            matches, unmatched_d, unmatched_t = numba_greedy_assignment(cost_matrix, max_distance=5.0)

            # Basic sanity checks - get matched detections and tracks
            if len(matches) > 0:
                matched_dets = set(int(m[0]) for m in matches)
                matched_tracks = set(int(m[1]) for m in matches)

                # No detection or track should be matched twice
                assert len(matched_dets) == len(matches), f"Duplicate detection in matches for size {n_det}x{n_track}"
                assert len(matched_tracks) == len(matches), f"Duplicate track in matches for size {n_det}x{n_track}"

    def test_greedy_deterministic(self):
        """Test that greedy assignment is deterministic."""
        np.random.seed(42)
        cost_matrix = np.random.rand(20, 20).astype(np.float32) * 10

        matches1, unmatched_d1, unmatched_t1 = numba_greedy_assignment(cost_matrix, max_distance=5.0)
        matches2, unmatched_d2, unmatched_t2 = numba_greedy_assignment(cost_matrix, max_distance=5.0)

        # Results should be identical
        assert len(matches1) == len(matches2)
        for m1, m2 in zip(matches1, matches2):
            assert m1[0] == m2[0] and m1[1] == m2[1]


class TestHybridAssignment:
    """Test suite for hybrid assignment strategy."""

    def test_hybrid_low_cost_matches_use_greedy(self):
        """Test that hybrid uses greedy phase for low cost matches."""
        # Create matrix with clear low-cost matches
        cost_matrix = np.array([[0.5, 10.0], [10.0, 0.3]], dtype=np.float32)

        matches, unmatched_d, unmatched_t = hybrid_assignment(
            cost_matrix, max_distance=5.0, greedy_threshold=5.0
        )

        # Should match optimally
        assert len(matches) == 2
        assert (0, 0) in matches
        assert (1, 1) in matches

    def test_hybrid_ambiguous_uses_hungarian(self):
        """Test that hybrid uses Hungarian phase for ambiguous matches."""
        # Create matrix where no costs are below greedy_threshold
        # This forces Hungarian algorithm to be used
        np.random.seed(42)
        cost_matrix = np.array([
            [5.0, 5.1, 5.2],
            [5.3, 5.0, 5.4],
            [5.5, 5.6, 5.0]
        ], dtype=np.float32)

        matches, unmatched_d, unmatched_t = hybrid_assignment(
            cost_matrix, max_distance=10.0, greedy_threshold=4.0,  # greedy_threshold < all costs
            hungarian_fallback_threshold=10.0
        )

        # Should still find optimal matches via Hungarian
        assert len(matches) == 3
        # Check that diagonal matches are made (optimal)
        matched_pairs = set(matches)
        assert (0, 0) in matched_pairs
        assert (1, 1) in matched_pairs
        assert (2, 2) in matched_pairs

    def test_hybrid_mixed_greedy_and_hungarian(self):
        """Test hybrid with some obvious and some ambiguous matches."""
        cost_matrix = np.array([
            [0.5, 100.0, 100.0],  # Obvious match to track 0
            [100.0, 6.0, 6.1],   # Ambiguous - needs Hungarian
            [100.0, 6.2, 6.0]    # Ambiguous - needs Hungarian
        ], dtype=np.float32)

        matches, unmatched_d, unmatched_t = hybrid_assignment(
            cost_matrix, max_distance=50.0, greedy_threshold=5.0,
            hungarian_fallback_threshold=50.0
        )

        # Should match all 3
        assert len(matches) == 3
        # Det 0 should definitely match track 0 (obvious)
        assert (0, 0) in matches

    def test_hybrid_respects_max_distance(self):
        """Test that hybrid respects max_distance threshold."""
        cost_matrix = np.array([
            [5.0, 100.0],
            [100.0, 5.0]
        ], dtype=np.float32)

        matches, unmatched_d, unmatched_t = hybrid_assignment(
            cost_matrix, max_distance=10.0, greedy_threshold=10.0,
            hungarian_fallback_threshold=10.0
        )

        # Should match both (costs 5.0 < max_distance 10.0)
        assert len(matches) == 2


class TestAssignmentValidation:
    """Validation tests for assignment algorithm properties."""

    def test_assignment_bijection(self):
        """Test that assignments are valid bijections (one-to-one)."""
        cost_matrix = np.random.rand(10, 8).astype(np.float32) * 10

        for assignment_func in [greedy_assignment, hungarian_assignment]:
            matches, unmatched_d, unmatched_t = assignment_func(cost_matrix, max_distance=15.0)

            # Check no duplicate assignments
            det_assigned = [d for d, t in matches]
            track_assigned = [t for d, t in matches]

            assert len(det_assigned) == len(set(det_assigned))  # All unique
            assert len(track_assigned) == len(set(track_assigned))  # All unique

            # Check completeness
            all_dets = set(range(10))
            all_tracks = set(range(8))

            matched_dets = set(det_assigned)
            matched_tracks = set(track_assigned)

            # Every detection is either matched or unmatched
            assert matched_dets | set(unmatched_d) == all_dets

            # Every track is either matched or unmatched
            assert matched_tracks | set(unmatched_t) == all_tracks

    def test_assignment_respects_threshold(self):
        """Test that no assignment exceeds the distance threshold."""
        cost_matrix = np.random.rand(15, 12).astype(np.float32) * 20
        max_distance = 10.0

        for assignment_func in [greedy_assignment, hungarian_assignment]:
            matches, _, _ = assignment_func(cost_matrix, max_distance=max_distance)

            # Check all matched pairs respect threshold
            for det_idx, track_idx in matches:
                assert cost_matrix[det_idx, track_idx] <= max_distance

    def test_assignment_stability(self):
        """Test that assignment algorithms are stable with identical costs."""
        # Matrix with some identical costs
        cost_matrix = np.array([[1.0, 2.0, 1.0], [2.0, 1.0, 2.0], [1.0, 1.0, 2.0]], dtype=np.float32)

        # Run multiple times
        results = []
        for _ in range(5):
            matches, _, _ = hungarian_assignment(cost_matrix, max_distance=10.0)
            results.append(sorted(matches))

        # Results should be consistent
        for i in range(1, len(results)):
            assert results[i] == results[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])