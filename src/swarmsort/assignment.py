"""
SwarmSort Assignment Algorithms Module

This module contains assignment algorithms for solving the tracking assignment
problem. It includes both Hungarian (optimal) and Greedy (fast) assignment
methods, as well as hybrid approaches for different tracking scenarios.

Functions:
    numba_greedy_assignment: Fast greedy assignment algorithm
    compute_assignment_priorities: Priority computation for assignments
    hungarian_assignment_wrapper: Wrapper for scipy's Hungarian algorithm
"""

# ============================================================================
# STANDARD IMPORTS
# ============================================================================
import numpy as np
import numba as nb
from typing import Tuple, List, Optional
from scipy.optimize import linear_sum_assignment


# ============================================================================
# GREEDY ASSIGNMENT ALGORITHM
# ============================================================================

def numba_greedy_assignment(
    cost_matrix: np.ndarray,
    max_distance: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimized greedy assignment using vectorized operations.

    This function performs greedy assignment by iteratively selecting the
    minimum cost match until all possible matches are made.

    Args:
        cost_matrix: Cost matrix [N_det, N_track]
        max_distance: Maximum allowed distance for matching

    Returns:
        Tuple of (matches, unmatched_detections, unmatched_tracks) where:
            - matches: Array of matched pairs [(det_idx, track_idx), ...]
            - unmatched_detections: Array of unmatched detection indices
            - unmatched_tracks: Array of unmatched track indices
    """
    n_dets, n_tracks = cost_matrix.shape

    # Create a working copy of the cost matrix
    working_matrix = cost_matrix.copy()
    working_matrix[working_matrix > max_distance] = np.inf

    # Track used detections and tracks
    used_dets = np.zeros(n_dets, dtype=bool)
    used_tracks = np.zeros(n_tracks, dtype=bool)

    # Pre-allocate matches array
    max_matches = min(n_dets, n_tracks)
    matches = np.zeros((max_matches, 2), dtype=np.int32)
    n_matches = 0

    # Greedy assignment using vectorized operations
    for _ in range(max_matches):
        # Find global minimum in the working matrix
        min_flat_idx = np.argmin(working_matrix)
        min_cost = working_matrix.flat[min_flat_idx]

        if min_cost == np.inf:
            break

        # Convert flat index to 2D coordinates
        best_det = min_flat_idx // n_tracks
        best_track = min_flat_idx % n_tracks

        # Record the match
        matches[n_matches, 0] = best_det
        matches[n_matches, 1] = best_track
        n_matches += 1

        # Mark as used by setting entire row and column to infinity
        working_matrix[best_det, :] = np.inf
        working_matrix[:, best_track] = np.inf

        used_dets[best_det] = True
        used_tracks[best_track] = True

    # Trim matches array to actual size
    matches_array = matches[:n_matches]

    # Build unmatched arrays
    unmatched_dets = np.where(~used_dets)[0].astype(np.int32)
    unmatched_tracks = np.where(~used_tracks)[0].astype(np.int32)

    return matches_array, unmatched_dets, unmatched_tracks


@nb.njit(fastmath=True, cache=True)
def compute_assignment_priorities(
    cost_matrix: np.ndarray,
    max_distance: float
) -> np.ndarray:
    """
    Compute priority scores for greedy assignment.

    Lower scores indicate higher priority (better matches should be assigned first).

    Args:
        cost_matrix: Cost matrix [N_det, N_track]
        max_distance: Maximum allowed distance

    Returns:
        Priority matrix with same shape as cost_matrix
    """
    n_dets, n_tracks = cost_matrix.shape
    priorities = np.copy(cost_matrix)

    # For each cell, priority is the cost plus a penalty based on
    # how many better alternatives exist
    for i in range(n_dets):
        for j in range(n_tracks):
            if cost_matrix[i, j] > max_distance:
                priorities[i, j] = np.inf
                continue

            base_cost = cost_matrix[i, j]

            # Count better alternatives for this detection
            better_for_det = 0
            for k in range(n_tracks):
                if k != j and cost_matrix[i, k] < base_cost:
                    better_for_det += 1

            # Count better alternatives for this track
            better_for_track = 0
            for k in range(n_dets):
                if k != i and cost_matrix[k, j] < base_cost:
                    better_for_track += 1

            # Priority considers both the cost and alternatives
            # Lower priority value = higher priority assignment
            alternative_penalty = 0.1 * (better_for_det + better_for_track)
            priorities[i, j] = base_cost + alternative_penalty

    return priorities


# ============================================================================
# HUNGARIAN ASSIGNMENT
# ============================================================================

def hungarian_assignment_wrapper(
    cost_matrix: np.ndarray,
    max_distance: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Wrapper for scipy's Hungarian algorithm with distance thresholding.

    Args:
        cost_matrix: Cost matrix [N_det, N_track]
        max_distance: Maximum allowed distance for matching

    Returns:
        Tuple of (matches, unmatched_detections, unmatched_tracks)
    """
    n_dets, n_tracks = cost_matrix.shape

    # Handle empty cases
    if n_dets == 0:
        return [], [], list(range(n_tracks))
    if n_tracks == 0:
        return [], list(range(n_dets)), []

    # Check if all costs are infinite
    if np.all(np.isinf(cost_matrix)):
        return [], list(range(n_dets)), list(range(n_tracks))

    # Prepare cost matrix for Hungarian algorithm
    working_matrix = cost_matrix.copy()
    working_matrix[working_matrix > max_distance] = max_distance * 2

    try:
        det_indices, track_indices = linear_sum_assignment(working_matrix)
    except ValueError:
        # Assignment failed
        return [], list(range(n_dets)), list(range(n_tracks))

    # Filter valid matches
    matches = []
    for d_idx, t_idx in zip(det_indices, track_indices):
        if cost_matrix[d_idx, t_idx] <= max_distance:
            matches.append((d_idx, t_idx))

    # Find unmatched - OPTIMIZED: Use numpy mask instead of set lookup
    matched_det_mask = np.zeros(n_dets, dtype=bool)
    matched_track_mask = np.zeros(n_tracks, dtype=bool)
    for d_idx, t_idx in matches:
        matched_det_mask[d_idx] = True
        matched_track_mask[t_idx] = True

    unmatched_dets = np.where(~matched_det_mask)[0].tolist()
    unmatched_tracks = np.where(~matched_track_mask)[0].tolist()

    return matches, unmatched_dets, unmatched_tracks


# ============================================================================
# HYBRID ASSIGNMENT
# ============================================================================

def hybrid_assignment(
    cost_matrix: np.ndarray,
    max_distance: float,
    greedy_threshold: float = 30.0,
    hungarian_fallback_threshold: float = 100.0
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Hybrid assignment using greedy for obvious matches and Hungarian for remainder.

    This function first performs greedy assignment on high-confidence (low-cost)
    matches, then uses Hungarian algorithm for the remaining ambiguous cases.

    Args:
        cost_matrix: Cost matrix [N_det, N_track]
        max_distance: Maximum allowed distance
        greedy_threshold: Threshold for greedy assignment
        hungarian_fallback_threshold: Threshold for Hungarian assignment

    Returns:
        Tuple of (matches, unmatched_detections, unmatched_tracks)
    """
    n_dets, n_tracks = cost_matrix.shape

    # Handle empty cases
    if n_dets == 0:
        return [], [], list(range(n_tracks))
    if n_tracks == 0:
        return [], list(range(n_dets)), []

    # Phase 1: Greedy assignment for obvious matches
    greedy_matches = []
    used_dets = set()
    used_tracks = set()

    # Find all costs below greedy threshold
    low_cost_indices = np.where(cost_matrix < greedy_threshold)

    if len(low_cost_indices[0]) > 0:
        # Sort by cost
        costs = cost_matrix[low_cost_indices]
        sorted_idx = np.argsort(costs)

        for idx in sorted_idx:
            det_idx = low_cost_indices[0][idx]
            track_idx = low_cost_indices[1][idx]

            if det_idx not in used_dets and track_idx not in used_tracks:
                greedy_matches.append((det_idx, track_idx))
                used_dets.add(det_idx)
                used_tracks.add(track_idx)

    # Phase 2: Hungarian assignment for remaining
    # OPTIMIZED: Use numpy mask instead of set lookup
    used_det_mask = np.zeros(n_dets, dtype=bool)
    used_track_mask = np.zeros(n_tracks, dtype=bool)
    for d_idx, t_idx in greedy_matches:
        used_det_mask[d_idx] = True
        used_track_mask[t_idx] = True

    remaining_dets = np.where(~used_det_mask)[0].tolist()
    remaining_tracks = np.where(~used_track_mask)[0].tolist()

    hungarian_matches = []
    if remaining_dets and remaining_tracks:
        # Create reduced cost matrix
        reduced_cost_matrix = cost_matrix[np.ix_(remaining_dets, remaining_tracks)]

        # Only run Hungarian if there are valid assignments possible
        if not np.all(np.isinf(reduced_cost_matrix)):
            # Cap infinite costs for Hungarian algorithm
            finite_costs = reduced_cost_matrix[~np.isinf(reduced_cost_matrix)]
            if len(finite_costs) > 0:
                max_finite_cost = np.max(finite_costs)
                reduced_cost_matrix[np.isinf(reduced_cost_matrix)] = max_finite_cost * 2

                try:
                    hun_det_indices, hun_track_indices = linear_sum_assignment(reduced_cost_matrix)

                    # Convert back to original indices and filter valid matches
                    for i, j in zip(hun_det_indices, hun_track_indices):
                        original_det_idx = remaining_dets[i]
                        original_track_idx = remaining_tracks[j]
                        original_cost = cost_matrix[original_det_idx, original_track_idx]

                        if original_cost <= hungarian_fallback_threshold:
                            hungarian_matches.append((original_det_idx, original_track_idx))

                except ValueError:
                    pass  # Hungarian failed, continue without these matches

    # Combine results
    all_matches = greedy_matches + hungarian_matches

    # OPTIMIZED: Use numpy mask instead of set lookup
    matched_det_mask = np.zeros(n_dets, dtype=bool)
    matched_track_mask = np.zeros(n_tracks, dtype=bool)
    for d_idx, t_idx in all_matches:
        matched_det_mask[d_idx] = True
        matched_track_mask[t_idx] = True

    unmatched_dets = np.where(~matched_det_mask)[0].tolist()
    unmatched_tracks = np.where(~matched_track_mask)[0].tolist()

    return all_matches, unmatched_dets, unmatched_tracks