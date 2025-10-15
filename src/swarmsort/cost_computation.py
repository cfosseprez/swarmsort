"""
SwarmSort Cost Computation Module

This module contains functions for computing cost matrices and distance metrics
used in the tracking assignment problem. It includes various cost computation
strategies including probabilistic, embedding-based, and OC-SORT style costs.

Functions:
    cosine_similarity_normalized: Normalized cosine similarity for embeddings
    compute_embedding_distances_with_method: Multi-method embedding distance computation
    compute_cost_matrix_vectorized: Standard vectorized cost matrix computation
    compute_probabilistic_cost_matrix_vectorized: Probabilistic cost with covariances
    compute_freeze_flags_vectorized: Collision detection for embedding freezing
"""

# ============================================================================
# STANDARD IMPORTS
# ============================================================================
import numpy as np
import numba as nb
from typing import Optional, Tuple
import math


# ============================================================================
# EMBEDDING DISTANCE FUNCTIONS
# ============================================================================

def cosine_similarity_normalized(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Fast cosine similarity normalized to [0, 1] distance.

    Args:
        emb1: First embedding vector
        emb2: Second embedding vector

    Returns:
        Distance in range [0, 1] where 0 is identical and 1 is opposite
    """
    norm1 = np.sqrt(np.sum(emb1 * emb1))
    norm2 = np.sqrt(np.sum(emb2 * emb2))

    if norm1 == 0 or norm2 == 0:
        return 1.0

    cos_sim = np.sum(emb1 * emb2) / (norm1 * norm2)
    return (1.0 - cos_sim) / 2.0


@nb.njit(fastmath=True, cache=True)
def fast_mahalanobis_distance(diff: np.ndarray, cov_inv: np.ndarray) -> float:
    """
    Fast 2D Mahalanobis distance computation.

    Args:
        diff: Difference vector [2]
        cov_inv: Inverse covariance matrix [2, 2]

    Returns:
        Mahalanobis distance
    """
    return np.sqrt(
        diff[0] * (cov_inv[0, 0] * diff[0] + cov_inv[0, 1] * diff[1])
        + diff[1] * (cov_inv[1, 0] * diff[0] + cov_inv[1, 1] * diff[1])
    )


@nb.njit(fastmath=True, parallel=False, cache=True)
def compute_embedding_distances_optimized(det_embeddings, track_embeddings):
    """
    Optimized embedding distance computation using dot product.

    Args:
        det_embeddings: Detection embeddings [N_det, emb_dim]
        track_embeddings: Track embeddings [N_track, emb_dim]

    Returns:
        Distance matrix [N_det, N_track]
    """
    n_dets, emb_dim = det_embeddings.shape
    n_tracks = track_embeddings.shape[0]

    distances = np.empty((n_dets, n_tracks), dtype=np.float32)

    for i in range(n_dets):
        det_emb = det_embeddings[i]
        for j in range(n_tracks):
            track_emb = track_embeddings[j]

            # Inline dot product for better cache performance
            dot_product = 0.0
            for k in range(emb_dim):
                dot_product += det_emb[k] * track_emb[k]

            # Convert to distance
            distances[i, j] = (1.0 - dot_product) / 2.0

    return distances


@nb.njit(fastmath=True, cache=True)
def compute_embedding_distances_with_method(
    det_embeddings: np.ndarray,
    track_embeddings_flat: np.ndarray,
    track_counts: np.ndarray,
    method: int  # 0=last, 1=average, 2=weighted_average, 3=best_match
) -> np.ndarray:
    """
    Compute embedding distances using various methods for historical embeddings.

    This function computes distances between detection embeddings and track
    embedding histories using different aggregation methods.

    Args:
        det_embeddings: Detection embeddings [N_det, emb_dim]
        track_embeddings_flat: Flattened track embedding histories
        track_counts: Number of embeddings per track
        method: Method for aggregating embedding distances
            0: Use last embedding only
            1: Average all distances
            2: Weighted average (recent embeddings have more weight)
            3: Best match (minimum distance)

    Returns:
        Distance matrix [N_det, N_track]
    """
    n_dets = det_embeddings.shape[0]
    n_tracks = len(track_counts)
    emb_dim = det_embeddings.shape[1]

    distances = np.zeros((n_dets, n_tracks), dtype=np.float32)

    # Compute track offsets for flat array
    track_offsets = np.zeros(n_tracks + 1, dtype=np.int32)
    for t in range(n_tracks):
        track_offsets[t + 1] = track_offsets[t] + track_counts[t]

    for i in range(n_dets):
        det_emb = det_embeddings[i]

        for j in range(n_tracks):
            if track_counts[j] == 0:
                distances[i, j] = 1.0
                continue

            start_idx = track_offsets[j] * emb_dim
            end_idx = track_offsets[j + 1] * emb_dim

            if method == 0:  # last
                # Use only the last embedding
                last_emb_start = end_idx - emb_dim
                dot_product = 0.0
                for k in range(emb_dim):
                    dot_product += det_emb[k] * track_embeddings_flat[last_emb_start + k]
                distances[i, j] = (1.0 - dot_product) / 2.0

            elif method == 1:  # average
                # Average distance to all embeddings
                total_dist = 0.0
                for emb_idx in range(track_offsets[j], track_offsets[j + 1]):
                    emb_start = emb_idx * emb_dim
                    dot_product = 0.0
                    for k in range(emb_dim):
                        dot_product += det_emb[k] * track_embeddings_flat[emb_start + k]
                    total_dist += (1.0 - dot_product) / 2.0
                distances[i, j] = total_dist / track_counts[j]

            elif method == 2:  # weighted_average
                # Weighted average (recent embeddings have more weight)
                total_dist = 0.0
                total_weight = 0.0
                for idx, emb_idx in enumerate(range(track_offsets[j], track_offsets[j + 1])):
                    weight = float(idx + 1)  # Recent embeddings have higher weight
                    emb_start = emb_idx * emb_dim
                    dot_product = 0.0
                    for k in range(emb_dim):
                        dot_product += det_emb[k] * track_embeddings_flat[emb_start + k]
                    total_dist += weight * (1.0 - dot_product) / 2.0
                    total_weight += weight
                distances[i, j] = total_dist / total_weight if total_weight > 0 else 1.0

            elif method == 3:  # best_match
                # Minimum distance to any embedding
                min_dist = 1.0
                for emb_idx in range(track_offsets[j], track_offsets[j + 1]):
                    emb_start = emb_idx * emb_dim
                    dot_product = 0.0
                    for k in range(emb_dim):
                        dot_product += det_emb[k] * track_embeddings_flat[emb_start + k]
                    dist = (1.0 - dot_product) / 2.0
                    if dist < min_dist:
                        min_dist = dist
                distances[i, j] = min_dist

    return distances


# ============================================================================
# COST MATRIX COMPUTATION FUNCTIONS
# ============================================================================

@nb.njit(fastmath=True, cache=True)
def compute_cost_matrix_vectorized(
    det_positions: np.ndarray,
    track_predicted_positions: np.ndarray,
    track_last_positions: np.ndarray,
    track_misses: np.ndarray,
    scaled_embedding_distances: np.ndarray,
    embedding_weight: float,
    max_distance: float,
    do_embeddings: bool
) -> np.ndarray:
    """
    Compute standard cost matrix with position and optional embedding costs.

    Args:
        det_positions: Detection positions [N_det, 2]
        track_predicted_positions: Predicted track positions [N_track, 2]
        track_last_positions: Last observed track positions [N_track, 2]
        track_misses: Number of misses for each track
        scaled_embedding_distances: Pre-computed embedding distances
        embedding_weight: Weight for embedding term (0-1)
        max_distance: Maximum allowed distance
        do_embeddings: Whether to include embedding costs

    Returns:
        Cost matrix [N_det, N_track]
    """
    n_dets = det_positions.shape[0]
    n_tracks = track_predicted_positions.shape[0]

    cost_matrix = np.full((n_dets, n_tracks), np.inf, dtype=np.float32)

    for i in range(n_dets):
        for j in range(n_tracks):
            # Use predicted position for recently seen tracks
            if track_misses[j] < 3:
                dx = det_positions[i, 0] - track_predicted_positions[j, 0]
                dy = det_positions[i, 1] - track_predicted_positions[j, 1]
            else:
                # Use last known position for lost tracks
                dx = det_positions[i, 0] - track_last_positions[j, 0]
                dy = det_positions[i, 1] - track_last_positions[j, 1]

            pos_distance = np.sqrt(dx * dx + dy * dy)

            if pos_distance > max_distance:
                continue

            if do_embeddings:
                # Combined cost with embedding
                emb_dist = scaled_embedding_distances[i, j]
                normalized_pos_dist = pos_distance / max_distance
                cost_matrix[i, j] = (
                    (1.0 - embedding_weight) * normalized_pos_dist +
                    embedding_weight * emb_dist
                ) * max_distance
            else:
                cost_matrix[i, j] = pos_distance

    return cost_matrix


@nb.njit(fastmath=True, cache=True)
def compute_probabilistic_cost_matrix_vectorized(
    det_positions: np.ndarray,
    track_predicted_positions: np.ndarray,
    track_last_positions: np.ndarray,
    track_misses: np.ndarray,
    track_covariances: np.ndarray,
    scaled_embedding_distances: np.ndarray,
    embedding_weight: float,
    max_distance: float,
    do_embeddings: bool
) -> np.ndarray:
    """
    Compute probabilistic cost matrix using Mahalanobis distance.

    Args:
        det_positions: Detection positions [N_det, 2]
        track_predicted_positions: Predicted track positions [N_track, 2]
        track_last_positions: Last observed track positions [N_track, 2]
        track_misses: Number of misses for each track
        track_covariances: Track covariance matrices [N_track, 2, 2]
        scaled_embedding_distances: Pre-computed embedding distances
        embedding_weight: Weight for embedding term
        max_distance: Maximum allowed distance
        do_embeddings: Whether to include embedding costs

    Returns:
        Cost matrix [N_det, N_track]
    """
    n_dets = det_positions.shape[0]
    n_tracks = track_predicted_positions.shape[0]

    cost_matrix = np.full((n_dets, n_tracks), np.inf, dtype=np.float32)

    for i in range(n_dets):
        for j in range(n_tracks):
            # Use predicted position for recently seen tracks
            if track_misses[j] < 3:
                diff = det_positions[i] - track_predicted_positions[j]
            else:
                diff = det_positions[i] - track_last_positions[j]

            # Euclidean distance for gating
            euclidean_dist = np.sqrt(diff[0] * diff[0] + diff[1] * diff[1])
            if euclidean_dist > max_distance * 1.5:
                continue

            # Get covariance for this track
            cov = track_covariances[j]

            # Time-dependent covariance inflation
            time_factor = 1.0 + 0.1 * track_misses[j]
            inflated_cov = cov * time_factor

            # Compute inverse covariance
            det = inflated_cov[0, 0] * inflated_cov[1, 1] - inflated_cov[0, 1] * inflated_cov[1, 0]
            if abs(det) < 1e-6:
                # Singular covariance, use Euclidean distance
                mahal_dist = euclidean_dist
            else:
                cov_inv = np.zeros((2, 2), dtype=np.float32)
                cov_inv[0, 0] = inflated_cov[1, 1] / det
                cov_inv[0, 1] = -inflated_cov[0, 1] / det
                cov_inv[1, 0] = -inflated_cov[1, 0] / det
                cov_inv[1, 1] = inflated_cov[0, 0] / det

                # Mahalanobis distance
                mahal_dist = fast_mahalanobis_distance(diff, cov_inv)

            # Normalize to be comparable with max_distance
            normalized_mahal = mahal_dist * 20.0

            if normalized_mahal > max_distance:
                continue

            if do_embeddings:
                emb_dist = scaled_embedding_distances[i, j]
                normalized_dist = normalized_mahal / max_distance
                cost_matrix[i, j] = (
                    (1.0 - embedding_weight) * normalized_dist +
                    embedding_weight * emb_dist
                ) * max_distance
            else:
                cost_matrix[i, j] = normalized_mahal

    return cost_matrix


@nb.njit(fastmath=True, cache=True, parallel=True)
def compute_cost_matrix_vectorized_parallel(
    det_positions: np.ndarray,
    track_predicted_positions: np.ndarray,
    track_last_positions: np.ndarray,
    track_misses: np.ndarray,
    scaled_embedding_distances: np.ndarray,
    embedding_weight: float,
    max_distance: float,
    do_embeddings: bool
) -> np.ndarray:
    """
    Parallel version of cost matrix computation for large-scale tracking.

    Args:
        det_positions: Detection positions [N_det, 2]
        track_predicted_positions: Predicted track positions [N_track, 2]
        track_last_positions: Last observed track positions [N_track, 2]
        track_misses: Number of misses for each track
        scaled_embedding_distances: Pre-computed embedding distances
        embedding_weight: Weight for embedding term
        max_distance: Maximum allowed distance
        do_embeddings: Whether to include embedding costs

    Returns:
        Cost matrix [N_det, N_track]
    """
    n_dets = det_positions.shape[0]
    n_tracks = track_predicted_positions.shape[0]

    cost_matrix = np.full((n_dets, n_tracks), np.inf, dtype=np.float32)

    for i in nb.prange(n_dets):
        for j in range(n_tracks):
            if track_misses[j] < 3:
                dx = det_positions[i, 0] - track_predicted_positions[j, 0]
                dy = det_positions[i, 1] - track_predicted_positions[j, 1]
            else:
                dx = det_positions[i, 0] - track_last_positions[j, 0]
                dy = det_positions[i, 1] - track_last_positions[j, 1]

            pos_distance = np.sqrt(dx * dx + dy * dy)

            if pos_distance > max_distance:
                continue

            if do_embeddings:
                emb_dist = scaled_embedding_distances[i, j]
                normalized_pos_dist = pos_distance / max_distance
                cost_matrix[i, j] = (
                    (1.0 - embedding_weight) * normalized_pos_dist +
                    embedding_weight * emb_dist
                ) * max_distance
            else:
                cost_matrix[i, j] = pos_distance

    return cost_matrix


@nb.njit(fastmath=True, cache=True)
def compute_cost_matrix_with_uncertainty(
    det_positions: np.ndarray,
    track_predicted_positions: np.ndarray,
    track_last_positions: np.ndarray,
    track_misses: np.ndarray,
    track_uncertainties: np.ndarray,
    uncertainty_weight: float,
    max_distance: float
) -> np.ndarray:
    """
    Compute cost matrix with uncertainty penalties for tracks.

    Args:
        det_positions: Detection positions [N_det, 2]
        track_predicted_positions: Predicted track positions [N_track, 2]
        track_last_positions: Last observed track positions [N_track, 2]
        track_misses: Number of misses for each track
        track_uncertainties: Uncertainty values for tracks
        uncertainty_weight: Weight for uncertainty penalty
        max_distance: Maximum allowed distance

    Returns:
        Cost matrix with uncertainty penalties
    """
    n_dets = det_positions.shape[0]
    n_tracks = track_predicted_positions.shape[0]
    cost_matrix = np.full((n_dets, n_tracks), np.inf, dtype=np.float32)

    for i in range(n_dets):
        for j in range(n_tracks):
            if track_misses[j] < 3:
                diff = det_positions[i] - track_predicted_positions[j]
            else:
                diff = det_positions[i] - track_last_positions[j]

            distance = np.sqrt(diff[0] * diff[0] + diff[1] * diff[1])

            if distance <= max_distance:
                # Add uncertainty penalty
                uncertainty_penalty = uncertainty_weight * track_uncertainties[j]
                cost_matrix[i, j] = distance + uncertainty_penalty

    return cost_matrix


@nb.njit(fastmath=True, cache=True)
def compute_cost_matrix_with_min_distance(
    det_positions: np.ndarray,
    track_predicted_positions: np.ndarray,
    track_last_positions: np.ndarray,
    track_misses: np.ndarray,
    max_distance: float,
    min_distance: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cost matrix and minimum distances for each detection.

    Args:
        det_positions: Detection positions [N_det, 2]
        track_predicted_positions: Predicted track positions [N_track, 2]
        track_last_positions: Last observed track positions [N_track, 2]
        track_misses: Number of misses for each track
        max_distance: Maximum allowed distance
        min_distance: Minimum distance threshold

    Returns:
        Tuple of (cost_matrix, min_distances_per_detection)
    """
    n_dets = det_positions.shape[0]
    n_tracks = track_predicted_positions.shape[0]

    cost_matrix = np.full((n_dets, n_tracks), np.inf, dtype=np.float32)
    min_distances = np.full(n_dets, np.inf, dtype=np.float32)

    for i in range(n_dets):
        for j in range(n_tracks):
            if track_misses[j] < 3:
                diff = det_positions[i] - track_predicted_positions[j]
            else:
                diff = det_positions[i] - track_last_positions[j]

            distance = np.sqrt(diff[0] * diff[0] + diff[1] * diff[1])

            if distance <= max_distance:
                cost_matrix[i, j] = distance
                if distance < min_distances[i]:
                    min_distances[i] = distance

    return cost_matrix, min_distances


# ============================================================================
# COLLISION AND FREEZE DETECTION
# ============================================================================

@nb.njit(fastmath=True, cache=True)
def compute_freeze_flags_vectorized(
    positions: np.ndarray,
    safety_distance: float
) -> np.ndarray:
    """
    Compute freeze flags for tracks based on proximity (collision detection).

    Args:
        positions: Track positions [N_tracks, 2]
        safety_distance: Distance threshold for freezing embeddings

    Returns:
        Boolean array where True means track should be frozen
    """
    n_tracks = positions.shape[0]
    freeze_flags = np.zeros(n_tracks, dtype=nb.boolean)

    # Use vectorized distance computation
    for i in range(n_tracks):
        for j in range(i + 1, n_tracks):
            # Compute squared distance to avoid sqrt
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            distance_sq = dx * dx + dy * dy

            if distance_sq < safety_distance * safety_distance:
                freeze_flags[i] = True
                freeze_flags[j] = True
                # Early termination for i once frozen
                break

    return freeze_flags


@nb.njit(fastmath=True, cache=True)
def compute_deduplication_mask(
        positions: np.ndarray,
        confidences: np.ndarray,
        dedup_distance: float
) -> np.ndarray:
    """
    Compute mask for detection deduplication based on proximity.

    Args:
        positions: Detection positions [N_det, 2]
        confidences: Detection confidences [N_det]
        dedup_distance: Distance threshold for deduplication

    Returns:
        Boolean mask where True means keep detection
    """
    n_dets = positions.shape[0]
    keep = np.ones(n_dets, dtype=np.bool_)  # Changed from nb.boolean to np.bool_

    # Sort by confidence (higher confidence first)
    # Numba's argsort works reliably on 1D arrays.
    confidences_1d = confidences.flatten()
    sorted_indices = np.argsort(-confidences_1d)

    for i in range(n_dets):
        idx_i = sorted_indices[i]

        # If this detection has already been suppressed by a higher-confidence one, skip
        if not keep[idx_i]:
            continue

        # Check against all lower confidence detections
        for j in range(i + 1, n_dets):
            idx_j = sorted_indices[j]

            if not keep[idx_j]:
                continue

            # Compute distance
            dx = positions[idx_i, 0] - positions[idx_j, 0]
            dy = positions[idx_i, 1] - positions[idx_j, 1]
            dist = np.sqrt(dx * dx + dy * dy)

            # If too close, suppress the lower-confidence detection
            if dist < dedup_distance:
                keep[idx_j] = False

    return keep