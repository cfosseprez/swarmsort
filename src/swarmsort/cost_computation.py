"""
SwarmSort Cost Computation Module

This module contains functions for computing cost matrices and distance metrics
used in the tracking assignment problem. It includes standard and probabilistic
cost computation strategies with embedding-based matching.

Functions:
    cosine_similarity_normalized: Normalized cosine similarity for embeddings
    compute_embedding_distances_with_method: Multi-method embedding distance computation
    compute_cost_matrix_vectorized: Optimized cost matrix computation with squared-distance gating
    compute_probabilistic_cost_matrix_vectorized: Probabilistic cost with Mahalanobis distance
    compute_freeze_flags_vectorized: Collision detection for embedding freezing
    compute_deduplication_mask: Detection deduplication based on proximity
"""

# ============================================================================
# STANDARD IMPORTS
# ============================================================================
import numpy as np
import numba as nb
from typing import Optional, Tuple
import math

# ============================================================================
# NAMED CONSTANTS
# These constants are used throughout the cost computation module.
# For production use, prefer passing values from SwarmSortConfig where applicable.
# ============================================================================
SINGULAR_COV_THRESHOLD: float = 1e-6
"""Threshold for detecting singular covariance matrices.
If determinant < this value, fall back to Euclidean distance."""

COSINE_DISTANCE_SCALE: float = 2.0
"""Scale factor for cosine distance: distance = (1.0 - dot_product) / SCALE.
Results in distance range [0, 1] for normalized embeddings."""

DEFAULT_PREDICTION_MISS_THRESHOLD: int = 3
"""Number of misses before using last position instead of prediction.
Tracks with >= this many misses use last known position for matching."""

DEFAULT_MAHALANOBIS_NORMALIZATION: float = 20.0
"""Normalization factor for Mahalanobis distance.
Scales Mahalanobis to be comparable with max_distance."""

DEFAULT_PROBABILISTIC_GATING_MULTIPLIER: float = 1.5
"""Multiplier for max_distance in probabilistic gating.
Euclidean pre-filter: max_distance * this value."""

DEFAULT_TIME_COVARIANCE_INFLATION: float = 0.1
"""Rate at which covariance inflates per missed frame.
Covariance *= (1 + this * misses)."""

DEFAULT_BASE_POSITION_VARIANCE: float = 5.0
"""Base position variance for covariance estimation."""

DEFAULT_VELOCITY_VARIANCE_SCALE: float = 2.0
"""Scale factor for velocity contribution to covariance."""


# ============================================================================
# COVARIANCE ESTIMATION FUNCTIONS
# ============================================================================

@nb.njit(fastmath=True, cache=True)
def estimate_track_covariances(
    track_velocities: np.ndarray,
    base_variance: float = 5.0,
    velocity_scale: float = 2.0
) -> np.ndarray:
    """
    Estimate base track covariance matrices from velocity.

    This function computes base uncertainty for each track based on:
    - Base position variance (isotropic component)
    - Velocity magnitude (higher velocity = more uncertainty)
    - Velocity direction (more uncertainty in direction of motion)

    Note: Time-dependent inflation (based on misses) is applied separately
    in compute_probabilistic_cost_matrix_vectorized to avoid double inflation.

    Args:
        track_velocities: Track velocities [N_track, 2] as [vx, vy]
        base_variance: Base position variance (default: 5.0)
        velocity_scale: Scale for velocity contribution (default: 2.0)

    Returns:
        Covariance matrices [N_track, 2, 2]
    """
    n_tracks = track_velocities.shape[0]
    covariances = np.zeros((n_tracks, 2, 2), dtype=np.float32)

    for i in range(n_tracks):
        vx = track_velocities[i, 0]
        vy = track_velocities[i, 1]

        # Velocity magnitude
        vel_magnitude = np.sqrt(vx * vx + vy * vy)

        if vel_magnitude < 0.1:
            # Low velocity: use isotropic covariance
            covariances[i, 0, 0] = base_variance
            covariances[i, 1, 1] = base_variance
            covariances[i, 0, 1] = 0.0
            covariances[i, 1, 0] = 0.0
        else:
            # High velocity: anisotropic covariance (more uncertainty in motion direction)
            # Normalize velocity to get direction
            nx = vx / vel_magnitude
            ny = vy / vel_magnitude

            # Variance along velocity direction (higher)
            var_along = base_variance + velocity_scale * vel_magnitude
            # Variance perpendicular (lower)
            var_perp = base_variance

            # Build covariance: C = R * D * R^T where D is diagonal variances
            # R = [[nx, -ny], [ny, nx]] is rotation matrix
            # This gives: C = var_along * v*v^T + var_perp * (I - v*v^T)
            covariances[i, 0, 0] = var_along * nx * nx + var_perp * ny * ny
            covariances[i, 1, 1] = var_along * ny * ny + var_perp * nx * nx
            covariances[i, 0, 1] = (var_along - var_perp) * nx * ny
            covariances[i, 1, 0] = covariances[i, 0, 1]

    return covariances


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
    do_embeddings: bool,
    miss_threshold: int = 3
) -> np.ndarray:
    """
    Compute optimized cost matrix with position and optional embedding costs.

    ALWAYS uses MINIMUM of predicted and last position for matching.
    This handles all cases:
    1. Object moving predictably -> predicted position is closer
    2. Object stopped or changed direction -> last position is closer

    Args:
        det_positions: Detection positions [N_det, 2]
        track_predicted_positions: Predicted track positions [N_track, 2]
        track_last_positions: Last observed track positions [N_track, 2]
        track_misses: Number of misses for each track (unused, kept for API compatibility)
        scaled_embedding_distances: Pre-computed embedding distances
        embedding_weight: Weight for embedding term (0-1)
        max_distance: Maximum allowed distance
        do_embeddings: Whether to include embedding costs
        miss_threshold: Unused (kept for API compatibility)

    Returns:
        Cost matrix [N_det, N_track]
    """
    n_dets = det_positions.shape[0]
    n_tracks = track_predicted_positions.shape[0]

    cost_matrix = np.full((n_dets, n_tracks), np.inf, dtype=np.float32)

    # Precompute squared max_distance for gating optimization
    max_distance_sq = max_distance * max_distance

    for i in range(n_dets):
        for j in range(n_tracks):
            # ALWAYS use MINIMUM of predicted and last position
            # This handles all cases:
            # 1. Object moving predictably → predicted position is closer
            # 2. Object stopped or changed direction → last position is closer
            # Using MIN ensures the best match is found regardless of motion pattern
            dx_pred = det_positions[i, 0] - track_predicted_positions[j, 0]
            dy_pred = det_positions[i, 1] - track_predicted_positions[j, 1]
            dist_sq_pred = dx_pred * dx_pred + dy_pred * dy_pred

            dx_last = det_positions[i, 0] - track_last_positions[j, 0]
            dy_last = det_positions[i, 1] - track_last_positions[j, 1]
            dist_sq_last = dx_last * dx_last + dy_last * dy_last

            # Take the minimum distance
            pos_distance_sq = min(dist_sq_pred, dist_sq_last)

            # Gating: skip if beyond max_distance
            if pos_distance_sq > max_distance_sq:
                continue

            # Only compute sqrt when we know it's within range
            pos_distance = np.sqrt(pos_distance_sq)

            if do_embeddings:
                # Combined cost matching paper formula:
                # C_i,j = min(d_k, d_l) × (1 - w_e) + w_e × d_s × d_max
                #
                # Position gating already done above (pos_distance <= max_distance)
                # The embedding term is scaled by d_max to be in the same range as position
                # This ensures w_e intuitively controls the balance between motion and appearance
                emb_dist = scaled_embedding_distances[i, j]
                cost_matrix[i, j] = (
                    (1.0 - embedding_weight) * pos_distance +
                    embedding_weight * emb_dist * max_distance
                )
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
    do_embeddings: bool,
    miss_threshold: int = 3,
    gating_multiplier: float = 1.5,
    mahal_normalization: float = 20.0,
    cov_inflation_rate: float = 0.1
) -> np.ndarray:
    """
    Compute probabilistic cost matrix using Mahalanobis distance.

    ALWAYS uses MINIMUM of predicted and last position for matching.
    This handles all cases:
    1. Object moving predictably -> predicted position is closer
    2. Object stopped or changed direction -> last position is closer

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
        miss_threshold: Unused (kept for API compatibility)
        gating_multiplier: Multiplier for max_distance in Euclidean pre-filter
        mahal_normalization: Normalization factor for Mahalanobis distance
        cov_inflation_rate: Rate of covariance inflation per missed frame

    Returns:
        Cost matrix [N_det, N_track]
    """
    n_dets = det_positions.shape[0]
    n_tracks = track_predicted_positions.shape[0]

    cost_matrix = np.full((n_dets, n_tracks), np.inf, dtype=np.float32)

    # Precompute gating threshold
    gating_threshold = max_distance * gating_multiplier
    gating_threshold_sq = gating_threshold * gating_threshold

    for i in range(n_dets):
        for j in range(n_tracks):
            # Get covariance for this track
            cov = track_covariances[j]

            # Time-dependent covariance inflation
            time_factor = 1.0 + cov_inflation_rate * track_misses[j]
            inflated_cov = cov * time_factor

            # Compute inverse covariance
            det_cov = inflated_cov[0, 0] * inflated_cov[1, 1] - inflated_cov[0, 1] * inflated_cov[1, 0]
            use_euclidean = abs(det_cov) < SINGULAR_COV_THRESHOLD

            if not use_euclidean:
                cov_inv = np.zeros((2, 2), dtype=np.float32)
                cov_inv[0, 0] = inflated_cov[1, 1] / det_cov
                cov_inv[0, 1] = -inflated_cov[0, 1] / det_cov
                cov_inv[1, 0] = -inflated_cov[1, 0] / det_cov
                cov_inv[1, 1] = inflated_cov[0, 0] / det_cov

            # ALWAYS use MINIMUM of predicted and last position
            # This handles all cases:
            # 1. Object moving predictably → predicted position is closer
            # 2. Object stopped or changed direction → last position is closer
            diff_pred = det_positions[i] - track_predicted_positions[j]
            diff_last = det_positions[i] - track_last_positions[j]

            # Euclidean distances for gating
            dist_sq_pred = diff_pred[0] * diff_pred[0] + diff_pred[1] * diff_pred[1]
            dist_sq_last = diff_last[0] * diff_last[0] + diff_last[1] * diff_last[1]

            # Check if at least one is within gating threshold
            min_dist_sq = min(dist_sq_pred, dist_sq_last)
            if min_dist_sq > gating_threshold_sq:
                continue

            # Compute Mahalanobis distances for both positions
            if use_euclidean:
                mahal_pred = np.sqrt(dist_sq_pred)
                mahal_last = np.sqrt(dist_sq_last)
            else:
                mahal_pred = fast_mahalanobis_distance(diff_pred, cov_inv)
                mahal_last = fast_mahalanobis_distance(diff_last, cov_inv)

            # Use minimum Mahalanobis distance
            mahal_dist = min(mahal_pred, mahal_last)

            # Normalize Mahalanobis to be comparable with max_distance
            normalized_mahal = mahal_dist * mahal_normalization

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
    keep = np.ones(n_dets, dtype=nb.boolean)

    # Precompute squared threshold to avoid sqrt in loop
    dedup_distance_sq = dedup_distance * dedup_distance

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

            # Compute squared distance (avoid sqrt for performance)
            dx = positions[idx_i, 0] - positions[idx_j, 0]
            dy = positions[idx_i, 1] - positions[idx_j, 1]
            dist_sq = dx * dx + dy * dy

            # If too close, suppress the lower-confidence detection
            if dist_sq < dedup_distance_sq:
                keep[idx_j] = False

    return keep