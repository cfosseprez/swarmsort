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

DEFAULT_MAHALANOBIS_NORMALIZATION: float = 5.0
"""Normalization factor for Mahalanobis distance.

Mahalanobis distance is a dimensionless statistical measure (chi-squared distributed).
This factor scales it to pixel-space for comparison with max_distance.

Empirical tuning: With base_variance=25.0 and typical velocities:
- mahal_dist=1.0 (1 std) → 5 pixels penalty
- mahal_dist=2.0 (2 std) → 10 pixels penalty
- mahal_dist=3.0 (3 std) → 15 pixels penalty

NOTE: Must match config.py default (5.0)."""

DEFAULT_PROBABILISTIC_GATING_MULTIPLIER: float = 1.5
"""Multiplier for max_distance in probabilistic gating.
Euclidean pre-filter: max_distance * this value."""

DEFAULT_TIME_COVARIANCE_INFLATION: float = 0.2
"""Rate at which covariance inflates per missed frame.
Covariance *= (1 + this * misses).
NOTE: Must match config.py default (0.2)."""

DEFAULT_BASE_POSITION_VARIANCE: float = 25.0
"""Base position variance for covariance estimation.
NOTE: Must match config.py default (25.0)."""

DEFAULT_VELOCITY_VARIANCE_SCALE: float = 2.0
"""Scale factor for velocity contribution to covariance."""

DEFAULT_VELOCITY_ISOTROPIC_THRESHOLD: float = 0.1
"""Velocity threshold below which covariance is isotropic (circular).
NOTE: Must match config.py default (0.1)."""


# ============================================================================
# COVARIANCE ESTIMATION FUNCTIONS
# ============================================================================

@nb.njit(fastmath=True, cache=True)
def estimate_track_covariances(
    track_velocities: np.ndarray,
    base_variance: float = 25.0,
    velocity_scale: float = 2.0,
    velocity_isotropic_threshold: float = 0.1
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
        base_variance: Base position variance (default: 15.0)
        velocity_scale: Scale for velocity contribution (default: 2.0)
        velocity_isotropic_threshold: Velocity below which covariance is isotropic (default: 0.1)

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

        if vel_magnitude < velocity_isotropic_threshold:
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


@nb.njit(fastmath=True, cache=True, inline='always')
def _compute_dot_product(vec1: np.ndarray, vec2_flat: np.ndarray, start: int, length: int) -> float:
    """Compute dot product between vec1 and a slice of vec2_flat. Inlined for performance."""
    result = 0.0
    for k in range(length):
        result += vec1[k] * vec2_flat[start + k]
    return result


def compute_embedding_distances_matmul(
    det_embeddings: np.ndarray,
    track_embeddings_flat: np.ndarray,
    track_counts: np.ndarray
) -> np.ndarray:
    """
    Fast matrix multiplication path for embedding distance (method=0, last embedding).

    Uses NumPy matmul instead of Numba loops for 2-3x speedup on large matrices.
    Only works for method=0 (last embedding) since it extracts just the last embedding per track.

    Args:
        det_embeddings: Detection embeddings [N_det, emb_dim] (must be L2-normalized)
        track_embeddings_flat: Flattened track embedding histories (L2-normalized)
        track_counts: Number of embeddings per track

    Returns:
        Distance matrix [N_det, N_track] with cosine distances in [0, 1]
    """
    n_dets = det_embeddings.shape[0]
    n_tracks = len(track_counts)
    emb_dim = det_embeddings.shape[1]

    # Extract last embedding per track into a dense matrix
    last_embeddings = np.zeros((n_tracks, emb_dim), dtype=np.float32)

    # Compute track offsets (cumulative sum of counts)
    offset = 0
    for j in range(n_tracks):
        if track_counts[j] > 0:
            # Get last embedding for this track
            last_start = (offset + track_counts[j] - 1) * emb_dim
            last_embeddings[j] = track_embeddings_flat[last_start:last_start + emb_dim]
        # else: leave as zeros (will give distance 1.0 after computation)
        offset += track_counts[j] if track_counts[j] > 0 else 1

    # Matrix multiply: [n_dets, emb_dim] @ [emb_dim, n_tracks] = [n_dets, n_tracks]
    # This computes all cosine similarities at once
    cos_similarities = det_embeddings @ last_embeddings.T

    # Convert to distances: distance = (1 - similarity) * 0.5, range [0, 1]
    distances = (1.0 - cos_similarities) * 0.5

    # Handle tracks with no embeddings (track_counts == 0)
    for j in range(n_tracks):
        if track_counts[j] == 0:
            distances[:, j] = 1.0

    return distances.astype(np.float32)


@nb.njit(fastmath=True, cache=True)
def _numba_median(arr: np.ndarray, n: int) -> float:
    """Compute median of first n elements using insertion sort (efficient for small n)."""
    if n == 0:
        return 1.0
    if n == 1:
        return arr[0]

    # Simple insertion sort for small arrays (max ~15 embeddings typically)
    sorted_arr = np.empty(n, dtype=np.float32)
    for i in range(n):
        sorted_arr[i] = arr[i]

    for i in range(1, n):
        key = sorted_arr[i]
        j = i - 1
        while j >= 0 and sorted_arr[j] > key:
            sorted_arr[j + 1] = sorted_arr[j]
            j -= 1
        sorted_arr[j + 1] = key

    # Return median
    if n % 2 == 1:
        return sorted_arr[n // 2]
    else:
        return (sorted_arr[n // 2 - 1] + sorted_arr[n // 2]) * 0.5


@nb.njit(fastmath=True, cache=True, parallel=True)
def compute_embedding_distances_with_method(
    det_embeddings: np.ndarray,
    track_embeddings_flat: np.ndarray,
    track_counts: np.ndarray,
    method: int  # 0=last, 1=average, 2=weighted_average, 3=best_match, 4=median
) -> np.ndarray:
    """
    Compute embedding distances using various methods for historical embeddings.

    OPTIMIZED: Uses parallel processing over detections and inlined dot product.

    Args:
        det_embeddings: Detection embeddings [N_det, emb_dim] (must be L2-normalized)
        track_embeddings_flat: Flattened track embedding histories (L2-normalized)
        track_counts: Number of embeddings per track
        method: Method for aggregating embedding distances
            0: Use last embedding only (fastest)
            1: Average all distances
            2: Weighted average (recent embeddings have more weight)
            3: Best match (minimum distance)
            4: Median distance (robust to outliers)

    Returns:
        Distance matrix [N_det, N_track] with cosine distances in [0, 1]
    """
    n_dets = det_embeddings.shape[0]
    n_tracks = len(track_counts)
    emb_dim = det_embeddings.shape[1]

    distances = np.zeros((n_dets, n_tracks), dtype=np.float32)

    # Compute track offsets for flat array (cumulative sum)
    track_offsets = np.zeros(n_tracks + 1, dtype=np.int32)
    for t in range(n_tracks):
        track_offsets[t + 1] = track_offsets[t] + track_counts[t]

    # Parallel over detections (independent computations)
    for i in nb.prange(n_dets):
        det_emb = det_embeddings[i]

        for j in range(n_tracks):
            if track_counts[j] == 0:
                distances[i, j] = 1.0
                continue

            start_offset = track_offsets[j] * emb_dim
            n_track_embs = track_counts[j]

            if method == 0:  # last - use only the last embedding
                last_start = start_offset + (n_track_embs - 1) * emb_dim
                dot_product = _compute_dot_product(det_emb, track_embeddings_flat, last_start, emb_dim)
                distances[i, j] = (1.0 - dot_product) * 0.5

            elif method == 1:  # average - mean distance to all embeddings
                total_dist = 0.0
                for emb_idx in range(n_track_embs):
                    emb_start = start_offset + emb_idx * emb_dim
                    dot_product = _compute_dot_product(det_emb, track_embeddings_flat, emb_start, emb_dim)
                    total_dist += (1.0 - dot_product) * 0.5
                distances[i, j] = total_dist / n_track_embs

            elif method == 2:  # weighted_average - recent embeddings weighted higher
                total_dist = 0.0
                total_weight = 0.0
                for emb_idx in range(n_track_embs):
                    weight = float(emb_idx + 1)
                    emb_start = start_offset + emb_idx * emb_dim
                    dot_product = _compute_dot_product(det_emb, track_embeddings_flat, emb_start, emb_dim)
                    total_dist += weight * (1.0 - dot_product) * 0.5
                    total_weight += weight
                distances[i, j] = total_dist / total_weight if total_weight > 0.0 else 1.0

            elif method == 3:  # best_match - minimum distance
                min_dist = 1.0
                for emb_idx in range(n_track_embs):
                    emb_start = start_offset + emb_idx * emb_dim
                    dot_product = _compute_dot_product(det_emb, track_embeddings_flat, emb_start, emb_dim)
                    dist = (1.0 - dot_product) * 0.5
                    if dist < min_dist:
                        min_dist = dist
                distances[i, j] = min_dist

            else:  # method == 4: median - robust to outliers
                # Collect all distances for this track
                dist_buffer = np.empty(n_track_embs, dtype=np.float32)
                for emb_idx in range(n_track_embs):
                    emb_start = start_offset + emb_idx * emb_dim
                    dot_product = _compute_dot_product(det_emb, track_embeddings_flat, emb_start, emb_dim)
                    dist_buffer[emb_idx] = (1.0 - dot_product) * 0.5
                distances[i, j] = _numba_median(dist_buffer, n_track_embs)

    return distances


# ============================================================================
# COST MATRIX COMPUTATION FUNCTIONS
# ============================================================================

@nb.njit(fastmath=True, cache=True, parallel=True)
def compute_cost_matrix_parallel(
    det_positions: np.ndarray,
    track_predicted_positions: np.ndarray,
    track_last_positions: np.ndarray,
    track_misses: np.ndarray,
    scaled_embedding_distances: np.ndarray,
    embedding_weight: float,
    max_distance: float,
    do_embeddings: bool,
    miss_threshold: int = 3,
    embedding_threshold_adjustment: float = 1.0,
    track_uncertainty_ratios: np.ndarray = np.empty(0, dtype=np.float32),
    uncertainty_weight: float = 0.0
) -> np.ndarray:
    """
    Parallel version of cost matrix computation for maximum performance.

    WARNING: May produce slightly non-deterministic results due to floating-point
    accumulation order. Use compute_cost_matrix_vectorized for deterministic behavior.

    See compute_cost_matrix_vectorized for detailed documentation.
    """
    n_dets = det_positions.shape[0]
    n_tracks = track_predicted_positions.shape[0]

    cost_matrix = np.full((n_dets, n_tracks), np.inf, dtype=np.float32)
    max_distance_sq = max_distance * max_distance

    # Effective max distance for assignment gating (accounts for embedding contribution)
    effective_max = max_distance * (1.0 + embedding_weight * embedding_threshold_adjustment) if do_embeddings else max_distance

    # Check if uncertainty penalty is enabled (avoid overhead when disabled)
    # Note: check weight only, array is always passed (empty if disabled)
    use_uncertainty = uncertainty_weight > 0.0

    # Parallel over detections (independent computations)
    for i in nb.prange(n_dets):
        for j in range(n_tracks):
            dx_pred = det_positions[i, 0] - track_predicted_positions[j, 0]
            dy_pred = det_positions[i, 1] - track_predicted_positions[j, 1]
            dist_sq_pred = dx_pred * dx_pred + dy_pred * dy_pred

            dx_last = det_positions[i, 0] - track_last_positions[j, 0]
            dy_last = det_positions[i, 1] - track_last_positions[j, 1]
            dist_sq_last = dx_last * dx_last + dy_last * dy_last

            pos_distance_sq = min(dist_sq_pred, dist_sq_last)

            # Position gating: skip if position alone exceeds max_distance
            if pos_distance_sq > max_distance_sq:
                continue

            pos_distance = np.sqrt(pos_distance_sq)

            if do_embeddings:
                # Additive cost: position is primary, embedding adds penalty
                # C_i,j = pos_distance + w_e × emb_distance × max_distance
                emb_dist = scaled_embedding_distances[i, j]
                cost = pos_distance + embedding_weight * emb_dist * max_distance

                # Apply uncertainty penalty if enabled
                if use_uncertainty:
                    cost = cost * (1.0 + uncertainty_weight * track_uncertainty_ratios[j])

                # Assignment gating: only accept if total cost within effective max
                if cost <= effective_max:
                    cost_matrix[i, j] = cost
            else:
                cost = pos_distance

                # Apply uncertainty penalty if enabled
                if use_uncertainty:
                    cost = cost * (1.0 + uncertainty_weight * track_uncertainty_ratios[j])

                cost_matrix[i, j] = cost

    return cost_matrix


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
    miss_threshold: int = 3,
    embedding_threshold_adjustment: float = 1.0,
    track_uncertainty_ratios: np.ndarray = np.empty(0, dtype=np.float32),
    uncertainty_weight: float = 0.0
) -> np.ndarray:
    """
    Compute optimized cost matrix with position-centric additive embedding costs.

    ALWAYS uses MINIMUM of predicted and last position for matching.
    This handles all cases:
    1. Object moving predictably -> predicted position is closer
    2. Object stopped or changed direction -> last position is closer

    Cost Formula (Position-Centric Additive):
        C_i,j = pos_distance + w_e × emb_distance × max_distance

    With uncertainty penalty (when uncertainty_weight > 0):
        C_i,j = base_cost × (1 + uncertainty_weight × miss_ratio_j)

    Position is always the primary cost. Embedding adds an additional penalty
    that increases the matching cost but never replaces position.

    Args:
        det_positions: Detection positions [N_det, 2]
        track_predicted_positions: Predicted track positions [N_track, 2]
        track_last_positions: Last observed track positions [N_track, 2]
        track_misses: Number of misses for each track (unused, kept for API compatibility)
        scaled_embedding_distances: Pre-computed embedding distances [0, 1]
        embedding_weight: Weight for embedding penalty term (0+)
        max_distance: Maximum position distance for gating
        do_embeddings: Whether to include embedding costs
        miss_threshold: Unused (kept for API compatibility)
        embedding_threshold_adjustment: Threshold adjustment factor for gating
        track_uncertainty_ratios: Recent miss ratios per track [N_track] (optional)
        uncertainty_weight: Weight for uncertainty penalty (0 = disabled, no overhead)

    Returns:
        Cost matrix [N_det, N_track]
    """
    n_dets = det_positions.shape[0]
    n_tracks = track_predicted_positions.shape[0]

    cost_matrix = np.full((n_dets, n_tracks), np.inf, dtype=np.float32)

    # Precompute squared max_distance for gating optimization
    max_distance_sq = max_distance * max_distance

    # Effective max distance for assignment gating (accounts for embedding contribution)
    effective_max = max_distance * (1.0 + embedding_weight * embedding_threshold_adjustment) if do_embeddings else max_distance

    # Check if uncertainty penalty is enabled (avoid overhead when disabled)
    # Note: check weight only, array is always passed (empty if disabled)
    use_uncertainty = uncertainty_weight > 0.0

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

            # Position gating: skip if position alone exceeds max_distance
            if pos_distance_sq > max_distance_sq:
                continue

            # Only compute sqrt when we know it's within range
            pos_distance = np.sqrt(pos_distance_sq)

            if do_embeddings:
                # Position-centric additive cost formula:
                # C_i,j = pos_distance + w_e × emb_distance × max_distance
                #
                # Position is always the primary cost component.
                # Embedding adds a penalty scaled by max_distance to be comparable.
                # This ensures position is never ignored, and embedding only increases cost.
                emb_dist = scaled_embedding_distances[i, j]
                cost = pos_distance + embedding_weight * emb_dist * max_distance

                # Apply uncertainty penalty if enabled
                if use_uncertainty:
                    cost = cost * (1.0 + uncertainty_weight * track_uncertainty_ratios[j])

                # Assignment gating: only accept if total cost within effective max
                if cost <= effective_max:
                    cost_matrix[i, j] = cost
            else:
                cost = pos_distance

                # Apply uncertainty penalty if enabled
                if use_uncertainty:
                    cost = cost * (1.0 + uncertainty_weight * track_uncertainty_ratios[j])

                cost_matrix[i, j] = cost

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
    mahal_normalization: float = 5.0,
    cov_inflation_rate: float = 0.1,
    embedding_threshold_adjustment: float = 1.0,
    singular_cov_threshold: float = 1e-6,
    track_uncertainty_ratios: np.ndarray = np.empty(0, dtype=np.float32),
    uncertainty_weight: float = 0.0
) -> np.ndarray:
    """
    Compute probabilistic cost matrix using Mahalanobis distance with additive embedding cost.

    ALWAYS uses MINIMUM of predicted and last position for matching.
    This handles all cases:
    1. Object moving predictably -> predicted position is closer
    2. Object stopped or changed direction -> last position is closer

    Cost Formula (Position-Centric Additive):
        C_i,j = normalized_mahal + w_e × emb_distance × max_distance

    With uncertainty penalty (when uncertainty_weight > 0):
        C_i,j = base_cost × (1 + uncertainty_weight × miss_ratio_j)

    Args:
        det_positions: Detection positions [N_det, 2]
        track_predicted_positions: Predicted track positions [N_track, 2]
        track_last_positions: Last observed track positions [N_track, 2]
        track_misses: Number of misses for each track
        track_covariances: Track covariance matrices [N_track, 2, 2]
        scaled_embedding_distances: Pre-computed embedding distances [0, 1]
        embedding_weight: Weight for embedding penalty term
        max_distance: Maximum allowed distance
        do_embeddings: Whether to include embedding costs
        miss_threshold: Unused (kept for API compatibility)
        gating_multiplier: Multiplier for max_distance in Euclidean pre-filter
        mahal_normalization: Normalization factor for Mahalanobis distance (default: 5.0)
        cov_inflation_rate: Rate of covariance inflation per missed frame
        embedding_threshold_adjustment: Threshold adjustment factor for gating
        singular_cov_threshold: Threshold for detecting singular covariance (default: 1e-6)
        track_uncertainty_ratios: Recent miss ratios per track [N_track] (optional)
        uncertainty_weight: Weight for uncertainty penalty (0 = disabled, no overhead)

    Returns:
        Cost matrix [N_det, N_track]
    """
    n_dets = det_positions.shape[0]
    n_tracks = track_predicted_positions.shape[0]

    cost_matrix = np.full((n_dets, n_tracks), np.inf, dtype=np.float32)

    # Precompute gating threshold
    gating_threshold = max_distance * gating_multiplier
    gating_threshold_sq = gating_threshold * gating_threshold

    # Effective max distance for assignment gating (accounts for embedding contribution)
    effective_max = max_distance * (1.0 + embedding_weight * embedding_threshold_adjustment) if do_embeddings else max_distance

    # Check if uncertainty penalty is enabled (avoid overhead when disabled)
    # Note: check weight only, array is always passed (empty if disabled)
    use_uncertainty = uncertainty_weight > 0.0

    for i in range(n_dets):
        for j in range(n_tracks):
            # Get covariance for this track
            cov = track_covariances[j]

            # Time-dependent covariance inflation
            time_factor = 1.0 + cov_inflation_rate * track_misses[j]
            inflated_cov = cov * time_factor

            # Compute inverse covariance
            det_cov = inflated_cov[0, 0] * inflated_cov[1, 1] - inflated_cov[0, 1] * inflated_cov[1, 0]
            use_euclidean = abs(det_cov) < singular_cov_threshold

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

            # Position gating: skip if normalized Mahalanobis exceeds max_distance
            if normalized_mahal > max_distance:
                continue

            if do_embeddings:
                # Position-centric additive cost formula:
                # C_i,j = normalized_mahal + w_e × emb_distance × max_distance
                emb_dist = scaled_embedding_distances[i, j]
                cost = normalized_mahal + embedding_weight * emb_dist * max_distance

                # Apply uncertainty penalty if enabled
                if use_uncertainty:
                    cost = cost * (1.0 + uncertainty_weight * track_uncertainty_ratios[j])

                # Assignment gating: only accept if total cost within effective max
                if cost <= effective_max:
                    cost_matrix[i, j] = cost
            else:
                cost = normalized_mahal

                # Apply uncertainty penalty if enabled
                if use_uncertainty:
                    cost = cost * (1.0 + uncertainty_weight * track_uncertainty_ratios[j])

                cost_matrix[i, j] = cost

    return cost_matrix


# ============================================================================
# COLLISION AND FREEZE DETECTION
# ============================================================================

@nb.njit(fastmath=True, cache=True)
def compute_neighbor_counts_vectorized(
    positions: np.ndarray,
    radius: float
) -> np.ndarray:
    """
    Count the number of neighbors within radius for each track.

    Used to implement embedding_freeze_density - tracks freeze when they have
    too many neighbors, indicating a crowded/collision scenario.

    Args:
        positions: Track positions [N_tracks, 2]
        radius: Distance threshold for counting neighbors

    Returns:
        Integer array with neighbor count for each track (excludes self)
    """
    n_tracks = positions.shape[0]
    neighbor_counts = np.zeros(n_tracks, dtype=np.int32)
    radius_sq = radius * radius

    for i in range(n_tracks):
        for j in range(i + 1, n_tracks):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            distance_sq = dx * dx + dy * dy

            if distance_sq < radius_sq:
                neighbor_counts[i] += 1
                neighbor_counts[j] += 1

    return neighbor_counts


@nb.njit(fastmath=True, cache=True)
def compute_freeze_flags_vectorized(
    positions: np.ndarray,
    safety_distance: float
) -> np.ndarray:
    """
    Compute freeze flags for tracks based on proximity (collision detection).

    DEPRECATED: Use compute_neighbor_counts_vectorized() with embedding_freeze_density
    for proper density-based freezing with hysteresis.

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
def _compute_pairwise_dist_sq(positions: np.ndarray) -> np.ndarray:
    """
    Compute pairwise squared distances between all positions.

    Returns upper triangular matrix (i < j only) to save memory.
    Uses vectorized operations within Numba for efficiency.
    """
    n = positions.shape[0]
    # Only store upper triangle (i < j) as flat array
    # Number of pairs = n*(n-1)/2
    n_pairs = (n * (n - 1)) // 2
    dist_sq = np.empty(n_pairs, dtype=np.float32)

    idx = 0
    for i in range(n):
        xi, yi = positions[i, 0], positions[i, 1]
        for j in range(i + 1, n):
            dx = xi - positions[j, 0]
            dy = yi - positions[j, 1]
            dist_sq[idx] = dx * dx + dy * dy
            idx += 1

    return dist_sq


@nb.njit(fastmath=True, cache=True, inline='always')
def _get_pair_index(i: int, j: int, n: int) -> int:
    """Get flat index for pair (i, j) where i < j in upper triangular storage."""
    # Formula: sum of (n-1) + (n-2) + ... + (n-i) + (j - i - 1)
    # = i*n - i*(i+1)/2 + (j - i - 1)
    return i * n - (i * (i + 1)) // 2 + j - i - 1


@nb.njit(fastmath=True, cache=True)
def compute_deduplication_mask(
        positions: np.ndarray,
        confidences: np.ndarray,
        dedup_distance: float
) -> np.ndarray:
    """
    Compute mask for detection deduplication based on proximity (NMS-style).

    OPTIMIZED: Pre-computes pairwise distances for O(1) lookups during NMS.

    Args:
        positions: Detection positions [N_det, 2]
        confidences: Detection confidences [N_det]
        dedup_distance: Distance threshold for deduplication

    Returns:
        Boolean mask where True means keep detection
    """
    n_dets = positions.shape[0]

    if n_dets <= 1:
        return np.ones(n_dets, dtype=nb.boolean)

    keep = np.ones(n_dets, dtype=nb.boolean)
    dedup_distance_sq = dedup_distance * dedup_distance

    # Sort by confidence (higher confidence first)
    confidences_1d = confidences.flatten()
    sorted_indices = np.argsort(-confidences_1d)

    # Pre-compute all pairwise squared distances (O(n²) but fast vectorized)
    dist_sq_flat = _compute_pairwise_dist_sq(positions)

    for i in range(n_dets):
        idx_i = sorted_indices[i]

        if not keep[idx_i]:
            continue

        # Check against all lower confidence detections
        for j in range(i + 1, n_dets):
            idx_j = sorted_indices[j]

            if not keep[idx_j]:
                continue

            # Get pre-computed distance (ensure i < j for lookup)
            if idx_i < idx_j:
                pair_idx = _get_pair_index(idx_i, idx_j, n_dets)
            else:
                pair_idx = _get_pair_index(idx_j, idx_i, n_dets)

            if dist_sq_flat[pair_idx] < dedup_distance_sq:
                keep[idx_j] = False

    return keep