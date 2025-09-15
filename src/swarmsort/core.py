"""
SwarmSort Core Implementation

This module contains the core SwarmSort multi-object tracking algorithm implementation.
SwarmSort combines Kalman filtering, Hungarian algorithm assignment, and deep learning
embeddings for robust real-time object tracking.

Key Features:
- Real-time multi-object tracking with motion prediction
- Embedding-based track association for improved accuracy
- Re-identification (ReID) capabilities for recovering lost tracks
- Probabilistic and non-probabilistic cost computation methods
- Novel 3D assignment across multiple prediction methods
- Numba-accelerated functions for high performance
- Configurable parameters for different tracking scenarios

Classes:
    SwarmSortTracker: Main tracking class implementing the full algorithm
    FastTrackState: Fast dataclass for tracking individual objects
    PendingDetection: Temporary storage for unconfirmed detections

Functions:
    Various Numba-accelerated utility functions for distance computation,
    cost matrix calculation, and embedding operations.
"""
# ============================================================================
# STANDARD IMPORTS
# ============================================================================
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Type, TypeVar, Union, Literal
from dataclasses import dataclass, field
from collections import deque
import numba as nb
from scipy.optimize import linear_sum_assignment
import os
from pathlib import Path
import time
import sys
from dataclasses import dataclass, asdict, field, fields
import gc

# ============================================================================
# LOGGER
# ============================================================================
from loguru import logger

# ============================================================================
# Internal imports
# ============================================================================
from .data_classes import Detection, TrackedObject
from .config import SwarmSortConfig
from .embedding_scaler import EmbeddingDistanceScaler


# ============================================================================
# PERFORMANCE TIMING UTILITIES
# ============================================================================
class Timer:
    """Simple high-resolution timer for performance profiling.

    Used internally by SwarmSort to measure execution times of different
    algorithm components for debugging and optimization purposes.

    Attributes:
        _start_times (dict): Dictionary storing start times for active timers

    Example:
        >>> timer = Timer()
        >>> store = {}
        >>> timer.start('detection')
        >>> # ... do some work ...
        >>> timer.stop('detection', store)  # Accumulates time in store['detection']
    """

    def __init__(self):
        """Initialize an empty timer with no active measurements."""
        self._start_times = {}

    def start(self, key: str) -> None:
        """Start timing for the given key.

        Args:
            key: Unique identifier for this timing measurement
        """
        self._start_times[key] = time.perf_counter()

    def stop(self, key: str, store: dict) -> None:
        """Stop timing for the given key and accumulate the duration.

        Args:
            key: The timing measurement identifier
            store: Dictionary to accumulate timing results in
        """
        if key in self._start_times:
            duration = time.perf_counter() - self._start_times[key]
            store[key] = store.get(key, 0.0) + duration


# ============================================================================
# INITIALIZATION SYSTEM
# ============================================================================
@dataclass
class PendingDetection:
    """Represents a detection waiting to become a track."""

    position: np.ndarray
    embedding: Optional[np.ndarray] = None
    bbox: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.float32)
    )  # [x1, y1, x2, y2]
    class_id: Optional[int] = None
    confidence: float = 1.0
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    consecutive_frames: int = 1
    total_detections: int = 1
    average_position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))

    def __post_init__(self):
        if self.average_position.sum() == 0:
            self.average_position = self.position.copy()


# ============================================================================
# NUMBA FUNCTIONS
# ============================================================================


@nb.njit(fastmath=True, cache=True)
def cosine_similarity_normalized(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Fast cosine similarity normalized to [0, 1] - ORIGINAL"""
    norm1 = np.sqrt(np.sum(emb1 * emb1))
    norm2 = np.sqrt(np.sum(emb2 * emb2))

    if norm1 == 0 or norm2 == 0:
        return 1.0

    cos_sim = np.sum(emb1 * emb2) / (norm1 * norm2)
    return (1.0 - cos_sim) / 2.0


@nb.njit(fastmath=True, cache=True)
def fast_mahalanobis_distance(diff: np.ndarray, cov_inv: np.ndarray) -> float:
    """Fast 2D Mahalanobis distance - ORIGINAL"""
    return np.sqrt(
        diff[0] * (cov_inv[0, 0] * diff[0] + cov_inv[0, 1] * diff[1])
        + diff[1] * (cov_inv[1, 0] * diff[0] + cov_inv[1, 1] * diff[1])
    )


@nb.njit(fastmath=True, parallel=False, cache=True)
def compute_embedding_distances_optimized(det_embeddings, track_embeddings):
    """Optimized embedding distance computation"""
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
    Compute embedding distances using specified method.

    Args:
        det_embeddings: (n_dets, emb_dim)
        track_embeddings_flat: Flattened array of all track embeddings
        track_counts: Number of embeddings per track
        method: 0=last, 1=average, 2=weighted_average, 3=best_match
    """
    n_dets = det_embeddings.shape[0]
    n_tracks = len(track_counts)
    emb_dim = det_embeddings.shape[1]

    distances = np.empty((n_dets, n_tracks), dtype=np.float32)

    track_start = 0
    for t in range(n_tracks):
        count = track_counts[t]
        if count == 0:
            # No embeddings - max distance
            for d in range(n_dets):
                distances[d, t] = 1.0
            continue

        track_end = track_start + count
        track_embs = track_embeddings_flat[track_start:track_end]

        for d in range(n_dets):
            det_emb = det_embeddings[d]

            if method == 0:  # last
                # Use only the most recent embedding
                track_emb = track_embs[-1]
                dot_product = 0.0
                for k in range(emb_dim):
                    dot_product += det_emb[k] * track_emb[k]
                distances[d, t] = (1.0 - dot_product) / 2.0

            elif method == 1:  # average
                # Simple average of all embeddings
                avg_emb = np.zeros(emb_dim, dtype=np.float32)
                for i in range(count):
                    for k in range(emb_dim):
                        avg_emb[k] += track_embs[i, k]
                for k in range(emb_dim):
                    avg_emb[k] /= count

                dot_product = 0.0
                for k in range(emb_dim):
                    dot_product += det_emb[k] * avg_emb[k]
                distances[d, t] = (1.0 - dot_product) / 2.0

            elif method == 2:  # weighted_average
                # Exponentially weighted average (recent = higher weight)
                weighted_emb = np.zeros(emb_dim, dtype=np.float32)
                weight_sum = 0.0

                for i in range(count):
                    # Exponential weight: older = less weight
                    weight = np.exp(-2.0 * (count - i - 1) / max(count, 1))
                    weight_sum += weight
                    for k in range(emb_dim):
                        weighted_emb[k] += weight * track_embs[i, k]

                for k in range(emb_dim):
                    weighted_emb[k] /= weight_sum

                dot_product = 0.0
                for k in range(emb_dim):
                    dot_product += det_emb[k] * weighted_emb[k]
                distances[d, t] = (1.0 - dot_product) / 2.0

            else:  # method == 3, best_match
                # Find the historical embedding with minimum distance
                min_dist = 2.0  # Max possible cosine distance
                for i in range(count):
                    track_emb = track_embs[i]
                    dot_product = 0.0
                    for k in range(emb_dim):
                        dot_product += det_emb[k] * track_emb[k]
                    dist = (1.0 - dot_product) / 2.0
                    if dist < min_dist:
                        min_dist = dist
                distances[d, t] = min_dist

        track_start = track_end

    return distances


@nb.njit(fastmath=True, cache=True)
def fast_gaussian_fusion(
        mu_k: np.ndarray, cov_k: np.ndarray, mu_d: np.ndarray, cov_d: np.ndarray
):
    """Fast 2D Gaussian fusion without scipy dependencies"""
    mu_k = mu_k.astype(np.float32)
    cov_k = cov_k.astype(np.float32)
    mu_d = mu_d.astype(np.float32)
    cov_d = cov_d.astype(np.float32)

    reg = np.float32(1e-4)
    cov_k_reg = cov_k + reg * np.eye(2, dtype=np.float32)
    cov_d_reg = cov_d + reg * np.eye(2, dtype=np.float32)

    det_k = cov_k_reg[0, 0] * cov_k_reg[1, 1] - cov_k_reg[0, 1] * cov_k_reg[1, 0]
    det_d = cov_d_reg[0, 0] * cov_d_reg[1, 1] - cov_d_reg[0, 1] * cov_d_reg[1, 0]

    if det_k <= 0 or det_d <= 0:
        fallback_mean = ((mu_k + mu_d) / 2.0).astype(np.float32)
        fallback_cov = (np.eye(2, dtype=np.float32) * 10.0).astype(np.float32)
        return fallback_mean, fallback_cov

    inv_cov_k = (
            np.array(
                [[cov_k_reg[1, 1], -cov_k_reg[0, 1]], [-cov_k_reg[1, 0], cov_k_reg[0, 0]]],
                dtype=np.float32,
            )
            / det_k
    )

    inv_cov_d = (
            np.array(
                [[cov_d_reg[1, 1], -cov_d_reg[0, 1]], [-cov_d_reg[1, 0], cov_d_reg[0, 0]]],
                dtype=np.float32,
            )
            / det_d
    )

    inv_cov_fused = inv_cov_k + inv_cov_d

    det_fused = (
            inv_cov_fused[0, 0] * inv_cov_fused[1, 1] - inv_cov_fused[0, 1] * inv_cov_fused[1, 0]
    )
    if det_fused <= 0:
        fallback_mean = ((mu_k + mu_d) / 2.0).astype(np.float32)
        fallback_cov = (np.eye(2, dtype=np.float32) * 10.0).astype(np.float32)
        return fallback_mean, fallback_cov

    cov_fused = (
            np.array(
                [
                    [inv_cov_fused[1, 1], -inv_cov_fused[0, 1]],
                    [-inv_cov_fused[1, 0], inv_cov_fused[0, 0]],
                ],
                dtype=np.float32,
            )
            / det_fused
    )

    mu_fused = cov_fused @ (inv_cov_k @ mu_k + inv_cov_d @ mu_d)

    return mu_fused.astype(np.float32), cov_fused.astype(np.float32)


@nb.njit(fastmath=True, cache=True)
def select_best_embeddings_numba(track_embeddings_list, det_embeddings, track_lengths):
    """
    Numba-optimized selection of best embeddings for each track
    track_embeddings_list: flattened array of all track embeddings
    track_lengths: array indicating number of embeddings per track
    """
    n_dets = det_embeddings.shape[0]
    n_tracks = len(track_lengths)
    emb_dim = det_embeddings.shape[1]

    result = np.zeros((n_tracks, emb_dim), dtype=np.float32)

    start_idx = 0
    for track_idx in range(n_tracks):
        track_len = track_lengths[track_idx]
        if track_len == 0:
            continue

        if track_len == 1:
            result[track_idx] = track_embeddings_list[start_idx]
        else:
            # Compute similarities for this track's embeddings
            best_avg_sim = -1.0
            best_emb_idx = 0

            for emb_idx in range(track_len):
                current_emb = track_embeddings_list[start_idx + emb_idx]
                avg_sim = 0.0

                # Compute average similarity to all detections
                for det_idx in range(n_dets):
                    det_emb = det_embeddings[det_idx]
                    sim = 0.0
                    for k in range(emb_dim):
                        sim += current_emb[k] * det_emb[k]
                    avg_sim += sim

                avg_sim /= n_dets

                if avg_sim > best_avg_sim:
                    best_avg_sim = avg_sim
                    best_emb_idx = emb_idx

            result[track_idx] = track_embeddings_list[start_idx + best_emb_idx]

        start_idx += track_len

    return result


@nb.njit(fastmath=True, cache=True, parallel=False)
def compute_embedding_distances_multi_history(
        det_embeddings, track_embeddings_list, track_embedding_counts, method="min"
):
    """
    Compute embedding distances considering multiple embeddings per track

    Args:
        det_embeddings: (n_dets, emb_dim) normalized detection embeddings
        track_embeddings_list: flattened array of all track embeddings
        track_embedding_counts: array with number of embeddings per track
        method: 'min' (best match), 'average', or 'weighted_average'

    Returns:
        distances: (n_dets, n_tracks) distance matrix
    """
    n_dets = det_embeddings.shape[0]
    n_tracks = len(track_embedding_counts)
    emb_dim = det_embeddings.shape[1]

    distances = np.empty((n_dets, n_tracks), dtype=np.float32)

    track_start_idx = 0

    for j in range(n_tracks):
        n_embs = track_embedding_counts[j]

        if n_embs == 0:
            # No embeddings for this track - set max distance
            for i in range(n_dets):
                distances[i, j] = 1.0
            track_start_idx += n_embs
            continue

        for i in range(n_dets):
            det_emb = det_embeddings[i]

            if method == "min":  # best_match
                # Find minimum distance across all track embeddings
                min_dist = 1.0

                for k in range(n_embs):
                    track_emb = track_embeddings_list[track_start_idx + k]

                    # Compute cosine similarity
                    dot_product = 0.0
                    for dim in range(emb_dim):
                        dot_product += det_emb[dim] * track_emb[dim]

                    # Convert to distance
                    dist = (1.0 - dot_product) / 2.0

                    if dist < min_dist:
                        min_dist = dist

                distances[i, j] = min_dist

            elif method == "average":
                # Average distance to all embeddings
                avg_dist = 0.0

                for k in range(n_embs):
                    track_emb = track_embeddings_list[track_start_idx + k]

                    dot_product = 0.0
                    for dim in range(emb_dim):
                        dot_product += det_emb[dim] * track_emb[dim]

                    dist = (1.0 - dot_product) / 2.0
                    avg_dist += dist

                distances[i, j] = avg_dist / n_embs

            elif method == "weighted_average":
                # Weighted average (more recent = higher weight)
                weighted_dist = 0.0
                weight_sum = 0.0

                for k in range(n_embs):
                    track_emb = track_embeddings_list[track_start_idx + k]

                    # Weight increases with k (more recent)
                    weight = np.exp((k - n_embs + 1) * 0.5)  # Exponential decay

                    dot_product = 0.0
                    for dim in range(emb_dim):
                        dot_product += det_emb[dim] * track_emb[dim]

                    dist = (1.0 - dot_product) / 2.0
                    weighted_dist += dist * weight
                    weight_sum += weight

                distances[i, j] = weighted_dist / weight_sum

        track_start_idx += n_embs

    return distances


@nb.njit(fastmath=True, parallel=False, cache=True)
def compute_cost_matrix_with_multi_embeddings(
        det_positions: np.ndarray,
        track_last_positions: np.ndarray,
        track_kalman_positions: np.ndarray,
        scaled_embedding_matrix: np.ndarray,
        do_embeddings: bool,
        max_distance: float,
        embedding_weight: float,
) -> np.ndarray:
    """
    Cost matrix computation with multi-embedding support
    Note: embedding distances are already computed with best match
    """
    n_dets = det_positions.shape[0]
    n_tracks = track_last_positions.shape[0]
    cost_matrix = np.full((n_dets, n_tracks), np.inf, dtype=np.float32)

    for i in nb.prange(n_dets):
        for j in range(n_tracks):
            # Spatial distances
            dist_to_last = np.sqrt(
                (det_positions[i, 0] - track_last_positions[j, 0]) ** 2
                + (det_positions[i, 1] - track_last_positions[j, 1]) ** 2
            )

            dist_to_kalman = np.sqrt(
                (det_positions[i, 0] - track_kalman_positions[j, 0]) ** 2
                + (det_positions[i, 1] - track_kalman_positions[j, 1]) ** 2
            )

            spatial_cost = min(dist_to_last, dist_to_kalman)

            if spatial_cost > max_distance:
                continue

            if do_embeddings:
                # The scaled_embedding_matrix already contains the best/avg distance
                scaled_emb_dist = scaled_embedding_matrix[i, j]
                # Scale embedding distance to the same range as spatial distance
                embedding_cost_scaled = scaled_emb_dist * max_distance
                # Weighted average of spatial and embedding costs
                total_cost = spatial_cost + embedding_weight * embedding_cost_scaled

            else:
                total_cost = spatial_cost

            cost_matrix[i, j] = total_cost

    return cost_matrix


@nb.njit(fastmath=True, parallel=False, cache=True)
def compute_probabilistic_cost_matrix_vectorized(
        det_positions: np.ndarray,
        track_positions: np.ndarray,
        track_last_positions: np.ndarray,
        track_frames_since_detection: np.ndarray,  # NEW
        scaled_embedding_matrix: np.ndarray,
        embedding_median: float,
        do_embeddings: bool,
        max_distance: float,
        embedding_weight: float,
) -> np.ndarray:
    """Probabilistic cost matrix with time-dependent covariances"""
    n_dets = det_positions.shape[0]
    n_tracks = track_positions.shape[0]
    cost_matrix = np.full((n_dets, n_tracks), np.inf, dtype=np.float32)

    # Base covariances
    base_kalman_cov = 20.0
    base_detection_cov = 10.0
    reg = np.float32(1e-4)

    for i in nb.prange(n_dets):
        det_pos = det_positions[i]

        for j in range(n_tracks):
            kalman_pos = track_positions[j]
            detection_pos = track_last_positions[j]
            frames_since = track_frames_since_detection[j]  # NEW

            # Quick spatial filter first
            simple_dist = np.sqrt(
                (det_pos[0] - kalman_pos[0]) ** 2 + (det_pos[1] - kalman_pos[1]) ** 2
            )
            if simple_dist > max_distance:
                continue

            # Time-dependent covariances
            kalman_cov_val = base_kalman_cov * (1.0 + 0.1 * frames_since)
            detection_cov_val = base_detection_cov * (1.0 + 0.5 * frames_since)

            kalman_cov_reg = kalman_cov_val + reg
            detection_cov_reg = detection_cov_val + reg

            # Simplified fusion weights
            weight_k = 1.0 / (2.0 * kalman_cov_reg)  # 2D trace
            weight_d = 1.0 / (2.0 * detection_cov_reg)
            total_weight = weight_k + weight_d

            fused_pos_x = (kalman_pos[0] * weight_k + detection_pos[0] * weight_d) / total_weight
            fused_pos_y = (kalman_pos[1] * weight_k + detection_pos[1] * weight_d) / total_weight

            # Distance from fused position
            diff_x = det_pos[0] - fused_pos_x
            diff_y = det_pos[1] - fused_pos_y
            spatial_cost = np.sqrt(diff_x * diff_x + diff_y * diff_y)

            # Combine with embedding cost using a weighted average
            if do_embeddings:
                scaled_emb_dist = scaled_embedding_matrix[i, j]
                # Scale embedding distance to the same range as spatial distance
                embedding_cost_scaled = scaled_emb_dist * max_distance
                # Weighted average of spatial and embedding costs
                total_cost = (
                                     1.0 - embedding_weight
                             ) * spatial_cost + embedding_weight * embedding_cost_scaled
            else:
                total_cost = spatial_cost

            if total_cost <= max_distance:
                cost_matrix[i, j] = total_cost

    return cost_matrix


@nb.njit(fastmath=True, parallel=True, cache=True)
def compute_cost_matrix_vectorized_parallel(
        det_positions: np.ndarray,
        track_last_positions: np.ndarray,
        track_kalman_positions: np.ndarray,
        det_embeddings: np.ndarray,
        track_embeddings: np.ndarray,
        do_embeddings: bool,
        max_distance: float,
        embedding_weight: float,
) -> np.ndarray:
    """Parallel version for large-scale scenarios"""
    n_dets = det_positions.shape[0]
    n_tracks = track_last_positions.shape[0]
    cost_matrix = np.full((n_dets, n_tracks), np.inf, dtype=np.float32)

    # Use parallel processing for outer loop
    for i in nb.prange(n_dets):
        for j in range(n_tracks):
            # Early spatial filtering - check rough distance first
            rough_dist = abs(det_positions[i, 0] - track_kalman_positions[j, 0]) + \
                        abs(det_positions[i, 1] - track_kalman_positions[j, 1])
            
            # Skip if Manhattan distance is too large (faster than Euclidean)
            if rough_dist > max_distance * 2:
                continue
            
            dist_to_last = np.sqrt(
                (det_positions[i, 0] - track_last_positions[j, 0]) ** 2
                + (det_positions[i, 1] - track_last_positions[j, 1]) ** 2
            )

            dist_to_kalman = np.sqrt(
                (det_positions[i, 0] - track_kalman_positions[j, 0]) ** 2
                + (det_positions[i, 1] - track_kalman_positions[j, 1]) ** 2
            )

            spatial_cost = min(dist_to_last, dist_to_kalman)

            if spatial_cost > max_distance:
                continue

            embedding_cost = 0.0
            if do_embeddings:
                emb_dist = cosine_similarity_normalized(det_embeddings[i], track_embeddings[j])
                embedding_cost = emb_dist * embedding_weight * max_distance

            cost_matrix[i, j] = spatial_cost + embedding_cost

    return cost_matrix


@nb.njit(fastmath=True, parallel=False, cache=True)
def compute_cost_matrix_vectorized(
        det_positions: np.ndarray,
        track_last_positions: np.ndarray,
        track_kalman_positions: np.ndarray,
        det_embeddings: np.ndarray,
        track_embeddings: np.ndarray,
        do_embeddings: bool,
        max_distance: float,
        embedding_weight: float,
) -> np.ndarray:
    """Vectorized cost matrix computation with parallel processing"""
    n_dets = det_positions.shape[0]
    n_tracks = track_last_positions.shape[0]
    cost_matrix = np.full((n_dets, n_tracks), np.inf, dtype=np.float32)

    # Use parallel processing for outer loop
    for i in nb.prange(n_dets):
        for j in range(n_tracks):
            # Early spatial filtering - check rough distance first
            rough_dist = abs(det_positions[i, 0] - track_kalman_positions[j, 0]) + \
                        abs(det_positions[i, 1] - track_kalman_positions[j, 1])
            
            # Skip if Manhattan distance is too large (faster than Euclidean)
            if rough_dist > max_distance * 2:
                continue
            
            dist_to_last = np.sqrt(
                (det_positions[i, 0] - track_last_positions[j, 0]) ** 2
                + (det_positions[i, 1] - track_last_positions[j, 1]) ** 2
            )

            dist_to_kalman = np.sqrt(
                (det_positions[i, 0] - track_kalman_positions[j, 0]) ** 2
                + (det_positions[i, 1] - track_kalman_positions[j, 1]) ** 2
            )

            spatial_cost = min(dist_to_last, dist_to_kalman)

            if spatial_cost > max_distance:
                continue

            embedding_cost = 0.0
            if do_embeddings:
                emb_dist = cosine_similarity_normalized(det_embeddings[i], track_embeddings[j])
                embedding_cost = emb_dist * embedding_weight * max_distance

            cost_matrix[i, j] = spatial_cost + embedding_cost

    return cost_matrix


def numba_greedy_assignment(cost_matrix: np.ndarray, max_distance: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    OPTIMIZED greedy assignment using vectorized operations.
    Complexity reduced from O(n³) to O(n²) by using numpy's optimized argmin.
    """
    n_dets, n_tracks = cost_matrix.shape

    # Create a working copy of the cost matrix
    working_matrix = cost_matrix.copy()
    working_matrix[working_matrix > max_distance] = np.inf

    # Track used detections and tracks
    used_dets = np.zeros(n_dets, dtype=bool)
    used_tracks = np.zeros(n_tracks, dtype=bool)

    # Store matches
    matches = []

    # Greedy assignment using vectorized operations
    for _ in range(min(n_dets, n_tracks)):
        # Find global minimum in the working matrix
        min_flat_idx = np.argmin(working_matrix)
        min_cost = working_matrix.flat[min_flat_idx]
        
        if min_cost == np.inf:
            break
            
        # Convert flat index to 2D coordinates
        best_det = min_flat_idx // n_tracks
        best_track = min_flat_idx % n_tracks
        
        # Record the match
        matches.append((best_det, best_track))
        
        # Mark as used by setting entire row and column to infinity
        working_matrix[best_det, :] = np.inf
        working_matrix[:, best_track] = np.inf
        
        used_dets[best_det] = True
        used_tracks[best_track] = True

    # Convert matches to numpy array
    if matches:
        matches_array = np.array(matches, dtype=np.int32)
    else:
        matches_array = np.empty((0, 2), dtype=np.int32)

    # Build unmatched arrays
    unmatched_dets = np.where(~used_dets)[0].astype(np.int32)
    unmatched_tracks = np.where(~used_tracks)[0].astype(np.int32)

    return matches_array, unmatched_dets, unmatched_tracks


@nb.njit(fastmath=True, cache=True)
def compute_assignment_priorities(cost_matrix: np.ndarray, max_distance: float) -> np.ndarray:
    """
    Compute priority scores for greedy assignment (SUPER FAST).
    Lower scores = higher priority (better matches should be assigned first)

    Priority considers:
    1. Cost (distance) - lower is better
    2. Number of alternatives - fewer alternatives = higher priority
    3. Confidence of the match relative to alternatives
    """
    n_dets, n_tracks = cost_matrix.shape
    priorities = np.full((n_dets, n_tracks), np.inf, dtype=np.float32)

    for i in range(n_dets):
        for j in range(n_tracks):
            cost = cost_matrix[i, j]
            if cost > max_distance:
                continue

            # Base priority is the cost itself
            priority = cost

            # Count valid alternatives for this detection
            det_alternatives = 0
            for k in range(n_tracks):
                if cost_matrix[i, k] <= max_distance:
                    det_alternatives += 1

            # Count valid alternatives for this track
            track_alternatives = 0
            for k in range(n_dets):
                if cost_matrix[k, j] <= max_distance:
                    track_alternatives += 1

            # Boost priority (lower score) for matches with fewer alternatives
            if det_alternatives > 1:
                priority *= (1.0 + 0.1 * (det_alternatives - 1))
            if track_alternatives > 1:
                priority *= (1.0 + 0.1 * (track_alternatives - 1))

            # Find second best option for confidence calculation
            second_best_det = np.inf
            second_best_track = np.inf

            for k in range(n_tracks):
                if k != j and cost_matrix[i, k] < second_best_det:
                    second_best_det = cost_matrix[i, k]

            for k in range(n_dets):
                if k != i and cost_matrix[k, j] < second_best_track:
                    second_best_track = cost_matrix[k, j]

            # Confidence boost: if this match is much better than alternatives
            min_second_best = min(second_best_det, second_best_track)
            if min_second_best < np.inf and min_second_best > cost * 1.5:
                priority *= 0.7  # Higher confidence = lower priority score

            priorities[i, j] = priority

    return priorities


@nb.njit(fastmath=True, cache=True)
def simple_kalman_update(x_pred: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Simplified Kalman update with proper type handling"""
    alpha = 0.7
    x_updated = np.zeros(4, dtype=np.float32)
    # Ensure we're getting scalars by using .item() or flatten
    z0 = z[0] if z.ndim == 1 else z.flat[0]
    z1 = z[1] if z.ndim == 1 else z.flat[1]
    x_updated[0] = alpha * z0 + (1 - alpha) * x_pred[0]
    x_updated[1] = alpha * z1 + (1 - alpha) * x_pred[1]
    x_updated[2] = x_pred[2]
    x_updated[3] = x_pred[3]
    return x_updated


@nb.njit(fastmath=True, cache=True)
def simple_kalman_predict(x: np.ndarray) -> np.ndarray:
    """Simplified Kalman prediction"""
    x_pred = np.zeros(4, dtype=np.float32)
    x_pred[0] = x[0] + x[2]
    x_pred[1] = x[1] + x[3]
    x_pred[2] = x[2] * 0.95
    x_pred[3] = x[3] * 0.95
    return x_pred


# ============================================================================
# OC-SORT STYLE FUNCTIONS
# ============================================================================

@nb.njit(fastmath=True, cache=True)
def oc_sort_predict(observation_history: np.ndarray, 
                     observation_frames: np.ndarray,
                     current_frame: int) -> np.ndarray:
    """
    OC-SORT style prediction using observation history.
    Returns [x, y, vx, vy] prediction.
    """
    n_obs = len(observation_frames)
    
    if n_obs == 0:
        # No observations, return zeros
        return np.zeros(4, dtype=np.float32)
    
    if n_obs == 1:
        # Single observation, no velocity
        pred = np.zeros(4, dtype=np.float32)
        pred[0] = observation_history[0, 0]
        pred[1] = observation_history[0, 1]
        return pred
    
    # Use last two observations for velocity estimation
    if n_obs >= 2:
        # Time delta between last two observations
        dt = observation_frames[-1] - observation_frames[-2]
        if dt == 0:
            dt = 1
        
        # Velocity from last two observations
        vx = (observation_history[-1, 0] - observation_history[-2, 0]) / dt
        vy = (observation_history[-1, 1] - observation_history[-2, 1]) / dt
        
        # Time since last observation
        delta_t = current_frame - observation_frames[-1]
        
        # Predict position
        pred = np.zeros(4, dtype=np.float32)
        pred[0] = observation_history[-1, 0] + vx * delta_t
        pred[1] = observation_history[-1, 1] + vy * delta_t
        pred[2] = vx
        pred[3] = vy
        
        return pred
    
    return np.zeros(4, dtype=np.float32)


@nb.njit(fastmath=True, cache=True)
def oc_sort_update(observation_history: np.ndarray,
                    observation_frames: np.ndarray, 
                    new_observation: np.ndarray,
                    current_frame: int,
                    max_history: int = 30) -> tuple:
    """
    OC-SORT style update - just store observations, no filtering.
    Returns updated history and frames arrays.
    """
    # Add new observation
    n_obs = len(observation_frames)
    
    if n_obs < max_history:
        # Append to history
        new_history = np.zeros((n_obs + 1, 2), dtype=np.float32)
        new_frames = np.zeros(n_obs + 1, dtype=np.int32)
        
        if n_obs > 0:
            new_history[:n_obs] = observation_history
            new_frames[:n_obs] = observation_frames
        
        new_history[n_obs] = new_observation
        new_frames[n_obs] = current_frame
    else:
        # Shift history (remove oldest)
        new_history = np.zeros((max_history, 2), dtype=np.float32)
        new_frames = np.zeros(max_history, dtype=np.int32)
        
        new_history[:-1] = observation_history[1:]
        new_frames[:-1] = observation_frames[1:]
        
        new_history[-1] = new_observation
        new_frames[-1] = current_frame
    
    return new_history, new_frames


@nb.njit(fastmath=True, cache=True)
def compute_oc_sort_cost_matrix(
    det_positions: np.ndarray,
    track_last_observed_positions: np.ndarray,  # Not predicted!
    track_velocities: np.ndarray,
    track_misses: np.ndarray,
    max_distance: float,
    velocity_weight: float = 0.2
) -> np.ndarray:
    """OC-SORT style cost computation"""
    n_dets = det_positions.shape[0]
    n_tracks = track_last_observed_positions.shape[0]
    cost_matrix = np.full((n_dets, n_tracks), np.inf, dtype=np.float32)
    
    for i in range(n_dets):
        for j in range(n_tracks):
            # Base distance from LAST OBSERVED position
            dist = np.sqrt(
                (det_positions[i, 0] - track_last_observed_positions[j, 0])**2 +
                (det_positions[i, 1] - track_last_observed_positions[j, 1])**2
            )
            
            # Adaptive threshold based on misses
            adaptive_threshold = max_distance * (1.0 + 0.2 * track_misses[j])
            
            if dist > adaptive_threshold:
                continue
            
            # Velocity consistency (optional, can be disabled for collisions)
            if track_misses[j] == 0:  # Only for recently seen tracks
                # Expected position based on velocity
                expected_x = track_last_observed_positions[j, 0] + track_velocities[j, 0]
                expected_y = track_last_observed_positions[j, 1] + track_velocities[j, 1]
                
                velocity_error = np.sqrt(
                    (det_positions[i, 0] - expected_x)**2 +
                    (det_positions[i, 1] - expected_y)**2
                )
                
                # Add velocity consistency cost
                cost = dist + velocity_weight * velocity_error
            else:
                # For lost tracks, only use position
                cost = dist
            
            cost_matrix[i, j] = cost
    
    return cost_matrix






@nb.njit(fastmath=True, cache=True)
def compute_track_uncertainties_numba(
    track_positions: np.ndarray,
    track_misses: np.ndarray,
    track_ages: np.ndarray,
    local_density_radius: float
) -> np.ndarray:
    """
    ULTRA-FAST: Numba-compiled uncertainty computation.
    """
    n_tracks = track_positions.shape[0]
    uncertainties = np.zeros(n_tracks, dtype=np.float32)
    
    for i in range(n_tracks):
        uncertainty = 0.0
        
        # Time component: 0.1 per missed frame, capped at 0.5
        uncertainty += min(track_misses[i] * 0.1, 0.5)
        
        # Local density component: 0.2 per nearby track, capped at 0.6
        nearby_count = 0
        for j in range(n_tracks):
            if i != j:
                dx = track_positions[i, 0] - track_positions[j, 0]
                dy = track_positions[i, 1] - track_positions[j, 1]
                distance = np.sqrt(dx * dx + dy * dy)
                if distance < local_density_radius:
                    nearby_count += 1
        uncertainty += min(nearby_count * 0.2, 0.6)
        
        # Reliability component: miss rate over track lifetime, capped at 0.3
        if track_ages[i] > 0:
            miss_rate = track_misses[i] / track_ages[i]
            uncertainty += min(miss_rate * 0.3, 0.3)
        
        # Cap total uncertainty at 1.0
        uncertainties[i] = min(uncertainty, 1.0)
    
    return uncertainties


@nb.njit(fastmath=True, cache=True)
def compute_cost_matrix_with_uncertainty(
    det_positions: np.ndarray,
    track_positions: np.ndarray,
    scaled_embedding_matrix: np.ndarray,
    track_uncertainties: np.ndarray,  # Already scaled by uncertainty_weight * max_distance
    do_embeddings: bool,
    max_distance: float,
    embedding_weight: float,
) -> np.ndarray:
    """
    Compute cost matrix with uncertainty penalties.
    Uncertainty is added directly to spatial costs: total_cost = spatial_cost + uncertainty
    """
    n_dets, n_tracks = det_positions.shape[0], track_positions.shape[0]
    cost_matrix = np.full((n_dets, n_tracks), np.inf, dtype=np.float32)
    
    for i in range(n_dets):
        for j in range(n_tracks):
            # Compute spatial distance
            dx = det_positions[i, 0] - track_positions[j, 0]
            dy = det_positions[i, 1] - track_positions[j, 1]
            spatial_cost = np.sqrt(dx*dx + dy*dy)
            
            # Add uncertainty penalty to spatial cost
            total_cost = spatial_cost + track_uncertainties[j]
            
            # Only proceed if within max distance (after uncertainty adjustment)
            if total_cost <= max_distance:
                # Add embedding cost if enabled
                if do_embeddings:
                    embedding_cost = scaled_embedding_matrix[i, j] * max_distance * embedding_weight
                    total_cost += embedding_cost
                
                cost_matrix[i, j] = total_cost
    
    return cost_matrix


@nb.njit(fastmath=True, cache=True)
def compute_cost_matrix_with_min_distance(
    det_positions: np.ndarray,
    track_last_positions: np.ndarray,
    track_kalman_positions: np.ndarray,
    scaled_embedding_matrix: np.ndarray,
    track_uncertainties: np.ndarray,  # Already scaled by uncertainty_weight * max_distance
    do_embeddings: bool,
    max_distance: float,
    embedding_weight: float,
) -> np.ndarray:
    """
    Compute cost matrix with minimum distance (like old version's compute_cost_matrix_vectorized).
    Uses minimum of distance to last detection and Kalman prediction.
    """
    n_dets, n_tracks = det_positions.shape[0], track_last_positions.shape[0]
    cost_matrix = np.full((n_dets, n_tracks), np.inf, dtype=np.float32)
    
    for i in range(n_dets):
        for j in range(n_tracks):
            # Distance to last detection
            dx_last = det_positions[i, 0] - track_last_positions[j, 0]
            dy_last = det_positions[i, 1] - track_last_positions[j, 1]
            dist_to_last = np.sqrt(dx_last * dx_last + dy_last * dy_last)
            
            # Distance to Kalman prediction
            dx_kalman = det_positions[i, 0] - track_kalman_positions[j, 0]
            dy_kalman = det_positions[i, 1] - track_kalman_positions[j, 1]
            dist_to_kalman = np.sqrt(dx_kalman * dx_kalman + dy_kalman * dy_kalman)
            
            # Use minimum distance (key logic from old version)
            spatial_cost = min(dist_to_last, dist_to_kalman)
            
            # Add uncertainty penalty to spatial cost
            spatial_cost = spatial_cost + track_uncertainties[j]
            
            # Only proceed if within max distance (after uncertainty adjustment)
            if spatial_cost <= max_distance:
                # Add embedding cost if enabled (CONSISTENT FORMULA)
                if do_embeddings:
                    embedding_cost = scaled_embedding_matrix[i, j] * embedding_weight * max_distance
                    spatial_cost += embedding_cost
                
                cost_matrix[i, j] = spatial_cost
    
    return cost_matrix


# ============================================================================
# COLLISION OPTIMIZATION FUNCTIONS  
# ============================================================================


@nb.njit(fastmath=True, cache=True)
def compute_freeze_flags_vectorized(positions: np.ndarray, safety_distance: float) -> np.ndarray:
    """
    Vectorized computation of freeze flags based on track proximity.
    Returns boolean array where True means track should be frozen.
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


# ============================================================================
# ENHANCED FAST TRACK CLASS WITH OBSERVATION HISTORY
# ============================================================================


@dataclass
class FastTrackState:
    """Enhanced track state with N-embedding history and kalman_type support"""

    id: int
    class_id: Optional[int] = None

    position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    predicted_position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    bbox: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))

    # Kalman state (for "simple" type)
    kalman_state: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))

    last_detection_pos: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    last_detection_frame: int = 0

    # Observation history for both types
    observation_history: deque = field(default_factory=lambda: deque(maxlen=5))
    observation_frames: deque = field(default_factory=lambda: deque(maxlen=5))
    
    # OC-SORT specific arrays (for "oc" type)
    observation_history_array: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.float32)
    )
    observation_frames_array: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )
    
    # Track type
    kalman_type: str = "simple"
    
    # Embedding freeze tracking  
    embedding_frozen: bool = False  # Whether embeddings are frozen due to high density
    last_safe_embedding: Optional[np.ndarray] = None  # Last embedding before freeze

    # Embedding history with configurable size
    embedding_history: deque = field(default_factory=lambda: deque(maxlen=5))
    embedding_method: Literal["average", "best_match", "weighted_average"] = "average"

    # Cache for average embedding
    _cached_avg_embedding: Optional[np.ndarray] = None
    _cache_valid: bool = False

    # Cache for multi-embedding computation
    _cached_representative_embedding: Optional[np.ndarray] = None
    _representative_cache_valid: bool = False

    # Keep for backward compatibility
    avg_embedding: Optional[np.ndarray] = None
    embedding_update_count: int = 0

    age: int = 0
    hits: int = 0
    misses: int = 0
    confirmed: bool = False
    detection_confidence: float = 0.0
    confidence_score: float = 0.5

    lost_frames: int = 0

    def __post_init__(self):
        self.kalman_state[:2] = self.position
        self.last_detection_pos = self.position.copy()
        self.predicted_position = self.position.copy()

    def get_observation_prediction(self, current_frame: int, max_history: int = 5) -> np.ndarray:
        """Get observation-based prediction using recent detection history"""
        if len(self.observation_history) < 2:
            return self.predicted_position

        # Simple linear extrapolation from last two observations
        pos1 = self.observation_history[-2]
        pos2 = self.observation_history[-1]
        frame1 = self.observation_frames[-2]
        frame2 = self.observation_frames[-1]

        if frame2 == frame1:  # Avoid division by zero
            return pos2.copy()

        dt = current_frame - frame2
        velocity = (pos2 - pos1) / (frame2 - frame1)
        predicted = pos2 + velocity * dt

        return predicted.astype(np.float32)

    def update_observation_history(self, position: np.ndarray, frame: int):
        """Update observation history for observation-based prediction"""
        self.observation_history.append(position.copy())
        self.observation_frames.append(frame)

    def set_embedding_params(
            self,
            max_embeddings: int = 5,
            method: Literal["average", "best_match", "weighted_average"] = "average",
    ):
        """Configure embedding storage parameters"""
        self.embedding_history = deque(maxlen=max_embeddings)
        self.embedding_method = method
        self._cache_valid = False

    def add_embedding(self, embedding: np.ndarray):
        """Add new embedding to history - FIXED to avoid double normalization"""
        # Don't update embeddings during collision/occlusion
        if self.embedding_frozen:
            return
            
        if embedding is not None:
            embedding = np.asarray(embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                # FIXED: Only normalize if not already normalized
                if abs(norm - 1.0) > 0.01:  # Allow small numerical errors
                    normalized_emb = embedding / (norm + 1e-8)
                else:
                    normalized_emb = embedding  # Already normalized

                # Store embedding as last safe embedding before potential collision
                # Only store if we don't already have one to avoid redundant copies
                if self.last_safe_embedding is None:
                    self.last_safe_embedding = normalized_emb.copy()

                self.embedding_history.append(normalized_emb.copy())
                self.embedding_update_count += 1

                # Always invalidate cache on new embedding
                self._cache_valid = False
                self._representative_cache_valid = False

                # Update avg_embedding for backward compatibility
                self._update_avg_embedding()

    def _update_avg_embedding(self):
        """Update avg_embedding with caching"""
        if len(self.embedding_history) > 0:
            if self.embedding_method == "average":
                if not self._cache_valid:
                    self._cached_avg_embedding = np.mean(list(self.embedding_history), axis=0)
                    self._cache_valid = True
                self.avg_embedding = self._cached_avg_embedding
            elif self.embedding_method == "weighted_average":
                # More recent embeddings have higher weight
                weights = np.exp(np.linspace(-1, 0, len(self.embedding_history)))
                weights /= weights.sum()
                self.avg_embedding = np.average(
                    list(self.embedding_history), axis=0, weights=weights
                )
            else:  # best_match
                self.avg_embedding = self.embedding_history[-1].copy()

    def get_representative_embedding_for_assignment(
            self, det_embeddings: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Get the best representative embedding for assignment, with caching"""
        if len(self.embedding_history) == 0:
            return None

        if not self._representative_cache_valid:
            if len(self.embedding_history) == 1:
                self._cached_representative_embedding = self.embedding_history[0].copy()
            elif self.embedding_method == "average":
                self._cached_representative_embedding = np.mean(
                    list(self.embedding_history), axis=0
                )
            elif self.embedding_method == "weighted_average":
                weights = np.exp(np.linspace(-1, 0, len(self.embedding_history)))
                weights /= weights.sum()
                self._cached_representative_embedding = np.average(
                    list(self.embedding_history), axis=0, weights=weights
                )
            else:  # best_match
                if det_embeddings is not None and len(det_embeddings) > 0:
                    # Find best match to current detections (same as multi-embedding logic)
                    best_avg_sim = -1.0
                    best_emb = None
                    for track_emb in self.embedding_history:
                        try:
                            avg_sim = np.mean(det_embeddings @ track_emb)
                            if avg_sim > best_avg_sim:
                                best_avg_sim = avg_sim
                                best_emb = track_emb
                        except:
                            # Fallback if shapes don't match
                            best_emb = track_emb
                            break
                    self._cached_representative_embedding = (
                        best_emb.copy()
                        if best_emb is not None
                        else self.embedding_history[-1].copy()
                    )
                else:
                    # Fallback to most recent
                    self._cached_representative_embedding = self.embedding_history[-1].copy()

            self._representative_cache_valid = True

        return self._cached_representative_embedding

    def get_embedding_stats(self) -> dict:
        """Get statistics about stored embeddings"""
        if len(self.embedding_history) == 0:
            return {"count": 0, "method": self.embedding_method}

        embeddings = np.array(list(self.embedding_history))

        # Compute pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j])
                similarities.append(sim)

        return {
            "count": len(self.embedding_history),
            "method": self.embedding_method,
            "avg_internal_similarity": np.mean(similarities) if similarities else 1.0,
            "std_internal_similarity": np.std(similarities) if similarities else 0.0,
            "min_similarity": np.min(similarities) if similarities else 1.0,
            "max_similarity": np.max(similarities) if similarities else 1.0,
        }

    def update_with_detection(
            self,
            new_pos: np.ndarray,
            embedding: Optional[np.ndarray],
            bbox: Optional[np.ndarray],
            current_frame: int,
            detection_confidence: float = 0.0,
            is_reid: bool = False,
    ):
        """Updated detection update with embedding history and observation history"""
        # Update observation history FIRST
        self.update_observation_history(new_pos, current_frame)

        # Same spatial update as before
        self.last_detection_pos = new_pos.astype(np.float32)
        self.last_detection_frame = current_frame
        self.detection_confidence = detection_confidence
        self.confidence_score = detection_confidence

        new_pos_f32 = new_pos.astype(np.float32)

        if self.kalman_type == "simple":
            # Check if this is a ReID update (track was lost and now found again)
            is_reid_update = is_reid or (self.misses > 0)
            
            if is_reid_update:
                # Smooth ReID update to prevent jerky motion from stale velocity
                # Update position to current detection
                self.kalman_state[0] = new_pos_f32[0]  # x position
                self.kalman_state[1] = new_pos_f32[1]  # y position  
                
                # Smooth velocity reset using exponential decay instead of hard reset
                velocity_decay = 0.3  # Keep 30% of old velocity to maintain some continuity
                self.kalman_state[2] *= velocity_decay  # Reduce x velocity smoothly
                self.kalman_state[3] *= velocity_decay  # Reduce y velocity smoothly
            else:
                # Normal Kalman filter update
                self.kalman_state = simple_kalman_update(self.kalman_state, new_pos_f32)

                if self.hits > 0:
                    dt = 1.0
                    new_velocity = (new_pos_f32 - self.position) / dt
                    self.kalman_state[2] = 0.7 * new_velocity[0] + 0.3 * self.kalman_state[2]
                    self.kalman_state[3] = 0.7 * new_velocity[1] + 0.3 * self.kalman_state[3]

            self.position = self.kalman_state[:2]
            self.velocity = self.kalman_state[2:]
            self.predicted_position = self.kalman_state[:2] + self.kalman_state[2:]
            
        elif self.kalman_type == "oc":
            # OC-SORT style update
            self.observation_history_array, self.observation_frames_array = oc_sort_update(
                self.observation_history_array,
                self.observation_frames_array,
                new_pos_f32,
                current_frame,
                max_history=30
            )
            
            # Update current state
            self.position = new_pos_f32
            
            # Compute velocity from observations
            if len(self.observation_frames_array) >= 2:
                dt = self.observation_frames_array[-1] - self.observation_frames_array[-2]
                if dt > 0:
                    self.velocity[0] = (self.observation_history_array[-1, 0] - 
                                       self.observation_history_array[-2, 0]) / dt
                    self.velocity[1] = (self.observation_history_array[-1, 1] - 
                                       self.observation_history_array[-2, 1]) / dt

        # Add embedding to history
        if embedding is not None:
            self.add_embedding(embedding)

        if bbox is not None:
            self.bbox = np.asarray(bbox, dtype=np.float32)

        # Increment counters
        self.hits += 1
        self.age += 1
        self.misses = 0
        self.lost_frames = 0

        # Note: confirmation is now handled by the tracker to respect config

    def predict_only(self):
        """Prediction step - behavior depends on kalman_type"""
        if self.kalman_type == "simple":
            # Simple Kalman prediction step - keeps position at last detection
            self.kalman_state = simple_kalman_predict(self.kalman_state)
            # Don't update self.position - keep it at last detection position for display
            self.velocity = self.kalman_state[2:].copy()
            self.predicted_position = self.kalman_state[:2].copy()  # Store predicted position separately
        elif self.kalman_type == "oc":
            # OC-SORT style - compute prediction and store it
            if hasattr(self, 'observation_history_array') and len(self.observation_history_array) > 0:
                pred_state = oc_sort_predict(
                    self.observation_history_array,
                    self.observation_frames_array,
                    self.frame_count  # Use current frame + 1 for prediction
                )
                self.predicted_position = pred_state[:2].copy()
            else:
                # No history available, use last position
                self.predicted_position = self.position.copy()
        
        self.age += 1
        self.misses += 1
        self.lost_frames += 1

    def get_predicted_position(self, current_frame: int) -> np.ndarray:
        """Get predicted position based on kalman_type"""
        if self.kalman_type == "simple":
            return self.predicted_position
        elif self.kalman_type == "oc":
            # Use OC-SORT prediction
            pred_state = oc_sort_predict(
                self.observation_history_array,
                self.observation_frames_array,
                current_frame
            )
            return pred_state[:2]  # Return just x, y position
        else:
            return self.predicted_position

    def add_embedding_fast(self, embedding: np.ndarray, pre_normalized: bool = True):
        """Fast embedding addition without expensive checks"""
        # Don't update embeddings during collision/occlusion
        if self.embedding_frozen:
            return
            
        if embedding is None:
            return

        if not pre_normalized:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        # Store embedding as last safe embedding before potential collision
        # Only store if we don't already have one to avoid redundant copies
        if self.last_safe_embedding is None:
            self.last_safe_embedding = embedding.copy()

        # Smart cache invalidation - only when it actually affects the result
        if len(self.embedding_history) > 0:  # Only invalidate if we have existing embeddings
            self._cache_valid = False
            self._representative_cache_valid = False

        self.embedding_history.append(embedding)
        self.embedding_update_count += 1
        self.avg_embedding = None  # Mark dirty for lazy evaluation

    @property
    def avg_embedding_lazy(self):
        """Compute average embedding only when accessed"""
        if self.avg_embedding is None and len(self.embedding_history) > 0:
            if self.embedding_method == "average":
                self.avg_embedding = np.mean(np.array(self.embedding_history), axis=0)
            elif self.embedding_method == "weighted_average":
                weights = np.exp(np.linspace(-1, 0, len(self.embedding_history)))
                weights /= weights.sum()
                self.avg_embedding = np.average(
                    np.array(self.embedding_history), axis=0, weights=weights
                )
            else:
                self.avg_embedding = self.embedding_history[-1]
        return self.avg_embedding

    def unfreeze_embeddings(self, restore_last_safe: bool = True):
        """Unfreeze embeddings and optionally restore the last safe embedding"""
        self.embedding_frozen = False
        
        # Optionally restore the last safe embedding to prevent drift during collision
        if restore_last_safe and self.last_safe_embedding is not None:
            # Clear any embeddings that might have been added during frozen state
            # and restore the safe embedding
            if len(self.embedding_history) == 0 or self.embedding_method == "best_match":
                # For best_match or empty history, replace with safe embedding
                if len(self.embedding_history) > 0:
                    self.embedding_history.pop()
                self.embedding_history.append(self.last_safe_embedding.copy())
            
            # Invalidate caches
            self._cache_valid = False
            self._representative_cache_valid = False
            self.avg_embedding = None
            self._update_avg_embedding()
        
        # Clear the safe embedding
        self.last_safe_embedding = None


# ============================================================================
# SWARM SORT TRACKER
# ============================================================================


class SwarmSortTracker:
    """SwarmSort Multi-Object Tracker.

    SwarmSortTracker is the main tracking class that implements a complete multi-object
    tracking pipeline combining:
    - Kalman filtering for motion prediction
    - Observation-based motion prediction
    - Hungarian/Greedy/Hybrid algorithm assignments
    - Deep learning embeddings for appearance matching
    - Re-identification for recovering lost tracks
    - Probabilistic cost computation for robust associations

    The tracker maintains active tracks and pending detections (for track initialization).
    ReID is performed on active tracks that have missed recent detections. It supports
    both embedding-based and motion-only tracking modes with extensive configuration options.

    Attributes:
        config (SwarmSortConfig): Configuration object containing all tracker parameters
        tracks (dict): Active tracks indexed by track ID
        pending_detections (dict): Unconfirmed detections being evaluated for track creation
        next_track_id (int): ID counter for new tracks
        frame_count (int): Current frame number
        embedding_scaler (EmbeddingDistanceScaler): Adaptive embedding distance scaler

    Example:
        >>> from swarmsort import SwarmSortTracker, SwarmSortConfig, Detection
        >>> import numpy as np

        >>> # Create tracker with 3D assignment
        >>> config = SwarmSortConfig(assignment_strategy="3d")
        >>> tracker = SwarmSortTracker(config)

        >>> # Create detection
        >>> detection = Detection(
        ...     position=np.array([10.0, 20.0], dtype=np.float32),
        ...     confidence=0.9
        ... )

        >>> # Update tracker
        >>> tracked_objects = tracker.update([detection])
        >>> print(f"Tracking {len(tracked_objects)} objects")
    """

    def __init__(
            self,
            config: Optional[Union[SwarmSortConfig, dict]] = None,
            embedding_type: Optional[str] = None,
            use_gpu: Optional[bool] = None,
            **kwargs,
    ):
        # Handle configuration
        if config is None:
            self.config = SwarmSortConfig()
        elif isinstance(config, dict):
            self.config = SwarmSortConfig.from_dict(config)
        else:
            self.config = config

        # Handle embedding extractor setup
        # Priority: explicit embedding_type parameter > config.embedding_function
        self.embedding_extractor = None
        actual_embedding_type = embedding_type or getattr(self.config, 'embedding_function', 'cupytexture')
        
        if actual_embedding_type is not None and self.config.do_embeddings:
            try:
                from .embeddings import get_embedding_extractor
                self.embedding_extractor = get_embedding_extractor(
                    actual_embedding_type,
                    use_gpu=(use_gpu if use_gpu is not None else True)
                )
                logger.debug(f"Using embedding extractor: {actual_embedding_type}")
            except Exception as e:
                logger.warning(f"Failed to create embedding extractor '{actual_embedding_type}': {e}")

        # Map config to internal parameters
        self.use_probabilistic_costs = getattr(self.config, "use_probabilistic_costs", True)

        # Core parameters
        self.max_distance = self.config.max_distance
        self.init_conf_threshold = getattr(self.config, "init_conf_threshold", 0.15)
        self.embedding_weight = self.config.embedding_weight
        self.max_track_age = self.config.max_track_age
        self.detection_conf_threshold = self.config.detection_conf_threshold
        self.do_embeddings = self.config.do_embeddings
        self.use_class_id_matching = getattr(self.config, "use_class_id_matching", False)

        # Embedding history configuration
        self.max_embeddings_per_track = self.config.max_embeddings_per_track
        self.embedding_matching_method = self.config.embedding_matching_method

        # Assignment strategy parameters
        self.assignment_strategy = getattr(self.config, "assignment_strategy", "hybrid")
        self.greedy_threshold = getattr(self.config, "greedy_threshold", 30.0)
        self.greedy_confidence_boost = getattr(self.config, "greedy_confidence_boost", 0.8)
        self.hungarian_fallback_threshold = getattr(self.config, "hungarian_fallback_threshold", 1.5)
        
        # Sparse computation parameters for large-scale scenarios
        self.sparse_computation_threshold = getattr(self.config, "sparse_computation_threshold", 300)

        # Kalman type
        self.kalman_type = getattr(self.config, "kalman_type", "simple")
        
        # Uncertainty-based cost system parameters
        self.uncertainty_weight = getattr(self.config, "uncertainty_weight", 0.0)
        self.local_density_radius = getattr(self.config, "local_density_radius", 30.0)
        
        # Simplified embedding freeze parameters
        self.collision_freeze_embeddings = getattr(self.config, "collision_freeze_embeddings", True)
        self.embedding_freeze_density = getattr(self.config, "embedding_freeze_density", 2)
        
        # Embedding freeze optimization - only check every N frames
        # Removed freeze throttling - collision detection must run every frame for safety


        # ReID parameters
        self.reid_enabled = self.config.reid_enabled
        self.reid_max_distance = self.config.reid_max_distance
        self.reid_embedding_threshold = self.config.reid_embedding_threshold

        # INITIALIZATION PARAMETERS
        self.min_consecutive_detections = self.config.min_consecutive_detections
        self.max_detection_gap = getattr(self.config, "max_detection_gap", 2)
        self.pending_detection_distance = getattr(self.config, "pending_detection_distance", 50.0)

        # Debug options
        self.debug_embeddings = getattr(self.config, "debug_embeddings", False)
        self.plot_embeddings = getattr(self.config, "plot_embeddings", False)
        self.debug_timings = getattr(self.config, "debug_timings", False)

        # Storage
        self._tracks = {}
        self._pending_detections = []  # INITIALIZATION LOGIC!
        self._next_id = 1
        self._frame_count = 0

        # Store assignment methods for debugging/analysis
        self._assignment_methods = {}

        # embedding scaler
        self.embedding_scaler = EmbeddingDistanceScaler(
            method="min_robustmax", update_rate=0.05, min_samples=200
        )

        # Pre-compile Numba
        self._precompile_numba()

        # Pre-allocate reusable arrays
        self._reusable_det_embeddings = None
        self._reusable_track_embeddings = None
        self._reusable_cost_matrix = None
        self._max_dets_seen = 0
        self._max_tracks_seen = 0
        
        # Frame-level embedding caches to avoid recomputation
        self._frame_det_embeddings_valid = -1
        self._cached_det_embeddings = None
        self._frame_track_embeddings_valid = -1
        self._cached_track_embeddings = None
        
        # Embedding distance matrix cache (most expensive computation)
        self._cached_embedding_distances = None
        self._embedding_distances_valid = -1

        logger.info(
            f"\033[0;38;5;45mInitialized SwarmSortTracker with parameters:\n"
            + "\n".join(f"{k}={v}" for k, v in vars(self).items() if not k.startswith("_") and k != "config")
            + "\033[0m"
        )

    def _create_new_track(self, track_id: int, position: np.ndarray) -> FastTrackState:
        """Create new track with embedding configuration"""
        track = FastTrackState(id=track_id, position=position, kalman_type=self.kalman_type)
        track.set_embedding_params(self.max_embeddings_per_track, self.embedding_matching_method)
        return track

    def _precompile_numba(self):
        """Pre-compile Numba functions"""
        try:
            dummy_emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            dummy_pos = np.array([[100.0, 100.0]], dtype=np.float32)
            dummy_embs = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
            dummy_diff = np.array([1.0, 1.0], dtype=np.float32)
            dummy_cov = np.eye(2, dtype=np.float32)

            _ = cosine_similarity_normalized(dummy_emb, dummy_emb)
            _ = compute_cost_matrix_vectorized(
                dummy_pos, dummy_pos, dummy_pos, dummy_embs, dummy_embs, True, 100.0, 0.3
            )
            _ = fast_mahalanobis_distance(dummy_diff, dummy_cov)
            _ = fast_gaussian_fusion(dummy_diff, dummy_cov, dummy_diff, dummy_cov)
            
            # Pre-compile uncertainty computation
            dummy_misses = np.array([0, 1], dtype=np.float32)
            dummy_ages = np.array([1, 2], dtype=np.float32)
            _ = compute_track_uncertainties_numba(dummy_pos, dummy_misses, dummy_ages, 50.0)


            logger.debug("Numba functions compiled successfully")
        except Exception as e:
            logger.warning(f"Numba compilation failed: {e}")

    def update(self, detections: List[Detection], frame: Optional[np.ndarray] = None) -> List[TrackedObject]:
        """Main update with optional timing per call"""
        self._frame_count += 1

        self.timings = {}
        timer = Timer()

        def start(key):
            timer.start(key)

        def stop(key):
            timer.stop(key, self.timings)

        if not detections:
            return self._handle_empty_frame()

        start("filter_conf")
        valid_detections = [
            det
            for det in detections
            if self._get_detection_confidence(det) >= self.detection_conf_threshold
        ]
        stop("filter_conf")

        if not valid_detections:
            return self._handle_empty_frame()

        if self.debug_embeddings and self._frame_count % 10 == 0:
            start("debug_embeddings")
            self._debug_embeddings(valid_detections)
            stop("debug_embeddings")

        # NOTE: Embedding normalization is handled by vectorized computation later
        # Removed redundant per-detection normalization to avoid double normalization

        # Update embedding freeze status based on local density
        start("embedding_freeze_update")
        self._update_embedding_freeze_status()
        stop("embedding_freeze_update")

        start("assignment")
        # CORRECTED LOGIC: First, decide on the cost model (probabilistic or standard).
        # Then, apply the chosen assignment strategy.
        if self.use_probabilistic_costs:
            # The probabilistic model has its own integrated assignment logic.
            # It's a specialized path.
            assignment_fn = self._fast_assignment_probabilistic
        else:
            # For the standard cost model, select the assignment strategy.
            if self.assignment_strategy == "greedy":
                assignment_fn = self._greedy_assignment
            elif self.assignment_strategy == "hybrid":
                assignment_fn = self._hybrid_assignment
            else:  # Default to "hungarian"
                assignment_fn = self._fast_assignment
        
        matches, unmatched_dets, unmatched_tracks = assignment_fn(valid_detections, timer=timer, start=start, stop=stop)
        stop("assignment")


        start("update_matched")
        self._update_matched_tracks(matches, valid_detections)
        stop("update_matched")

        start("handle_unmatched_tracks")
        self._handle_unmatched_tracks(unmatched_tracks)
        stop("handle_unmatched_tracks")

        if self.reid_enabled and any(track.misses > 0 for track in self._tracks.values()):
            start("reid")
            reid_matches = self._attempt_reid(valid_detections, unmatched_dets)
            unmatched_dets = [idx for idx in unmatched_dets if idx not in reid_matches]
            stop("reid")

        start("handle_unmatched_dets")
        self._handle_unmatched_detections(unmatched_dets, valid_detections)
        stop("handle_unmatched_dets")

        start("update_pending")
        self._update_pending_detections()
        stop("update_pending")

        start("cleanup")
        self._cleanup_tracks()
        stop("cleanup")

        start("get_results")
        result = self._get_results()
        stop("get_results")

        if self.debug_timings:
            formatted_timings = {}
            for k, v in self.timings.items():
                if isinstance(v, str):
                    formatted_timings[k] = v  # Already formatted
                else:
                    formatted_timings[k] = f"{v * 1000:.2f} ms"
            print(f"[Frame {self._frame_count}] Timings:", formatted_timings)

        return result

    def _handle_unmatched_detections(self, unmatched_det_indices, detections):
        """FAST handle unmatched detections with vectorized pending matching"""
        if not unmatched_det_indices:
            return

        # Filter detections that meet initialization confidence threshold
        init_eligible_detections = []
        for det_idx in unmatched_det_indices:
            detection = detections[det_idx]
            det_conf = self._get_detection_confidence(detection)
            if det_conf >= self.init_conf_threshold:
                init_eligible_detections.append((det_idx, detection))

        if not init_eligible_detections:
            return

        # Extract positions of initialization-eligible detections
        det_positions = np.array(
            [detection.position.flatten()[:2] for _, detection in init_eligible_detections],
            dtype=np.float32,
        )

        # Extract pending positions for vectorized comparison
        if self._pending_detections:
            pending_positions = np.array(
                [pending.average_position for pending in self._pending_detections], dtype=np.float32
            )

            # Optimized: Use squared distances to avoid sqrt until necessary
            diff = det_positions[:, None, :] - pending_positions[None, :, :]
            distances_sq = np.sum(diff * diff, axis=2)  # Shape: (n_dets, n_pending)
            
            threshold_sq = self.pending_detection_distance ** 2
            
            # Find closest pending for each detection (using squared distances)
            min_distances_sq = np.min(distances_sq, axis=1)
            closest_pending_indices = np.argmin(distances_sq, axis=1)

            # Process each detection
            matched_pending_indices = set()
            for i, (det_idx, detection) in enumerate(init_eligible_detections):
                if min_distances_sq[i] < threshold_sq:  # Compare squared distances
                    # Found a close pending detection
                    pending_idx = closest_pending_indices[i]

                    # Avoid double-matching the same pending detection
                    if pending_idx not in matched_pending_indices:
                        matched_pending_indices.add(pending_idx)

                        # Update the pending detection
                        position = det_positions[i]
                        embedding = getattr(detection, "embedding", None)
                        bbox = getattr(detection, "bbox", None)

                        self._update_pending_detection(
                            self._pending_detections[pending_idx], position, embedding, bbox
                        )
                    # else: skip this detection (another detection already matched this pending)
                else:
                    # Create new pending detection
                    position = det_positions[i]
                    embedding = getattr(detection, "embedding", None)
                    bbox = getattr(detection, "bbox", None)
                    det_conf = self._get_detection_confidence(detection)

                    new_pending = PendingDetection(
                        position=position.copy(),
                        embedding=embedding.copy() if embedding is not None else None,
                        bbox=np.asarray(bbox, dtype=np.float32)
                        if bbox is not None
                        else np.zeros(4, dtype=np.float32),
                        confidence=det_conf,
                        first_seen_frame=self._frame_count,
                        last_seen_frame=self._frame_count,
                        average_position=position.copy(),
                    )
                    self._pending_detections.append(new_pending)
        else:
            # No existing pending detections - create new ones for all initialization-eligible detections
            for det_idx, detection in init_eligible_detections:
                position = detection.position.flatten()[:2].astype(np.float32)
                embedding = getattr(detection, "embedding", None)
                bbox = getattr(detection, "bbox", None)
                det_conf = self._get_detection_confidence(detection)

                new_pending = PendingDetection(
                    position=position.copy(),
                    embedding=embedding.copy() if embedding is not None else None,
                    bbox=np.asarray(bbox, dtype=np.float32)
                    if bbox is not None
                    else np.zeros(4, dtype=np.float32),
                    confidence=det_conf,
                    first_seen_frame=self._frame_count,
                    last_seen_frame=self._frame_count,
                    average_position=position.copy(),
                )
                self._pending_detections.append(new_pending)

    def _update_pending_detection(
        self,
        pending: PendingDetection,
        position: np.ndarray,
        embedding: Optional[np.ndarray],
        bbox: Optional[np.ndarray] = None,
    ):
        """Update pending detection with new observation"""
        # Simple running average
        alpha = 0.3
        pending.average_position = (1 - alpha) * pending.average_position + alpha * position
        pending.position = position.copy()

        # Safe bbox update
        if bbox is not None:
            if isinstance(bbox, np.ndarray):
                pending.bbox = bbox.copy()
            else:
                pending.bbox = np.array(bbox, dtype=np.float32)
        elif pending.bbox.sum() == 0:
            pending.bbox = np.zeros(4, dtype=np.float32)

        # Update embedding if available
        if embedding is not None:
            if pending.embedding is not None:
                pending.embedding = 0.7 * pending.embedding + 0.3 * embedding
            else:
                pending.embedding = embedding.copy()

        # Update frame tracking
        frame_gap = self._frame_count - pending.last_seen_frame
        pending.last_seen_frame = self._frame_count
        pending.total_detections += 1

        if frame_gap == 1:  # Consecutive frame
            pending.consecutive_frames += 1
        else:  # There was a gap
            pending.consecutive_frames = 1

    def _update_pending_detections(self):
        """Update pending detections and promote to tracks"""
        detections_to_remove = []

        for i, pending in enumerate(self._pending_detections):
            frame_gap = self._frame_count - pending.last_seen_frame

            # Check if pending detection should become a track
            if (
                pending.consecutive_frames >= self.min_consecutive_detections
                and pending.total_detections >= self.min_consecutive_detections
                and frame_gap <= 1
            ):
                # Create new track
                self._create_track_from_pending(pending)
                detections_to_remove.append(i)

            # Remove old pending detections
            elif frame_gap > self.max_detection_gap + 2:
                detections_to_remove.append(i)

        # Remove in reverse order
        for i in reversed(detections_to_remove):
            del self._pending_detections[i]

    def _create_track_from_pending(self, pending: PendingDetection):
        """Create track from pending detection with proper embedding setup"""
        new_track = FastTrackState(id=self._next_id, position=pending.average_position.copy(), kalman_type=self.kalman_type)

        # IMPORTANT: Set embedding parameters
        new_track.set_embedding_params(
            max_embeddings=self.max_embeddings_per_track, method=self.embedding_matching_method
        )

        # FIXED: Initialize track with pending embedding if available (ensure normalization)
        if pending.embedding is not None:
            emb = np.asarray(pending.embedding, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                if abs(norm - 1.0) > 0.01:
                    normalized_emb = emb / norm
                else:
                    normalized_emb = emb
                new_track.embedding_history.append(normalized_emb)
                new_track.embedding_update_count = 1
                new_track._cache_valid = False
                new_track._representative_cache_valid = False

        # Update with detection
        new_track.update_with_detection(
            pending.average_position,
            pending.embedding,  # This will add to history again, but that's ok
            pending.bbox,
            self._frame_count,
            pending.confidence,
        )

        new_track.hits = min(pending.total_detections, 3)
        new_track.age = pending.consecutive_frames

        self._tracks[self._next_id] = new_track

        if self.debug_embeddings:
            logger.info(f"Created track {self._next_id} from pending detection")
            logger.info(f"  Embedding method: {self.embedding_matching_method}")
            logger.info(f"  Initial embeddings: {len(new_track.embedding_history)}")

        self._next_id += 1

    def _compute_cost_matrix(self, detections, tracks, start=None, stop=None):
        """
        Centralized cost matrix computation.
        This function computes the cost matrix based on the configured cost function
        (probabilistic or standard) and handles embedding calculations.
        """
        n_dets = len(detections)
        n_tracks = len(tracks)

        # Extract positions
        det_positions = np.array(
            [det.position.flatten()[:2] for det in detections], dtype=np.float32
        )
        track_last_positions = np.array([t.last_detection_pos for t in tracks], dtype=np.float32)
        track_kalman_positions = np.array([t.predicted_position for t in tracks], dtype=np.float32)

        # Check if we need embeddings
        do_embeddings = (
            self.do_embeddings and
            all(
                hasattr(det, "embedding") and det.embedding is not None for det in detections
            )
        )

        scaled_embedding_matrix = np.zeros((n_dets, n_tracks), dtype=np.float32)

        if do_embeddings:
            if start:
                start("embedding_computation")

            # Normalize detection embeddings
            det_embeddings = np.empty((n_dets, detections[0].embedding.shape[0]), dtype=np.float32)
            for i, det in enumerate(detections):
                emb = np.asarray(det.embedding, dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0 and abs(norm - 1.0) > 0.01:
                    det_embeddings[i] = emb / norm
                else:
                    det_embeddings[i] = emb

            # Collect ALL track embeddings for multi-history matching
            track_embeddings_list = []
            track_embedding_counts = []

            for track in tracks:
                if len(track.embedding_history) > 0:
                    # Collect ALL embeddings for this track
                    track_embs = np.array(list(track.embedding_history), dtype=np.float32)
                    track_embeddings_list.append(track_embs)
                    track_embedding_counts.append(len(track.embedding_history))
                else:
                    # Empty track gets a single zero embedding
                    track_embeddings_list.append(np.zeros((1, det_embeddings.shape[1]), dtype=np.float32))
                    track_embedding_counts.append(0)

            # Flatten all track embeddings for efficient Numba processing
            if track_embeddings_list:
                track_embeddings_flat = np.vstack(track_embeddings_list)
            else:
                track_embeddings_flat = np.zeros((0, det_embeddings.shape[1]), dtype=np.float32)
            track_counts = np.array(track_embedding_counts, dtype=np.int32)

            # Convert method string to int for Numba
            method_map = {
                "last": 0,
                "average": 1,
                "weighted_average": 2,
                "best_match": 3
            }
            method_int = method_map.get(self.embedding_matching_method, 2)  # default to weighted_average

            # Compute distances using the specified method
            raw_distances = compute_embedding_distances_with_method(
                det_embeddings,
                track_embeddings_flat,
                track_counts,
                method_int
            )

            # Scale distances as before
            raw_distances_flat = raw_distances.flatten()
            scaled_distances_flat = self.embedding_scaler.scale_distances(raw_distances_flat)
            self.embedding_scaler.update_statistics(raw_distances_flat)
            scaled_embedding_matrix = scaled_distances_flat.reshape(n_dets, n_tracks)

            if stop: stop("embedding_computation")

        # Compute track uncertainties and scale them
        if self.uncertainty_weight > 0.0:
            start("uncertainty_computation")
            track_uncertainties = self._compute_track_uncertainties_batch(tracks)
            
            # Center uncertainty around zero: low uncertainty reduces cost, high uncertainty increases it.
            # The original uncertainty is in [0, 1]. Centering it makes it [-0.5, 0.5].
            centered_uncertainties = track_uncertainties - 0.5
            
            uncertainty_penalties = centered_uncertainties * self.uncertainty_weight * self.max_distance
            if stop: stop("uncertainty_computation")
        else:
            uncertainty_penalties = np.zeros(n_tracks, dtype=np.float32)

        # Compute final cost matrix using the appropriate Numba function
        # This function is now ONLY for the standard (non-probabilistic) cost model.
        cost_matrix = compute_cost_matrix_with_min_distance(
            det_positions,
            track_last_positions,
            track_kalman_positions,
            scaled_embedding_matrix,
            uncertainty_penalties,
            do_embeddings,
            self.max_distance,
            self.embedding_weight,
        )
        return cost_matrix

    def _fast_assignment(self, detections, timer=None, start=None, stop=None):
        """OPTIMIZED Hungarian assignment."""
        n_dets = len(detections)
        n_tracks = len(self._tracks)

        if n_tracks == 0:
            return [], list(range(n_dets)), []

        tracks = list(self._tracks.values())

        cost_matrix = self._compute_cost_matrix(detections, tracks, start, stop)

        if np.all(np.isinf(cost_matrix)):
            return [], list(range(n_dets)), list(range(n_tracks))

        cost_matrix[cost_matrix > self.max_distance] = self.max_distance * 2

        if start: start("hungarian_assignment")
        try:
            det_indices, track_indices = linear_sum_assignment(cost_matrix)
        except ValueError:
            return [], list(range(n_dets)), list(range(n_tracks))

        # Filter valid matches
        matches = []
        for d_idx, t_idx in zip(det_indices, track_indices):
            if cost_matrix[d_idx, t_idx] <= self.max_distance:
                matches.append((d_idx, t_idx))

        # Find unmatched
        matched_dets = {m[0] for m in matches}
        matched_tracks = {m[1] for m in matches}

        unmatched_dets = [i for i in range(n_dets) if i not in matched_dets]
        unmatched_tracks = [i for i in range(n_tracks) if i not in matched_tracks]
        if stop: stop("hungarian_assignment")

        return matches, unmatched_dets, unmatched_tracks

    def _fast_assignment_probabilistic(self, detections, timer=None, start=None, stop=None):
        """probabilistic assignment with proper multi-embedding support"""
        n_dets = len(detections)
        n_tracks = len(self._tracks)

        if n_tracks == 0:
            return [], list(range(n_dets)), []

        tracks = list(self._tracks.values())

        # Extract positions
        det_positions = np.array(
            [det.position.flatten()[:2] for det in detections], dtype=np.float32
        )
        track_positions = np.array([t.predicted_position for t in tracks], dtype=np.float32)
        track_last_positions = np.array([t.last_detection_pos for t in tracks], dtype=np.float32)

        # Check embeddings
        # Fixed: Don't require tracks to already have embeddings - they need to start somewhere!
        do_embeddings = all(
            hasattr(det, "embedding") and det.embedding is not None for det in detections
        )
        # Removed: and all(len(t.embedding_history) > 0 for t in tracks) - this prevented new tracks from getting embeddings!

        scaled_embedding_matrix = np.zeros((n_dets, n_tracks), dtype=np.float32)

        if do_embeddings:
            # Normalize detection embeddings (same as in _compute_cost_matrix)
            det_embeddings = np.empty((n_dets, detections[0].embedding.shape[0]), dtype=np.float32)
            for i, det in enumerate(detections):
                emb = np.asarray(det.embedding, dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0 and abs(norm - 1.0) > 0.01:
                    det_embeddings[i] = emb / norm
                else:
                    det_embeddings[i] = emb

            # Collect ALL track embeddings for multi-history matching (same as _compute_cost_matrix)
            track_embeddings_list = []
            track_embedding_counts = []

            for track in tracks:
                if len(track.embedding_history) > 0:
                    # Collect ALL embeddings for this track
                    track_embs = np.array(list(track.embedding_history), dtype=np.float32)
                    track_embeddings_list.append(track_embs)
                    track_embedding_counts.append(len(track.embedding_history))
                else:
                    # Empty track gets a single zero embedding
                    track_embeddings_list.append(np.zeros((1, det_embeddings.shape[1]), dtype=np.float32))
                    track_embedding_counts.append(0)

            # Flatten all track embeddings for efficient Numba processing
            if track_embeddings_list:
                track_embeddings_flat = np.vstack(track_embeddings_list)
            else:
                track_embeddings_flat = np.zeros((0, det_embeddings.shape[1]), dtype=np.float32)
            track_counts = np.array(track_embedding_counts, dtype=np.int32)

            # Convert method string to int for Numba
            method_map = {
                "last": 0,
                "average": 1,
                "weighted_average": 2,
                "best_match": 3
            }
            method_int = method_map.get(self.embedding_matching_method, 2)  # default to weighted_average

            # Compute distances using the specified method
            raw_distances_matrix = compute_embedding_distances_with_method(
                det_embeddings,
                track_embeddings_flat,
                track_counts,
                method_int
            )

            # Scale distances as before
            raw_distances_flat = raw_distances_matrix.flatten()
            scaled_distances_flat = self.embedding_scaler.scale_distances(raw_distances_flat)
            self.embedding_scaler.update_statistics(raw_distances_flat)
            scaled_embedding_matrix = scaled_distances_flat.reshape(n_dets, n_tracks)

        embedding_median = np.median(scaled_embedding_matrix)
        track_frames_since_detection = np.array(
            [self._frame_count - track.last_detection_frame for track in tracks], dtype=np.float32
        )

        cost_matrix = compute_probabilistic_cost_matrix_vectorized(
            det_positions,
            track_positions,
            track_last_positions,
            track_frames_since_detection,
            scaled_embedding_matrix,
            embedding_median,
            do_embeddings,
            self.max_distance,
            self.embedding_weight,
        )

        # Hungarian assignment
        if np.all(np.isinf(cost_matrix)):
            return [], list(range(n_dets)), list(range(n_tracks))

        cost_matrix[cost_matrix > self.max_distance] = self.max_distance * 2

        try:
            det_indices, track_indices = linear_sum_assignment(cost_matrix)
        except ValueError:
            return [], list(range(n_dets)), list(range(n_tracks))

        # Filter valid matches
        matches = []
        for d_idx, t_idx in zip(det_indices, track_indices):
            if cost_matrix[d_idx, t_idx] <= self.max_distance:
                matches.append((d_idx, t_idx))

        # Find unmatched
        matched_dets = {m[0] for m in matches}
        matched_tracks = {m[1] for m in matches}

        unmatched_dets = [i for i in range(n_dets) if i not in matched_dets]
        unmatched_tracks = [i for i in range(n_tracks) if i not in matched_tracks]

        return matches, unmatched_dets, unmatched_tracks

    def _compute_sparse_cost_matrix(self, det_positions, tracks, detections, max_search_radius):
        """Compute sparse cost matrix only for nearby pairs using spatial indexing."""
        n_dets = len(detections)
        n_tracks = len(tracks)
        
        # Only use spatial indexing for large numbers where overhead is worth it
        if n_dets < 200 or n_tracks < 200:
            return None, None  # Signal to use full matrix
        
        # Build spatial index using grid partitioning
        grid_size = max_search_radius
        track_grid = {}
        
        # Place tracks in grid cells
        for j, track in enumerate(tracks):
            x, y = track.predicted_position
            grid_x = int(x / grid_size)
            grid_y = int(y / grid_size)
            key = (grid_x, grid_y)
            if key not in track_grid:
                track_grid[key] = []
            track_grid[key].append(j)
        
        # Find candidate pairs
        sparse_pairs = []
        for i, det in enumerate(detections):
            x, y = det.position.flatten()[:2]
            grid_x = int(x / grid_size)
            grid_y = int(y / grid_size)
            
            # Check neighboring grid cells (3x3 neighborhood)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    key = (grid_x + dx, grid_y + dy)
                    if key in track_grid:
                        for j in track_grid[key]:
                            sparse_pairs.append((i, j))
        
        if not sparse_pairs:
            return None, None
            
        return sparse_pairs, track_grid
    
    def _hybrid_assignment(self, detections, timer=None, start=None, stop=None):
        """
        OPTIMIZED Hybrid assignment: Greedy for confident matches, Hungarian for the rest.
        Reduces ID switching by prioritizing high-confidence assignments.
        """
        n_dets = len(detections)
        n_tracks = len(self._tracks)

        if n_tracks == 0:
            return [], list(range(n_dets)), []

        tracks = list(self._tracks.values())

        cost_matrix = self._compute_cost_matrix(detections, tracks, start, stop)

        # OPTIMIZED HYBRID ASSIGNMENT LOGIC
        if start: start("hybrid_assignment")

        # Phase 1: NUMBA-accelerated greedy assignment for confident matches
        greedy_matches_array, remaining_dets_array, remaining_tracks_array = numba_greedy_assignment(
            cost_matrix, self.greedy_threshold
        )

        # Convert to expected format
        greedy_matches = [(int(greedy_matches_array[i, 0]), int(greedy_matches_array[i, 1]))
                         for i in range(greedy_matches_array.shape[0])]

        # Phase 2: Hungarian assignment for remaining detections/tracks
        remaining_dets = [int(x) for x in remaining_dets_array]
        remaining_tracks = [int(x) for x in remaining_tracks_array]

        hungarian_matches = []
        if remaining_dets and remaining_tracks:
            # Create reduced cost matrix
            reduced_cost_matrix = np.full((len(remaining_dets), len(remaining_tracks)),
                                        np.inf, dtype=np.float32)

            fallback_threshold = self.max_distance * self.hungarian_fallback_threshold
            
            for i, det_idx in enumerate(remaining_dets):
                for j, track_idx in enumerate(remaining_tracks):
                    original_cost = cost_matrix[det_idx, track_idx]
                    if original_cost <= fallback_threshold:
                        reduced_cost_matrix[i, j] = original_cost

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

                            if original_cost <= fallback_threshold:
                                hungarian_matches.append((original_det_idx, original_track_idx))

                    except ValueError:
                        pass  # Hungarian failed, continue without these matches

        if stop: stop("hybrid_assignment")

        # Combine results
        all_matches = greedy_matches + hungarian_matches
        matched_dets = {m[0] for m in all_matches}
        matched_tracks = {m[1] for m in all_matches}

        unmatched_dets = [i for i in range(n_dets) if i not in matched_dets]
        unmatched_tracks = [i for i in range(n_tracks) if i not in matched_tracks]

        return all_matches, unmatched_dets, unmatched_tracks

    def _greedy_assignment(self, detections, timer=None, start=None, stop=None):
        """OPTIMIZED Pure greedy assignment."""
        n_dets = len(detections)
        n_tracks = len(self._tracks)

        if n_tracks == 0:
            return [], list(range(n_dets)), []
        
        tracks = list(self._tracks.values())

        cost_matrix = self._compute_cost_matrix(detections, tracks, start, stop)

        # NUMBA-ACCELERATED greedy assignment - as fast as Hungarian!
        if start: start("greedy_assignment")

        matches_array, unmatched_dets_array, unmatched_tracks_array = numba_greedy_assignment(
            cost_matrix, self.max_distance
        )

        # Convert to expected format
        matches = [(int(matches_array[i, 0]), int(matches_array[i, 1]))
                  for i in range(matches_array.shape[0])]
        unmatched_dets = [int(x) for x in unmatched_dets_array]
        unmatched_tracks = [int(x) for x in unmatched_tracks_array]

        if stop: stop("greedy_assignment")

        return matches, unmatched_dets, unmatched_tracks

    def _get_detection_confidence(self, detection: Detection) -> float:
        """Extract confidence from detection object"""
        return detection.confidence

    def _debug_embeddings(self, detections):
        """Debug embedding information with SCALED embeddings comparison + plots"""
        if not detections:
            return

        logger.info(f"=== EMBEDDING DEBUG WITH SCALING COMPARISON (Frame {self._frame_count}) ===")

        det_with_emb = [
            det for det in detections if hasattr(det, "embedding") and det.embedding is not None
        ]
        tracks_with_emb = [t for t in self._tracks.values() if t.avg_embedding is not None]

        logger.info(f"Detections: {len(detections)}, With embeddings: {len(det_with_emb)}")
        logger.info(f"Active tracks: {len(self._tracks)}, With embeddings: {len(tracks_with_emb)}")

        if det_with_emb and tracks_with_emb:
            # Normalize embeddings (same as in assignment)
            det_embeddings = []
            for det in det_with_emb:
                emb = np.asarray(det.embedding, dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    det_embeddings.append(emb / norm)
                else:
                    det_embeddings.append(emb)
            det_embeddings = np.array(det_embeddings)
            track_embeddings = np.array([t.avg_embedding for t in tracks_with_emb])

            # VECTORIZED computation (same as ultra fast assignment)
            cos_similarities = det_embeddings @ track_embeddings.T
            raw_distances_matrix = (1.0 - cos_similarities) / 2.0
            raw_distances_flat = raw_distances_matrix.flatten()

            # Apply scaling
            scaled_distances_flat = self.embedding_scaler.scale_distances(raw_distances_flat)
            self.embedding_scaler.update_statistics(raw_distances_flat)
            scaled_distances_matrix = scaled_distances_flat.reshape(raw_distances_matrix.shape)

            # BEFORE/AFTER COMPARISON
            logger.info("--- BEFORE/AFTER SCALING COMPARISON ---")
            logger.info(
                f"RAW distances - Min: {np.min(raw_distances_flat):.6f}, Max: {np.max(raw_distances_flat):.6f}"
            )
            logger.info(
                f"RAW distances - Mean: {np.mean(raw_distances_flat):.6f}, Std: {np.std(raw_distances_flat):.6f}"
            )
            logger.info(
                f"RAW distances - Range: {np.max(raw_distances_flat) - np.min(raw_distances_flat):.6f}"
            )

            logger.info(
                f"SCALED distances - Min: {np.min(scaled_distances_flat):.6f}, Max: {np.max(scaled_distances_flat):.6f}"
            )
            logger.info(
                f"SCALED distances - Mean: {np.mean(scaled_distances_flat):.6f}, Std: {np.std(scaled_distances_flat):.6f}"
            )
            logger.info(
                f"SCALED distances - Range: {np.max(scaled_distances_flat) - np.min(scaled_distances_flat):.6f}"
            )

            # Range utilization analysis
            scaled_range = np.max(scaled_distances_flat) - np.min(scaled_distances_flat)
            logger.info(f"Range utilization: {scaled_range:.3f}/1.0 = {100 * scaled_range:.1f}%")

            if scaled_range < 0.3:
                logger.warning("Poor range utilization - embeddings may not be discriminative")
            elif scaled_range > 0.6:
                logger.info("Excellent range utilization!")

            # Scaler statistics
            scaler_stats = self.embedding_scaler.get_statistics()
            logger.info(
                f"Scaler ready: {scaler_stats['ready']} (samples: {scaler_stats['sample_count']})"
            )

        logger.info("=== END EMBEDDING DEBUG ===\n")

    def _update_matched_tracks(self, matches, detections):
        """Ultra-fast vectorized track update with batch embedding processing"""
        if not matches:
            return

        tracks = list(self._tracks.values())
        
        # Separate simple and OC tracks
        simple_matches = []
        oc_matches = []
        
        for det_idx, track_idx in matches:
            track = tracks[track_idx]
            if track.kalman_type == "simple":
                simple_matches.append((det_idx, track_idx))
            else:
                oc_matches.append((det_idx, track_idx))
        
        # Ultra-fast vectorized simple Kalman updates
        if simple_matches:
            self._update_simple_tracks_ultra_fast(simple_matches, detections, tracks)
        
        # Handle OC-SORT tracks (rare) - keep individual processing for complex logic
        for det_idx, track_idx in oc_matches:
            detection = detections[det_idx]
            track = tracks[track_idx]
            position = detection.position.flatten()[:2].astype(np.float32)
            track.update_with_detection(
                position, 
                getattr(detection, "embedding", None),
                getattr(detection, "bbox", None),
                self._frame_count, 
                detection.confidence
            )
    
    def _update_simple_tracks_ultra_fast(self, matches, detections, tracks):
        """Ultra-fast batch update for simple Kalman tracks with minimal overhead"""
        if not matches:
            return
            
        n_matches = len(matches)
        alpha = 0.7  # Kalman smoothing factor
        
        # Pre-allocate arrays for batch operations
        positions = np.empty((n_matches, 2), dtype=np.float32)
        embeddings_to_add = []
        bboxes_to_add = []
        tracks_to_update = []
        
        # First pass: extract all data and prepare batch operations
        for i, (det_idx, track_idx) in enumerate(matches):
            track = tracks[track_idx]
            detection = detections[det_idx]
            pos = detection.position.flatten()[:2].astype(np.float32)
            positions[i] = pos
            
            # Collect data for batch processing (only if needed)
            # Skip embedding if track already has enough embeddings
            if (hasattr(detection, "embedding") and detection.embedding is not None 
                and not track.embedding_frozen and len(track.embedding_history) < self.max_embeddings_per_track):
                embeddings_to_add.append((track, detection.embedding))
            
            if hasattr(detection, "bbox") and detection.bbox is not None:
                bboxes_to_add.append((track, detection.bbox))
                
            tracks_to_update.append((track, detection, pos))
        
        # Second pass: ultra-fast track updates with minimal per-track overhead
        for i, (track, detection, pos) in enumerate(tracks_to_update):
            # Basic state updates (fastest possible)
            track.last_detection_pos = pos
            track.last_detection_frame = self._frame_count
            track.detection_confidence = detection.confidence
            track.confidence_score = detection.confidence
            
            # Store old position for velocity calculation
            old_pos_x, old_pos_y = track.position[0], track.position[1]
            
            # Inline Kalman update (no function calls) - matches simple_kalman_update logic
            track.kalman_state[0] = alpha * pos[0] + (1 - alpha) * track.kalman_state[0]
            track.kalman_state[1] = alpha * pos[1] + (1 - alpha) * track.kalman_state[1]
            
            # Update velocity if track has hits (matches original logic)
            if track.hits > 0:
                new_velocity_x = pos[0] - old_pos_x  # dt = 1.0
                new_velocity_y = pos[1] - old_pos_y
                track.kalman_state[2] = 0.7 * new_velocity_x + 0.3 * track.kalman_state[2]
                track.kalman_state[3] = 0.7 * new_velocity_y + 0.3 * track.kalman_state[3]
            
            # Update derived positions
            track.position[0] = track.kalman_state[0]
            track.position[1] = track.kalman_state[1]
            track.velocity[0] = track.kalman_state[2]
            track.velocity[1] = track.kalman_state[3]
            track.predicted_position[0] = track.position[0] + track.velocity[0]
            track.predicted_position[1] = track.position[1] + track.velocity[1]
            
            # Update counters (fastest operations)
            track.hits += 1
            track.age += 1
            track.misses = 0
            track.lost_frames = 0
            
            # Confirmation check (minimal logic)
            if not track.confirmed and track.hits >= self.min_consecutive_detections:
                track.confirmed = True
        
        # ALWAYS vectorize embedding processing regardless of batch size
        if embeddings_to_add:
            # Extract all embeddings for vectorized processing
            embeddings_array = []
            tracks_for_embeddings = []
            
            for track, embedding in embeddings_to_add:
                if embedding is not None:
                    embeddings_array.append(np.asarray(embedding, dtype=np.float32))
                    tracks_for_embeddings.append(track)
            
            if embeddings_array:
                # ALWAYS use vectorized normalization (faster even for small batches)
                emb_matrix = np.array(embeddings_array)
                norms_sq = np.sum(emb_matrix * emb_matrix, axis=1)
                all_normalized = np.all(np.abs(norms_sq - 1.0) <= 0.01)
                
                if all_normalized:
                    # Skip normalization - use embeddings as-is
                    normalized_embs = emb_matrix
                else:
                    # Vectorized normalization
                    norms = np.sqrt(norms_sq)
                    safe_norms = np.where(norms > 0, norms, 1.0)
                    normalized_embs = emb_matrix / safe_norms[:, None]
                
                # Batch update all tracks at once
                for track, norm_emb in zip(tracks_for_embeddings, normalized_embs):
                    track.embedding_history.append(norm_emb.copy())
                    track.embedding_update_count += 1
                
                # Batch invalidate caches (faster than individual)
                for track in tracks_for_embeddings:
                    track._cache_valid = False
                    track._representative_cache_valid = False
        
        # Batch process bboxes (minimal overhead)
        for track, bbox in bboxes_to_add:
            track.bbox = bbox


    def _update_embedding_freeze_status(self):
        """OPTIMIZED vectorized density-based embedding freeze status"""
        if not self.collision_freeze_embeddings or len(self._tracks) < 2:
            return

        # REMOVED throttling - collision detection must run every frame for safety
        # This is critical for preventing ID swaps during close encounters

        tracks = list(self._tracks.values())
        n_tracks = len(tracks)
        
        # Vectorized distance computation using broadcasting
        positions = np.array([track.position for track in tracks], dtype=np.float32)
        
        # Compute all pairwise distances at once using broadcasting
        # positions.shape = (n_tracks, 2)
        # diff.shape = (n_tracks, n_tracks, 2) 
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)  # shape: (n_tracks, n_tracks)
        
        # Count nearby tracks for each track (exclude self with diagonal mask)
        nearby_mask = (distances < self.local_density_radius) & (distances > 0)  # exclude self
        nearby_counts = np.sum(nearby_mask, axis=1)  # Count neighbors for each track
        
        # Update freeze status for all tracks
        for i, track in enumerate(tracks):
            should_freeze = nearby_counts[i] >= self.embedding_freeze_density
            
            if should_freeze and not track.embedding_frozen:
                # Start freezing - store a copy of existing embedding for safety
                track.embedding_frozen = True
                if len(track.embedding_history) > 0 and track.last_safe_embedding is None:
                    track.last_safe_embedding = track.embedding_history[-1].copy()  # Use copy() to prevent corruption
                    
            elif not should_freeze and track.embedding_frozen:
                # Safe to unfreeze - restore embeddings
                track.unfreeze_embeddings(restore_last_safe=True)

    def _unfreeze_track_embedding(self, track):
        """Optimized embedding unfreezing with minimal overhead"""
        track.embedding_frozen = False
        
        # Only restore if we have a safe embedding and embedding method benefits from it
        if (track.last_safe_embedding is not None and 
            track.embedding_method == "best_match" and 
            len(track.embedding_history) > 0):
            
            # Replace last embedding with safe one for best_match method
            track.embedding_history.pop()
            track.embedding_history.append(track.last_safe_embedding)
            
            # Quick cache invalidation without expensive operations
            track._cache_valid = False
            track._representative_cache_valid = False
            track.avg_embedding = None
        
        # Clear safe embedding reference
        track.last_safe_embedding = None

    def _compute_track_uncertainties_batch(self, tracks):
        """
        ULTRA-FAST: Numba-compiled uncertainty computation for maximum performance.
        Uses compiled functions for critical path optimization.
        
        Components:
        - Time: Tracks with recent misses are more uncertain
        - Density: Tracks in crowded areas are harder to assign correctly  
        - Reliability: Tracks with poor hit/miss history are less reliable
        """
        n_tracks = len(tracks)
        if n_tracks == 0:
            return np.array([], dtype=np.float32)
        
        # Extract track data for Numba computation
        track_positions = np.array([t.position for t in tracks], dtype=np.float32)
        track_misses = np.array([t.misses for t in tracks], dtype=np.float32)
        track_ages = np.array([t.age for t in tracks], dtype=np.float32)
        
        # Use Numba-compiled function for maximum speed
        return compute_track_uncertainties_numba(
            track_positions, track_misses, track_ages, self.local_density_radius
        )

    def _handle_unmatched_tracks(self, unmatched_track_indices):
        """Handle unmatched tracks - predict position and increment miss counter"""
        tracks = list(self._tracks.values())

        for track_idx in unmatched_track_indices:
            if track_idx < len(tracks):
                track = tracks[track_idx]
                # Predict position using Kalman filter and increment misses
                track.predict_only()

    def _attempt_reid(self, detections, unmatched_det_indices):
        """ReID with proper multi-embedding support"""
        reid_matches = []

        # Filter detections with embeddings
        unmatched_dets_with_emb = []
        for det_idx in unmatched_det_indices:
            det = detections[det_idx]
            if hasattr(det, "embedding") and det.embedding is not None:
                unmatched_dets_with_emb.append((det_idx, det))

        if not unmatched_dets_with_emb:
            return reid_matches

        # Filter tracks that are missing detections but still eligible for ReID
        valid_reid_tracks = []
        for track_id, track in self._tracks.items():
            if (
                track.misses > 0
                and track.confirmed
                and len(track.embedding_history) > 0
                and track.misses <= self.max_track_age // 2  # Only try ReID in first half of track lifetime
            ):
                valid_reid_tracks.append((track_id, track))

        if not valid_reid_tracks:
            return reid_matches

        start_setup = time.perf_counter() if self.debug_timings else None

        n_dets = len(unmatched_dets_with_emb)
        n_tracks = len(valid_reid_tracks)

        # Extract positions
        det_positions = np.array(
            [det.position.flatten()[:2] for _, det in unmatched_dets_with_emb], dtype=np.float32
        )

        track_kalman_positions = np.array(
            [track.predicted_position for _, track in valid_reid_tracks], dtype=np.float32
        )

        track_last_positions = np.array(
            [track.last_detection_pos for _, track in valid_reid_tracks], dtype=np.float32
        )

        # Normalize detection embeddings
        det_embeddings = np.array(
            [
                np.asarray(det.embedding, dtype=np.float32) / np.linalg.norm(det.embedding)
                for _, det in unmatched_dets_with_emb
            ],
            dtype=np.float32,
        )

        end_setup = time.perf_counter() if self.debug_timings else None

        # Use representative embeddings for ReID - but avoid per-track computation
        start_repr = time.perf_counter() if self.debug_timings else None
        track_embeddings = []
        for _, track in valid_reid_tracks:
            # For ReID, use cached representative or fallback to most recent
            if (
                track._representative_cache_valid
                and track._cached_representative_embedding is not None
            ):
                track_embeddings.append(track._cached_representative_embedding)
            elif len(track.embedding_history) > 0:
                # Use most recent embedding instead of computing best match for each track
                track_embeddings.append(track.embedding_history[-1])
            else:
                track_embeddings.append(np.zeros(det_embeddings.shape[1], dtype=np.float32))

        track_embeddings = np.array(track_embeddings, dtype=np.float32)
        end_repr = time.perf_counter() if self.debug_timings else None

        # Use vectorized distance computation (bypass numba for small matrices)
        start_distances = time.perf_counter() if self.debug_timings else None
        if (
            n_dets * n_tracks < 50
        ):  # For small matrices, use pure NumPy (faster than numba overhead)
            cos_similarities = det_embeddings @ track_embeddings.T
            raw_distances_matrix = (1.0 - cos_similarities) / 2.0
        else:
            raw_distances_matrix = compute_embedding_distances_optimized(
                det_embeddings, track_embeddings
            )
        end_distances = time.perf_counter() if self.debug_timings else None

        start_scaling = time.perf_counter() if self.debug_timings else None
        scaled_distances_flat = self.embedding_scaler.scale_distances(
            raw_distances_matrix.flatten()
        )
        scaled_embedding_matrix = scaled_distances_flat.reshape(n_dets, n_tracks)
        end_scaling = time.perf_counter() if self.debug_timings else None

        # Use the same cost matrix method as main assignment (FIXED: ReID now respects use_probabilistic_costs)
        start_cost = time.perf_counter() if self.debug_timings else None
        if self.use_probabilistic_costs:
            # For probabilistic ReID, need additional parameters
            track_frames_since_detection = np.array(
                [self._frame_count - track.last_detection_frame for _, track in valid_reid_tracks],
                dtype=np.float32,
            )
            embedding_median = (
                np.median(scaled_embedding_matrix) if scaled_embedding_matrix.size > 0 else 0.5
            )

            cost_matrix = compute_probabilistic_cost_matrix_vectorized(
                det_positions,
                track_kalman_positions,  # Use predicted positions as track_positions for lost tracks
                track_last_positions,
                track_frames_since_detection,
                scaled_embedding_matrix,
                embedding_median,
                True,
                self.reid_max_distance,
                self.embedding_weight,
            )
        else:
            cost_matrix = compute_cost_matrix_with_multi_embeddings(
                det_positions,
                track_last_positions,
                track_kalman_positions,
                scaled_embedding_matrix,
                True,
                self.reid_max_distance,
                self.embedding_weight,
            )

        end_cost = time.perf_counter() if self.debug_timings else None

        # Apply ReID threshold filter - vectorized
        start_filter = time.perf_counter() if self.debug_timings else None
        threshold_mask = scaled_embedding_matrix > self.reid_embedding_threshold
        cost_matrix[threshold_mask] = np.inf
        end_filter = time.perf_counter() if self.debug_timings else None

        # Vectorized greedy assignment
        start_assignment = time.perf_counter() if self.debug_timings else None
        used_tracks = np.zeros(n_tracks, dtype=bool)

        # Filter out inf costs for faster assignment
        valid_mask = cost_matrix <= self.reid_max_distance

        for det_idx_in_matrix in range(n_dets):
            # Early exit if all tracks are used
            if np.all(used_tracks):
                break

            # Get costs for this detection, mask out used tracks and invalid costs
            det_costs = cost_matrix[det_idx_in_matrix].copy()
            det_costs[used_tracks] = np.inf
            det_costs[~valid_mask[det_idx_in_matrix]] = np.inf

            # Find best track
            best_track_idx = np.argmin(det_costs)
            best_cost = det_costs[best_track_idx]

            if best_cost <= self.reid_max_distance:
                used_tracks[best_track_idx] = True
                original_det_idx, detection = unmatched_dets_with_emb[det_idx_in_matrix]
                track_id, best_track = valid_reid_tracks[best_track_idx]

                if self.debug_embeddings:
                    logger.info(
                        f"ReID: Matched detection {original_det_idx} with track {track_id} (misses: {best_track.misses})"
                    )
                    logger.info(
                        f"  Used representative from {len(best_track.embedding_history)} stored embeddings"
                    )
                    logger.info(
                        f"  Embedding distance: {scaled_embedding_matrix[det_idx_in_matrix, best_track_idx]:.3f}"
                    )

                # Update track with ReID flag
                position = detection.position.flatten()[:2].astype(np.float32)
                embedding = detection.embedding
                bbox = getattr(detection, "bbox", None)
                det_conf = self._get_detection_confidence(detection)

                best_track.update_with_detection(
                    position, embedding, bbox, self._frame_count, det_conf, is_reid=True
                )
                # Track is already in self._tracks, no need to move it
                reid_matches.append(original_det_idx)

        end_assignment = time.perf_counter() if self.debug_timings else None

        return reid_matches

    def _handle_empty_frame(self):
        """Handle empty frame"""
        for track in self._tracks.values():
            track.predict_only()
        self._update_pending_detections()
        self._cleanup_tracks()
        return self._get_results()

    def _cleanup_tracks(self):
        """Remove old tracks from both active and lost, handle ReID transitions"""
        to_remove = []

        for track_id, track in self._tracks.items():
            if track.misses > self.max_track_age or (not track.confirmed and track.misses > 3):
                # Remove tracks that are too old or unconfirmed with too many misses
                to_remove.append(track_id)

        # Remove tracks that should be deleted
        for track_id in to_remove:
            if track_id in self._tracks:
                del self._tracks[track_id]

    def _get_results(self) -> List[TrackedObject]:
        """Get tracking results converted to TrackedObject instances"""
        results = []
        for track in self._tracks.values():
            # Only return confirmed tracks that are currently alive (detected this frame)
            # This ensures only truly active tracks are returned
            if track.confirmed and track.misses == 0:
                # Convert FastTrackState to TrackedObject
                tracked_obj = TrackedObject(
                    id=track.id,
                    position=track.position.copy(),
                    velocity=track.velocity.copy(),
                    confidence=track.detection_confidence,
                    age=track.age,
                    hits=track.hits,
                    time_since_update=track.misses,
                    state=1,  # Alive and confirmed
                    bbox=track.bbox.copy() if track.bbox is not None else None,
                    class_id=None,
                    predicted_position=(track.position + track.velocity).copy(),  # Add predicted position
                )
                results.append(tracked_obj)
        return results
    
    def get_recently_lost_tracks(self, max_frames_lost: int = 5) -> List[TrackedObject]:
        """
        Get tracks that were recently lost (not detected but still tracked).
        
        Args:
            max_frames_lost: Maximum number of frames since last detection to consider "recently lost"
            
        Returns:
            List of TrackedObject instances for recently lost tracks
        """
        results = []
        for track in self._tracks.values():
            # Return confirmed tracks that are recently lost (missed but within threshold)
            if track.confirmed and 0 < track.misses <= max_frames_lost:
                # Convert FastTrackState to TrackedObject
                tracked_obj = TrackedObject(
                    id=track.id,
                    position=track.position.copy(),  # Last known position
                    velocity=track.velocity.copy(),
                    confidence=track.detection_confidence * (1.0 - track.misses / max_frames_lost),  # Decay confidence
                    age=track.age,
                    hits=track.hits,
                    time_since_update=track.misses,
                    state=2,  # Recently lost state
                    bbox=track.bbox.copy() if track.bbox is not None else None,
                    class_id=None,
                    predicted_position=track.predicted_position.copy() if hasattr(track, 'predicted_position') else track.position.copy(),
                )
                results.append(tracked_obj)
        return results

    def get_all_active_tracks(self, max_frames_lost: int = 5) -> List[TrackedObject]:
        """
        Get all active tracks (both currently detected and recently lost).
        
        This is a convenience method that combines alive and recently lost tracks,
        useful for visualization when you want to show tracks that have temporarily
        lost detection but are still being tracked.
        
        Args:
            max_frames_lost: Maximum frames since last detection for recently lost tracks
            
        Returns:
            List of all active TrackedObject instances (alive + recently lost)
        """
        alive_tracks = self._get_results()
        recently_lost = self.get_recently_lost_tracks(max_frames_lost)
        return alive_tracks + recently_lost

    def reset(self):
        """Reset tracker state."""
        self._tracks.clear()
        self._pending_detections.clear()
        self._next_id = 1
        self._frame_count = 0

    def get_statistics(self) -> dict:
        """Get tracker statistics."""
        stats = {
            "frame_count": self._frame_count,
            "active_tracks": len(self._tracks),
            "pending_detections": len(self._pending_detections),
            "next_id": self._next_id,
            "assignment_strategy": self.assignment_strategy,
            "embedding_scaler_stats": self.embedding_scaler.get_statistics(),
        }


        return stats

    def get_frame_count(self):
        return self._frame_count

    def get_tracks(self):
        return self._tracks
    
    @property
    def next_id(self):
        """Get the next track ID."""
        return self._next_id
    
    @next_id.setter
    def next_id(self, value):
        """Set the next track ID (for testing purposes)."""
        self._next_id = value
    
    @property
    def frame_count(self):
        """Get the current frame count."""
        return self._frame_count
    
    @frame_count.setter
    def frame_count(self, value):
        """Set the frame count (for testing purposes)."""
        self._frame_count = value
    
    @property
    def tracks(self):
        """Get the tracks dictionary."""
        return self._tracks
    
    @tracks.setter
    def tracks(self, value):
        """Set the tracks dictionary (for testing purposes)."""
        self._tracks = value
    
    @property
    def pending_detections(self):
        """Get the pending detections list."""
        return self._pending_detections

    def get_pendings(self):
        return self._pending_detections