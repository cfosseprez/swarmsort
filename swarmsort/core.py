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
# KEEP ALL YOUR ORIGINAL FAST NUMBA FUNCTIONS - UNCHANGED
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
    """Optimized embedding distance computation for Python 3.11"""
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
def fast_gaussian_fusion(
    mu_k: np.ndarray, cov_k: np.ndarray, mu_d: np.ndarray, cov_d: np.ndarray
) -> tuple:
    """Fast 2D Gaussian fusion without scipy dependencies - ORIGINAL"""
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
    use_embeddings: bool,
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

            if use_embeddings:
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
    use_embeddings: bool,
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
            if use_embeddings:
                scaled_emb_dist = scaled_embedding_matrix[i, j]
                # Scale embedding distance to the same range as spatial distance
                embedding_cost_scaled = scaled_emb_dist * max_distance
                # Weighted average of spatial and embedding costs
                total_cost = (
                    (1.0 - embedding_weight) * spatial_cost
                    + embedding_weight * embedding_cost_scaled
                )
            else:
                total_cost = spatial_cost

            if total_cost <= max_distance:
                cost_matrix[i, j] = total_cost

    return cost_matrix


@nb.njit(fastmath=True, parallel=False, cache=True)
def compute_cost_matrix_vectorized(
    det_positions: np.ndarray,
    track_last_positions: np.ndarray,
    track_kalman_positions: np.ndarray,
    det_embeddings: np.ndarray,
    track_embeddings: np.ndarray,
    use_embeddings: bool,
    max_distance: float,
    embedding_weight: float,
) -> np.ndarray:
    """Vectorized cost matrix computation"""
    n_dets = det_positions.shape[0]
    n_tracks = track_last_positions.shape[0]
    cost_matrix = np.full((n_dets, n_tracks), np.inf, dtype=np.float32)

    for i in nb.prange(n_dets):
        for j in range(n_tracks):
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
            if use_embeddings:
                emb_dist = cosine_similarity_normalized(det_embeddings[i], track_embeddings[j])
                embedding_cost = emb_dist * embedding_weight * max_distance

            cost_matrix[i, j] = spatial_cost + embedding_cost

    return cost_matrix


@nb.njit(fastmath=True, cache=True)
def simple_kalman_update(x_pred: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Simplified Kalman update - ORIGINAL"""
    alpha = 0.7
    x_updated = np.zeros(4, dtype=np.float32)
    x_updated[0] = alpha * z[0] + (1 - alpha) * x_pred[0]
    x_updated[1] = alpha * z[1] + (1 - alpha) * x_pred[1]
    x_updated[2] = x_pred[2]
    x_updated[3] = x_pred[3]
    return x_updated


@nb.njit(fastmath=True, cache=True)
def simple_kalman_predict(x: np.ndarray) -> np.ndarray:
    """Simplified Kalman prediction - ORIGINAL"""
    x_pred = np.zeros(4, dtype=np.float32)
    x_pred[0] = x[0] + x[2]
    x_pred[1] = x[1] + x[3]
    x_pred[2] = x[2] * 0.95
    x_pred[3] = x[3] * 0.95
    return x_pred


# ============================================================================
# KEEP YOUR ORIGINAL FAST TRACK CLASS - UNCHANGED
# ============================================================================


@dataclass
class FastTrackState:
    """Enhanced track state with N-embedding history and caching"""

    id: int

    position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    predicted_position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    bbox: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))

    kalman_state: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))

    last_detection_pos: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    last_detection_frame: int = 0

    # Embedding history with configurable size
    embedding_history: deque = field(default_factory=lambda: deque(maxlen=5))
    embedding_method: Literal["average", "best_match", "weighted_average"] = "average"

    # Cache for average embedding
    _cached_avg_embedding: Optional[np.ndarray] = None
    _cache_valid: bool = False

    # NEW: Cache for multi-embedding computation
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
        """Add new embedding to history with smart cache invalidation"""
        if embedding is not None:
            embedding = np.asarray(embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                normalized_emb = embedding / norm

                # Only invalidate if embedding is significantly different
                should_invalidate = True
                if (
                    len(self.embedding_history) > 0
                    and self._cached_representative_embedding is not None
                ):
                    # Check similarity to current representative
                    similarity = np.dot(normalized_emb, self._cached_representative_embedding)
                    # Only invalidate if new embedding is quite different (< 0.85 similarity)
                    # This reduces cache invalidation frequency significantly
                    should_invalidate = similarity < 0.85

                self.embedding_history.append(normalized_emb.copy())
                self.embedding_update_count += 1

                # Smart cache invalidation
                if should_invalidate:
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
    ):
        """Updated detection update with embedding history"""
        # Same spatial update as before
        self.last_detection_pos = new_pos.astype(np.float32)
        self.last_detection_frame = current_frame
        self.detection_confidence = detection_confidence
        self.confidence_score = detection_confidence

        new_pos_f32 = new_pos.astype(np.float32)
        self.kalman_state = simple_kalman_update(self.kalman_state, new_pos_f32)

        if self.hits > 0:
            dt = 1.0
            new_velocity = (new_pos_f32 - self.position) / dt
            self.kalman_state[2] = 0.7 * new_velocity[0] + 0.3 * self.kalman_state[2]
            self.kalman_state[3] = 0.7 * new_velocity[1] + 0.3 * self.kalman_state[3]

        self.position = self.kalman_state[:2].copy()
        self.velocity = self.kalman_state[2:].copy()
        self.predicted_position = self.position + self.velocity

        # NEW: Add embedding to history
        if embedding is not None:
            self.add_embedding(embedding)

        if bbox is not None:
            self.bbox = np.asarray(bbox, dtype=np.float32)

        self.hits += 1
        self.age += 1
        self.misses = 0
        self.lost_frames = 0

        if self.hits >= 3:
            self.confirmed = True

    def predict_only(self):
        """Fast Kalman prediction step"""
        self.kalman_state = simple_kalman_predict(self.kalman_state)
        self.position = self.kalman_state[:2].copy()
        self.velocity = self.kalman_state[2:].copy()
        self.predicted_position = self.position + self.velocity
        self.age += 1
        self.misses += 1
        self.lost_frames += 1


# ============================================================================
# SWARM SORT TRACKER - PORT OF SWARMTRACKER IMPLEMENTATION
# ============================================================================


class SwarmSortTracker:
    """SwarmSort Multi-Object Tracker - Complete implementation with embedding support.

    SwarmSortTracker is the main tracking class that implements a complete multi-object
    tracking pipeline combining:
    - Kalman filtering for motion prediction
    - Hungarian algorithm for optimal assignment
    - Deep learning embeddings for appearance matching
    - Re-identification for recovering lost tracks
    - Probabilistic cost computation for robust associations

    The tracker maintains active tracks, lost tracks (for ReID), and pending detections
    (for track initialization). It supports both embedding-based and motion-only tracking
    modes with extensive configuration options.

    Attributes:
        config (SwarmSortConfig): Configuration object containing all tracker parameters
        tracks (dict): Active tracks indexed by track ID
        lost_tracks (dict): Lost tracks available for re-identification
        pending_detections (dict): Unconfirmed detections being evaluated for track creation
        next_track_id (int): ID counter for new tracks
        frame_count (int): Current frame number
        embedding_scaler (EmbeddingDistanceScaler): Adaptive embedding distance scaler

    Example:
        >>> from swarmsort import SwarmSortTracker, SwarmSortConfig, Detection
        >>> import numpy as np

        >>> # Create tracker with default configuration
        >>> tracker = SwarmSortTracker()

        >>> # Create detection
        >>> detection = Detection(
        ...     position=np.array([10.0, 20.0], dtype=np.float32),
        ...     confidence=0.9
        ... )

        >>> # Update tracker
        >>> tracked_objects = tracker.update([detection])
        >>> print(f"Tracking {len(tracked_objects)} objects")
    """

    def __init__(self, config: Optional[Union[SwarmSortConfig, dict]] = None):
        # Handle configuration
        if config is None:
            self.config = SwarmSortConfig()
        elif isinstance(config, dict):
            self.config = SwarmSortConfig.from_dict(config)
        else:
            self.config = config

        # Map config to internal parameters
        self.use_probabilistic_costs = getattr(self.config, "use_probabilistic_costs", True)

        # Core parameters
        self.max_distance = self.config.max_distance
        self.high_score_threshold = getattr(self.config, "high_score_threshold", 0.8)
        self.embedding_weight = self.config.embedding_weight
        self.max_age = self.config.max_age
        self.detection_conf_threshold = self.config.detection_conf_threshold

        # Embedding history configuration
        self.max_embeddings_per_track = self.config.max_embeddings_per_track
        self.embedding_matching_method = self.config.embedding_matching_method

        # Anti-duplicate parameters
        self.duplicate_detection_threshold = self.config.duplicate_detection_threshold

        # ReID parameters
        self.reid_enabled = self.config.reid_enabled
        self.reid_max_distance = self.config.reid_max_distance
        self.reid_embedding_threshold = self.config.reid_embedding_threshold
        self.reid_max_frames = self.config.reid_max_frames

        # INITIALIZATION PARAMETERS
        self.min_consecutive_detections = self.config.min_consecutive_detections
        self.max_detection_gap = getattr(self.config, "max_detection_gap", 2)
        self.pending_detection_distance = getattr(self.config, "pending_detection_distance", 50.0)

        # Debug options
        self.debug_embeddings = getattr(self.config, "debug_embeddings", False)
        self.plot_embeddings = getattr(self.config, "plot_embeddings", False)
        self.debug_timings = getattr(self.config, "debug_timings", False)

        # Storage
        self.tracks = {}
        self.lost_tracks = {}
        self.pending_detections = []  # INITIALIZATION LOGIC!
        self.next_id = 1
        self.frame_count = 0

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
        self._frame_det_embeddings_valid = -1  # Track which frame embeddings are valid for

        logger.info(
            f"Initialized SwarmSortTracker with initialization logic and fixed embedding scaling"
        )

    def _create_new_track(self, track_id: int, position: np.ndarray) -> FastTrackState:
        """Create new track with embedding configuration"""
        track = FastTrackState(id=track_id, position=position)
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

            logger.info("Numba functions compiled successfully")
        except Exception as e:
            logger.warning(f"Numba compilation failed: {e}")

    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """Main update with optional timing per call"""
        self.frame_count += 1

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

        if not valid_detections:
            return self._handle_empty_frame()

        if self.debug_embeddings and self.frame_count % 10 == 0:
            start("debug_embeddings")
            self._debug_embeddings(valid_detections)
            stop("debug_embeddings")

        start("predict_tracks")
        for track in self.tracks.values():
            track.predict_only()
        stop("predict_tracks")

        start("assignment")
        if self.use_probabilistic_costs:
            matches, unmatched_dets, unmatched_tracks = self._fast_assignment_probabilistic(
                valid_detections, timer, start, stop
            )
        else:
            matches, unmatched_dets, unmatched_tracks = self._fast_assignment(
                valid_detections, timer, start, stop
            )
        stop("assignment")

        start("update_matched")
        self._update_matched_tracks(matches, valid_detections)
        stop("update_matched")

        start("handle_unmatched_tracks")
        self._handle_unmatched_tracks(unmatched_tracks)
        stop("handle_unmatched_tracks")

        if self.reid_enabled and self.lost_tracks:
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
            print(f"[Frame {self.frame_count}] Timings:", formatted_timings)

        return result

    def _handle_unmatched_detections(self, unmatched_det_indices, detections):
        """FAST handle unmatched detections with vectorized pending matching"""
        if not unmatched_det_indices:
            return

        # Filter high-confidence detections
        high_conf_detections = []
        for det_idx in unmatched_det_indices:
            detection = detections[det_idx]
            det_conf = self._get_detection_confidence(detection)
            if det_conf >= self.high_score_threshold:
                high_conf_detections.append((det_idx, detection))

        if not high_conf_detections:
            return

        # Extract positions of high-confidence detections
        det_positions = np.array(
            [detection.position.flatten()[:2] for _, detection in high_conf_detections],
            dtype=np.float32,
        )

        # Extract pending positions for vectorized comparison
        if self.pending_detections:
            pending_positions = np.array(
                [pending.average_position for pending in self.pending_detections], dtype=np.float32
            )

            # Vectorized distance computation: (n_dets, n_pending)
            # Broadcasting: det_positions[:, None, :] - pending_positions[None, :, :]
            diff = det_positions[:, None, :] - pending_positions[None, :, :]
            distances = np.sqrt(np.sum(diff * diff, axis=2))  # Shape: (n_dets, n_pending)

            # Find closest pending for each detection
            min_distances = np.min(distances, axis=1)
            closest_pending_indices = np.argmin(distances, axis=1)

            # Process each detection
            matched_pending_indices = set()
            for i, (det_idx, detection) in enumerate(high_conf_detections):
                if min_distances[i] < self.pending_detection_distance:
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
                            self.pending_detections[pending_idx], position, embedding, bbox
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
                        first_seen_frame=self.frame_count,
                        last_seen_frame=self.frame_count,
                        average_position=position.copy(),
                    )
                    self.pending_detections.append(new_pending)
        else:
            # No existing pending detections - create new ones for all high-confidence detections
            for det_idx, detection in high_conf_detections:
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
                    first_seen_frame=self.frame_count,
                    last_seen_frame=self.frame_count,
                    average_position=position.copy(),
                )
                self.pending_detections.append(new_pending)

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
        frame_gap = self.frame_count - pending.last_seen_frame
        pending.last_seen_frame = self.frame_count
        pending.total_detections += 1

        if frame_gap == 1:  # Consecutive frame
            pending.consecutive_frames += 1
        else:  # There was a gap
            pending.consecutive_frames = 1

    def _update_pending_detections(self):
        """Update pending detections and promote to tracks"""
        detections_to_remove = []

        for i, pending in enumerate(self.pending_detections):
            frame_gap = self.frame_count - pending.last_seen_frame

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
            del self.pending_detections[i]

    def _create_track_from_pending(self, pending: PendingDetection):
        """Create track from pending detection with proper embedding setup"""
        new_track = FastTrackState(id=self.next_id, position=pending.average_position.copy())

        # IMPORTANT: Set embedding parameters
        new_track.set_embedding_params(
            max_embeddings=self.max_embeddings_per_track, method=self.embedding_matching_method
        )

        # Initialize track with pending embedding if available
        if pending.embedding is not None:
            new_track.add_embedding(pending.embedding)

        # Update with detection
        new_track.update_with_detection(
            pending.average_position,
            pending.embedding,  # This will add to history again, but that's ok
            pending.bbox,
            self.frame_count,
            pending.confidence,
        )

        new_track.hits = min(pending.total_detections, 3)
        new_track.age = pending.consecutive_frames

        self.tracks[self.next_id] = new_track

        if self.debug_embeddings:
            logger.info(f"Created track {self.next_id} from pending detection")
            logger.info(f"  Embedding method: {self.embedding_matching_method}")
            logger.info(f"  Initial embeddings: {len(new_track.embedding_history)}")

        self.next_id += 1

    def _fast_assignment(self, detections, timer=None, start=None, stop=None):
        """OPTIMIZED assignment with cached representative embeddings (same logic, faster)"""
        n_dets = len(detections)
        n_tracks = len(self.tracks)

        if n_tracks == 0:
            return [], list(range(n_dets)), []

        tracks = list(self.tracks.values())

        # Extract positions
        det_positions = np.array(
            [det.position.flatten()[:2] for det in detections], dtype=np.float32
        )
        track_last_positions = np.array([t.last_detection_pos for t in tracks], dtype=np.float32)
        track_kalman_positions = np.array([t.position for t in tracks], dtype=np.float32)

        # Check embeddings
        use_embeddings = all(
            hasattr(det, "embedding") and det.embedding is not None for det in detections
        ) and all(len(t.embedding_history) > 0 for t in tracks)

        scaled_embedding_matrix = np.zeros((n_dets, n_tracks), dtype=np.float32)

        if use_embeddings:
            if start:
                start("embedding_computation")

            # EARLY SPATIAL FILTER: Only compute embeddings for spatially feasible pairs
            # Use squared distances to avoid sqrt computation
            spatial_distances_sq = np.sum(
                (det_positions[:, None, :] - track_last_positions[None, :, :]) ** 2, axis=2
            )
            max_dist_sq = (self.max_distance * 1.2) ** 2  # Slightly larger threshold, squared
            spatial_mask = spatial_distances_sq <= max_dist_sq
            
            # Always compute embeddings if they are available. A previous optimization
            # here was too aggressive and disabled embeddings in sparse scenes,
            # leading to mismatches that corrupt track state and break ReID.

            # OPTIMIZED: Reuse embedding arrays when possible
            try:
                first_emb = detections[0].embedding
                emb_dim = (
                    len(first_emb) if hasattr(first_emb, "__len__") else first_emb.shape[0]
                )
            except:
                emb_dim = 128  # fallback dimension

            if (
                self._reusable_det_embeddings is None
                or self._reusable_det_embeddings.shape[0] < n_dets
                or n_dets > self._max_dets_seen
            ):
                self._reusable_det_embeddings = np.empty(
                    (max(n_dets, self._max_dets_seen + 10), emb_dim), dtype=np.float32
                )
                self._max_dets_seen = max(n_dets, self._max_dets_seen)

            # OPTIMIZED: Normalize only once per frame, reuse across all assignment methods
            if (
                not hasattr(self, "_frame_det_embeddings_valid")
                or self._frame_det_embeddings_valid != self.frame_count
            ):
                for i, det in enumerate(detections):
                    emb = np.asarray(det.embedding, dtype=np.float32)
                    norm = np.linalg.norm(emb)
                    self._reusable_det_embeddings[i] = emb / norm if norm > 0 else emb
                self._frame_det_embeddings_valid = self.frame_count

            det_embeddings = self._reusable_det_embeddings[:n_dets]

            # OPTIMIZED: Get representative embeddings with caching
            if (
                self._reusable_track_embeddings is None
                or self._reusable_track_embeddings.shape[0] < n_tracks
                or n_tracks > self._max_tracks_seen
            ):
                emb_dim = (
                    det_embeddings.shape[1] if det_embeddings.size > 0 else 128
                )  # fallback
                self._reusable_track_embeddings = np.empty(
                    (max(n_tracks, self._max_tracks_seen + 10), emb_dim), dtype=np.float32
                )
                self._max_tracks_seen = max(n_tracks, self._max_tracks_seen)

            # OPTIMIZED: Batch check cache validity
            for i, track in enumerate(tracks):
                if not track._representative_cache_valid:
                    repr_emb = track.get_representative_embedding_for_assignment(det_embeddings)
                    if repr_emb is not None:
                        self._reusable_track_embeddings[i] = repr_emb
                    else:
                        self._reusable_track_embeddings[i] = np.zeros(emb_dim, dtype=np.float32)
                else:
                    # Use cached embedding directly
                    cached_emb = track._cached_representative_embedding
                    if cached_emb is not None:
                        self._reusable_track_embeddings[i] = cached_emb
                    else:
                        self._reusable_track_embeddings[i] = np.zeros(emb_dim, dtype=np.float32)

            track_embeddings = self._reusable_track_embeddings[:n_tracks]

            # Use faster distance computation
            raw_distances_matrix = compute_embedding_distances_optimized(
                det_embeddings, track_embeddings
            )

            # Apply scaling
            raw_distances_flat = raw_distances_matrix.flatten()
            scaled_distances_flat = self.embedding_scaler.scale_distances(raw_distances_flat)
            self.embedding_scaler.update_statistics(raw_distances_flat)
            scaled_embedding_matrix = scaled_distances_flat.reshape(n_dets, n_tracks)

            if stop:
                stop("embedding_computation")

            # Only do debug output if explicitly requested (saves time)
            # if self.debug_embeddings and self.frame_count % 50 == 0:  # Less frequent
            #     logger.info(f"OPTIMIZED assignment: method={self.embedding_matching_method}")
            #     logger.info(f"Using cached representative embeddings")

        # Compute cost matrix
        cost_matrix = compute_cost_matrix_with_multi_embeddings(
            det_positions,
            track_last_positions,
            track_kalman_positions,
            scaled_embedding_matrix,
            use_embeddings,
            self.max_distance,
            self.embedding_weight,
        )

        # Hungarian assignment (same as before)
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

    def _fast_assignment_probabilistic(self, detections, timer=None, start=None, stop=None):
        """FIXED probabilistic assignment with proper multi-embedding support"""
        n_dets = len(detections)
        n_tracks = len(self.tracks)

        if n_tracks == 0:
            return [], list(range(n_dets)), []

        tracks = list(self.tracks.values())

        # Extract positions
        det_positions = np.array(
            [det.position.flatten()[:2] for det in detections], dtype=np.float32
        )
        track_positions = np.array([t.position for t in tracks], dtype=np.float32)
        track_last_positions = np.array([t.last_detection_pos for t in tracks], dtype=np.float32)

        # Check embeddings
        use_embeddings = all(
            hasattr(det, "embedding") and det.embedding is not None for det in detections
        ) and all(len(t.embedding_history) > 0 for t in tracks)

        scaled_embedding_matrix = np.zeros((n_dets, n_tracks), dtype=np.float32)

        if use_embeddings:
            # Same as regular assignment - use cached representatives
            det_embeddings = np.array(
                [
                    np.asarray(det.embedding, dtype=np.float32) / np.linalg.norm(det.embedding)
                    for det in detections
                ],
                dtype=np.float32,
            )

            # Get representative embeddings with caching (reuse array from regular assignment)
            if (
                self._reusable_track_embeddings is None
                or self._reusable_track_embeddings.shape[0] < n_tracks
                or n_tracks > self._max_tracks_seen
            ):
                emb_dim = det_embeddings.shape[1] if det_embeddings.size > 0 else 128  # fallback
                self._reusable_track_embeddings = np.empty(
                    (max(n_tracks, self._max_tracks_seen + 10), emb_dim), dtype=np.float32
                )
                self._max_tracks_seen = max(n_tracks, self._max_tracks_seen)

            for i, track in enumerate(tracks):
                # Only recompute if cache is invalid
                if not track._representative_cache_valid:
                    repr_emb = track.get_representative_embedding_for_assignment(det_embeddings)
                    if repr_emb is not None:
                        self._reusable_track_embeddings[i] = repr_emb
                    else:
                        self._reusable_track_embeddings[i] = np.zeros(emb_dim, dtype=np.float32)
                else:
                    # Use cached embedding directly
                    cached_emb = track._cached_representative_embedding
                    if cached_emb is not None:
                        self._reusable_track_embeddings[i] = cached_emb
                    else:
                        self._reusable_track_embeddings[i] = np.zeros(emb_dim, dtype=np.float32)

            track_embeddings = self._reusable_track_embeddings[:n_tracks]

            # Use faster distance computation
            raw_distances_matrix = compute_embedding_distances_optimized(
                det_embeddings, track_embeddings
            )

            raw_distances_flat = raw_distances_matrix.flatten()
            scaled_distances_flat = self.embedding_scaler.scale_distances(raw_distances_flat)
            self.embedding_scaler.update_statistics(raw_distances_flat)
            scaled_embedding_matrix = scaled_distances_flat.reshape(n_dets, n_tracks)

        embedding_median = np.median(scaled_embedding_matrix)
        track_frames_since_detection = np.array(
            [self.frame_count - track.last_detection_frame for track in tracks], dtype=np.float32
        )

        cost_matrix = compute_probabilistic_cost_matrix_vectorized(
            det_positions,
            track_positions,
            track_last_positions,
            track_frames_since_detection,
            scaled_embedding_matrix,
            embedding_median,
            use_embeddings,
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

    def _get_detection_confidence(self, detection: Detection) -> float:
        """Extract confidence from detection object"""
        return detection.confidence

    def _debug_embeddings(self, detections):
        """Debug embedding information with SCALED embeddings comparison + plots"""
        if not detections:
            return

        logger.info(f"=== EMBEDDING DEBUG WITH SCALING COMPARISON (Frame {self.frame_count}) ===")

        det_with_emb = [
            det for det in detections if hasattr(det, "embedding") and det.embedding is not None
        ]
        tracks_with_emb = [t for t in self.tracks.values() if t.avg_embedding is not None]

        logger.info(f"Detections: {len(detections)}, With embeddings: {len(det_with_emb)}")
        logger.info(f"Active tracks: {len(self.tracks)}, With embeddings: {len(tracks_with_emb)}")

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
        """update matched tracks with batched operations"""
        if not matches:
            return

        tracks = list(self.tracks.values())

        # Batch position and confidence extraction
        positions = []
        embeddings = []
        bboxes = []
        confidences = []
        track_objects = []

        for det_idx, track_idx in matches:
            detection = detections[det_idx]
            track = tracks[track_idx]

            positions.append(detection.position.flatten()[:2].astype(np.float32))
            embeddings.append(getattr(detection, "embedding", None))
            bboxes.append(getattr(detection, "bbox", None))
            confidences.append(self._get_detection_confidence(detection))
            track_objects.append(track)

        # Batch update all tracks (reduces individual cache invalidations)
        for i, track in enumerate(track_objects):
            track.update_with_detection(
                positions[i], embeddings[i], bboxes[i], self.frame_count, confidences[i]
            )

    def _handle_unmatched_tracks(self, unmatched_track_indices):
        """Handle unmatched tracks - move to lost for potential ReID"""
        tracks = list(self.tracks.values())
        to_remove = []

        for track_idx in unmatched_track_indices:
            if track_idx < len(tracks):
                track = tracks[track_idx]

                if track.confirmed and track.misses <= 3 and self.reid_enabled:
                    if self.debug_embeddings:
                        logger.info(
                            f"Moving track {track.id} to lost tracks for ReID (misses: {track.misses})"
                        )
                    self.lost_tracks[track.id] = track
                    to_remove.append(track.id)
                elif track.misses > 5:
                    to_remove.append(track.id)

        for track_id in to_remove:
            if track_id in self.tracks:
                del self.tracks[track_id]

    def _attempt_reid(self, detections, unmatched_det_indices):
        """FIXED ReID with proper multi-embedding support"""
        reid_matches = []

        if not self.lost_tracks or not unmatched_det_indices:
            return reid_matches

        start_setup = time.perf_counter() if self.debug_timings else None

        # Filter detections with embeddings
        unmatched_dets_with_emb = []
        for det_idx in unmatched_det_indices:
            det = detections[det_idx]
            if hasattr(det, "embedding") and det.embedding is not None:
                unmatched_dets_with_emb.append((det_idx, det))

        if not unmatched_dets_with_emb:
            return reid_matches

        # Filter valid lost tracks
        valid_lost_tracks = []
        for track_id, lost_track in self.lost_tracks.items():
            if (
                lost_track.lost_frames <= self.reid_max_frames
                and len(lost_track.embedding_history) > 0
            ):
                valid_lost_tracks.append((track_id, lost_track))

        if not valid_lost_tracks:
            return reid_matches

        n_dets = len(unmatched_dets_with_emb)
        n_tracks = len(valid_lost_tracks)

        # Extract positions
        det_positions = np.array(
            [det.position.flatten()[:2] for _, det in unmatched_dets_with_emb], dtype=np.float32
        )

        track_kalman_positions = np.array(
            [track.predicted_position for _, track in valid_lost_tracks], dtype=np.float32
        )

        track_last_positions = np.array(
            [track.last_detection_pos for _, track in valid_lost_tracks], dtype=np.float32
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
        for _, track in valid_lost_tracks:
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
                [self.frame_count - track.last_detection_frame for _, track in valid_lost_tracks], 
                dtype=np.float32
            )
            embedding_median = np.median(scaled_embedding_matrix) if scaled_embedding_matrix.size > 0 else 0.5
            
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
                track_id, best_track = valid_lost_tracks[best_track_idx]

                if self.debug_embeddings:
                    logger.info(
                        f"ReID: Matched detection {original_det_idx} with lost track {track_id}"
                    )
                    logger.info(
                        f"  Used representative from {len(best_track.embedding_history)} stored embeddings"
                    )
                    logger.info(
                        f"  Embedding distance: {scaled_embedding_matrix[det_idx_in_matrix, best_track_idx]:.3f}"
                    )

                # Update track
                position = detection.position.flatten()[:2].astype(np.float32)
                embedding = detection.embedding
                bbox = getattr(detection, "bbox", None)
                det_conf = self._get_detection_confidence(detection)

                best_track.update_with_detection(
                    position, embedding, bbox, self.frame_count, det_conf
                )
                self.tracks[track_id] = best_track
                del self.lost_tracks[track_id]
                reid_matches.append(original_det_idx)

        end_assignment = time.perf_counter() if self.debug_timings else None

        # Log detailed ReID timings
        if self.debug_timings:
            # The main update loop already times the entire ReID step.
            # Uncomment the following lines for a detailed breakdown of ReID performance.
            # self.timings["reid_setup"] = f"{(end_setup - start_setup) * 1000:.2f} ms"
            # self.timings["reid_repr_emb"] = f"{(end_repr - start_repr) * 1000:.2f} ms"
            # self.timings["reid_distances"] = f"{(end_distances - start_distances) * 1000:.2f} ms"
            # self.timings["reid_scaling"] = f"{(end_scaling - start_scaling) * 1000:.2f} ms"
            # self.timings["reid_cost_matrix"] = f"{(end_cost - start_cost) * 1000:.2f} ms"
            # self.timings["reid_filter"] = f"{(end_filter - start_filter) * 1000:.2f} ms"
            # self.timings["reid_assignment"] = f"{(end_assignment - start_assignment) * 1000:.2f} ms"
            pass

        return reid_matches

    def _handle_empty_frame(self):
        """Handle empty frame"""
        for track in self.tracks.values():
            track.predict_only()
        self._update_pending_detections()
        self._cleanup_tracks()
        return self._get_results()

    def _cleanup_tracks(self):
        """Remove old tracks from both active and lost, handle ReID transitions"""
        to_remove = []
        to_move_to_lost = []

        for track_id, track in self.tracks.items():
            if (
                track.confirmed
                and track.misses >= 3
                and track.misses <= self.max_age
                and self.reid_enabled
            ):
                # Move confirmed tracks to lost for potential ReID
                to_move_to_lost.append(track_id)
            elif track.misses > self.max_age or (not track.confirmed and track.misses > 3):
                # Remove tracks that are too old or unconfirmed with too many misses
                to_remove.append(track_id)

        # Move tracks to lost for ReID
        for track_id in to_move_to_lost:
            if track_id in self.tracks:
                track = self.tracks[track_id]
                if self.debug_embeddings:
                    logger.info(
                        f"Moving track {track_id} to lost tracks for ReID (misses: {track.misses})"
                    )
                self.lost_tracks[track_id] = track
                del self.tracks[track_id]

        # Remove tracks that should be deleted
        for track_id in to_remove:
            if track_id in self.tracks:
                del self.tracks[track_id]

        to_remove_lost = [
            track_id
            for track_id, track in self.lost_tracks.items()
            if track.lost_frames > self.reid_max_frames
        ]

        for track_id in to_remove_lost:
            del self.lost_tracks[track_id]

    def _get_results(self) -> List[TrackedObject]:
        """Get tracking results converted to TrackedObject instances"""
        results = []
        for track in self.tracks.values():
            if track.confirmed:
                # Convert FastTrackState to TrackedObject
                tracked_obj = TrackedObject(
                    id=track.id,
                    position=track.position.copy(),
                    velocity=track.velocity.copy(),
                    confidence=track.detection_confidence,
                    age=track.age,
                    hits=track.hits,
                    time_since_update=track.misses,
                    state=1 if track.confirmed else 0,
                    bbox=track.bbox.copy() if track.bbox is not None else None,
                    class_id=None,
                )
                results.append(tracked_obj)
        return results

    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.lost_tracks.clear()
        self.pending_detections.clear()
        self.next_id = 1
        self.frame_count = 0

    def get_statistics(self) -> dict:
        """Get tracker statistics."""
        return {
            "frame_count": self.frame_count,
            "active_tracks": len(self.tracks),
            "lost_tracks": len(self.lost_tracks),
            "pending_detections": len(self.pending_detections),
            "next_id": self.next_id,
            "embedding_scaler_stats": self.embedding_scaler.get_statistics(),
        }
