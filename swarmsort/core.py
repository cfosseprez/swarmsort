"""
SwarmSort Core Implementation - Standalone Multi-Object Tracker

This module contains the main SwarmSortTracker class with optimized algorithms
for real-time multi-object tracking using deep learning embeddings.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union, Literal
from dataclasses import dataclass, field
from collections import deque
import time
from pathlib import Path

# External dependencies
import numba as nb
from scipy.optimize import linear_sum_assignment
from loguru import logger

# Internal imports
from .data_classes import Detection, TrackedObject
from .config import SwarmSortConfig
from .embedding_scaler import EmbeddingDistanceScaler
from .embeddings import (
    get_embedding_extractor,
    list_available_embeddings,
    compute_embedding_distance,
    is_gpu_available
)


# ============================================================================
# PERFORMANCE OPTIMIZATIONS - Numba JIT Functions
# ============================================================================

@nb.njit(fastmath=True, cache=True)
def cosine_similarity_normalized(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Fast cosine similarity normalized to [0, 1] distance."""
    norm1 = np.sqrt(np.sum(emb1 * emb1))
    norm2 = np.sqrt(np.sum(emb2 * emb2))
    
    if norm1 == 0 or norm2 == 0:
        return 1.0
    
    cos_sim = np.sum(emb1 * emb2) / (norm1 * norm2)
    return (1.0 - cos_sim) / 2.0


@nb.njit(fastmath=True, cache=True)
def fast_mahalanobis_distance(diff: np.ndarray, cov_inv: np.ndarray) -> float:
    """Fast 2D Mahalanobis distance calculation."""
    return np.sqrt(diff[0] * (cov_inv[0, 0] * diff[0] + cov_inv[0, 1] * diff[1]) +
                   diff[1] * (cov_inv[1, 0] * diff[0] + cov_inv[1, 1] * diff[1]))


@nb.njit(fastmath=True, cache=True, parallel=False)
def compute_embedding_distances_multi_history(
    det_embeddings, 
    track_embeddings_list, 
    track_embedding_counts,
    method='best_match'
):
    """
    Compute embedding distances considering multiple embeddings per track.
    
    Args:
        det_embeddings: (n_dets, emb_dim) normalized detection embeddings
        track_embeddings_list: flattened array of all track embeddings
        track_embedding_counts: array with number of embeddings per track
        method: 'best_match', 'average', or 'weighted_average'
    
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
            for i in range(n_dets):
                distances[i, j] = 1.0
            track_start_idx += n_embs
            continue
        
        for i in range(n_dets):
            det_emb = det_embeddings[i]
            
            if method == 'best_match':
                min_dist = 1.0
                for k in range(n_embs):
                    track_emb = track_embeddings_list[track_start_idx + k]
                    dot_product = 0.0
                    for dim in range(emb_dim):
                        dot_product += det_emb[dim] * track_emb[dim]
                    dist = (1.0 - dot_product) / 2.0
                    if dist < min_dist:
                        min_dist = dist
                distances[i, j] = min_dist
                
            elif method == 'average':
                avg_dist = 0.0
                for k in range(n_embs):
                    track_emb = track_embeddings_list[track_start_idx + k]
                    dot_product = 0.0
                    for dim in range(emb_dim):
                        dot_product += det_emb[dim] * track_emb[dim]
                    avg_dist += (1.0 - dot_product) / 2.0
                distances[i, j] = avg_dist / n_embs
                
            elif method == 'weighted_average':
                weighted_dist = 0.0
                weight_sum = 0.0
                for k in range(n_embs):
                    track_emb = track_embeddings_list[track_start_idx + k]
                    weight = k + 1  # More recent = higher weight
                    dot_product = 0.0
                    for dim in range(emb_dim):
                        dot_product += det_emb[dim] * track_emb[dim]
                    weighted_dist += weight * (1.0 - dot_product) / 2.0
                    weight_sum += weight
                distances[i, j] = weighted_dist / weight_sum
        
        track_start_idx += n_embs
    
    return distances


@nb.njit(fastmath=True, cache=True, parallel=True)
def compute_cost_matrix_vectorized(positions_det, positions_track, max_distance):
    """Vectorized cost matrix computation using Euclidean distance."""
    n_det = positions_det.shape[0]
    n_track = positions_track.shape[0]
    cost_matrix = np.full((n_det, n_track), max_distance * 2, dtype=np.float32)
    
    for i in nb.prange(n_det):
        for j in range(n_track):
            dx = positions_det[i, 0] - positions_track[j, 0]
            dy = positions_det[i, 1] - positions_track[j, 1]
            distance = np.sqrt(dx * dx + dy * dy)
            cost_matrix[i, j] = distance
    
    return cost_matrix


# ============================================================================
# PENDING DETECTION SYSTEM FOR TRACK INITIALIZATION
# ============================================================================

@dataclass
class PendingDetection:
    """Represents a detection waiting to become a track."""
    position: np.ndarray
    embedding: Optional[np.ndarray] = None
    bbox: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    confidence: float = 1.0
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    consecutive_frames: int = 1
    total_detections: int = 1
    average_position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    
    def __post_init__(self):
        if self.average_position.sum() == 0:
            self.average_position = self.position.copy()
    
    def __eq__(self, other):
        """Custom equality to handle numpy arrays properly."""
        if not isinstance(other, PendingDetection):
            return False
        
        return (
            np.allclose(self.position, other.position) and
            self.confidence == other.confidence and
            self.first_seen_frame == other.first_seen_frame and
            self.last_seen_frame == other.last_seen_frame and
            self.consecutive_frames == other.consecutive_frames and
            self.total_detections == other.total_detections and
            np.allclose(self.average_position, other.average_position) and
            np.array_equal(self.bbox, other.bbox) and
            (self.embedding is None and other.embedding is None or
             self.embedding is not None and other.embedding is not None and 
             np.array_equal(self.embedding, other.embedding))
        )
    
    def __hash__(self):
        """Custom hash method to work with sets and dicts."""
        return hash((
            tuple(self.position),
            self.confidence,
            self.first_seen_frame,
            self.last_seen_frame,
            self.consecutive_frames,
            self.total_detections
        ))


# ============================================================================
# TRACK STATE MANAGEMENT
# ============================================================================

@dataclass
class TrackState:
    """Internal track state with Kalman filter and embedding history."""
    id: int
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    bbox: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    
    # Kalman filter state
    state: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))  # [x, y, vx, vy]
    covariance: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32) * 10.0)
    
    # Track statistics
    confidence: float = 0.0
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    hit_streak: int = 0
    class_id: Optional[int] = None
    
    # Embedding history
    embeddings: deque = field(default_factory=deque)
    
    def __post_init__(self):
        # Initialize Kalman state
        self.state[:2] = self.position
        self.state[2:] = self.velocity
    
    def __eq__(self, other):
        """Custom equality to handle numpy arrays properly."""
        if not isinstance(other, TrackState):
            return False
        
        return (
            self.id == other.id and
            np.allclose(self.position, other.position) and
            np.allclose(self.velocity, other.velocity) and
            np.array_equal(self.bbox, other.bbox) and
            np.allclose(self.state, other.state) and
            np.allclose(self.covariance, other.covariance) and
            self.confidence == other.confidence and
            self.age == other.age and
            self.hits == other.hits and
            self.time_since_update == other.time_since_update and
            self.hit_streak == other.hit_streak and
            self.class_id == other.class_id and
            len(self.embeddings) == len(other.embeddings) and
            all(np.array_equal(e1, e2) for e1, e2 in zip(self.embeddings, other.embeddings))
        )
    
    def __hash__(self):
        """Custom hash method to work with sets and dicts."""
        return hash((
            self.id,
            tuple(self.position),
            self.confidence,
            self.age,
            self.hits
        ))


# ============================================================================
# MAIN SWARM SORT TRACKER CLASS
# ============================================================================

class SwarmSortTracker:
    """
    SwarmSort Multi-Object Tracker with Deep Learning Embeddings.
    
    A high-performance tracker combining:
    - Kalman filtering for motion prediction
    - Deep learning embeddings for appearance matching
    - Probabilistic data fusion
    - Advanced initialization and re-identification
    """
    
    def __init__(self, config: Optional[Union[SwarmSortConfig, dict]] = None):
        """
        Initialize SwarmSort tracker.
        
        Args:
            config: SwarmSortConfig instance or dict with configuration parameters
        """
        # Handle configuration
        if config is None:
            self.config = SwarmSortConfig()
        elif isinstance(config, dict):
            self.config = SwarmSortConfig.from_dict(config)
        else:
            self.config = config
        
        self.config.validate()
        
        # Core tracking state
        self.tracks: Dict[int, TrackState] = {}
        self.lost_tracks: Dict[int, TrackState] = {}
        self.pending_detections: List[PendingDetection] = []
        self.next_id = 1
        self.frame_count = 0
        
        # Embedding distance scaler
        self.embedding_scaler = EmbeddingDistanceScaler(
            method=self.config.embedding_scaling_method,
            update_rate=self.config.embedding_scaling_update_rate,
            min_samples=self.config.embedding_scaling_min_samples
        )
        
        # Motion model parameters
        self.dt = 1.0  # Time step
        self.process_noise = 0.1
        self.measurement_noise = 1.0
        
        # Initialize motion matrices
        self._setup_motion_model()
        
        # Performance tracking
        self.timing_stats = {}
        self.debug_info = {}
    
    def _setup_motion_model(self):
        """Setup Kalman filter motion model matrices."""
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Observation matrix (observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        self.Q = np.array([
            [self.dt**4/4, 0, self.dt**3/2, 0],
            [0, self.dt**4/4, 0, self.dt**3/2],
            [self.dt**3/2, 0, self.dt**2, 0],
            [0, self.dt**3/2, 0, self.dt**2]
        ], dtype=np.float32) * self.process_noise
        
        # Measurement noise covariance
        self.R = np.eye(2, dtype=np.float32) * self.measurement_noise
    
    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            List of TrackedObject instances representing current tracks
        """
        start_time = time.perf_counter()
        self.frame_count += 1
        
        # Filter detections by confidence
        detections = [det for det in detections 
                     if det.confidence >= self.config.detection_conf_threshold]
        
        # Remove duplicate detections
        detections = self._remove_duplicate_detections(detections)
        
        # Predict tracks
        self._predict_tracks()
        
        # Associate detections with tracks
        matched_tracks, matched_detections, unmatched_detections, unmatched_tracks = \
            self._associate_detections_to_tracks(detections)
        
        # Update matched tracks
        self._update_matched_tracks(matched_tracks, matched_detections, detections)
        
        # Handle unmatched tracks
        self._handle_unmatched_tracks(unmatched_tracks)
        
        # Process unmatched detections
        self._process_unmatched_detections(unmatched_detections, detections)
        
        # Re-identification
        if self.config.reid_enabled:
            self._attempt_reidentification(detections)
        
        # Get current tracked objects
        tracked_objects = self._get_tracked_objects()
        
        # Update timing
        self.timing_stats['total'] = time.perf_counter() - start_time
        
        if self.config.debug_timings:
            logger.info(f"Frame {self.frame_count}: {len(tracked_objects)} tracks, "
                       f"{self.timing_stats.get('total', 0)*1000:.1f}ms")
        
        return tracked_objects
    
    def _remove_duplicate_detections(self, detections: List[Detection]) -> List[Detection]:
        """Remove detections that are too close to each other."""
        if len(detections) <= 1:
            return detections
        
        unique_detections = []
        positions = np.array([det.position for det in detections])
        
        for i, detection in enumerate(detections):
            is_unique = True
            detection_pos = np.asarray(detection.position)
            for j in range(len(unique_detections)):
                other_pos = np.asarray(unique_detections[j].position)
                distance = np.linalg.norm(detection_pos - other_pos)
                if distance < self.config.duplicate_detection_threshold:
                    # Keep the one with higher confidence
                    if detection.confidence <= unique_detections[j].confidence:
                        is_unique = False
                        break
                    else:
                        # Replace the existing one
                        unique_detections[j] = detection
                        is_unique = False
                        break
            
            if is_unique:
                unique_detections.append(detection)
        
        return unique_detections
    
    def _predict_tracks(self):
        """Predict track states using Kalman filter."""
        for track in self.tracks.values():
            # Kalman prediction
            track.state = self.F @ track.state
            track.covariance = self.F @ track.covariance @ self.F.T + self.Q
            
            # Update position and velocity from state
            track.position = track.state[:2]
            track.velocity = track.state[2:]
            
            # Update track statistics
            track.time_since_update += 1
            track.age += 1
    
    def _associate_detections_to_tracks(self, detections: List[Detection]) -> Tuple:
        """
        Associate detections with existing tracks using Hungarian algorithm.
        
        Returns:
            Tuple of (matched_tracks, matched_detections, unmatched_detections, unmatched_tracks)
        """
        if not detections or not self.tracks:
            return [], [], list(range(len(detections))), list(self.tracks.keys())
        
        # Build cost matrix
        cost_matrix = self._build_cost_matrix(detections, list(self.tracks.values()))
        
        # Hungarian assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Extract matches and unmatched
        matched_tracks = []
        matched_detections = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())
        
        for row_idx, col_idx in zip(row_indices, col_indices):
            cost = cost_matrix[row_idx, col_idx]
            if cost < self.config.max_distance:
                track_id = list(self.tracks.keys())[col_idx]
                matched_tracks.append(track_id)
                matched_detections.append(row_idx)
                unmatched_detections.remove(row_idx)
                unmatched_tracks.remove(track_id)
        
        return matched_tracks, matched_detections, unmatched_detections, unmatched_tracks
    
    def _build_cost_matrix(self, detections: List[Detection], tracks: List[TrackState]) -> np.ndarray:
        """Build cost matrix for detection-track association."""
        if not detections or not tracks:
            return np.array([])
        
        # Position-based costs
        det_positions = np.array([det.position for det in detections])
        track_positions = np.array([track.position for track in tracks])
        
        position_costs = compute_cost_matrix_vectorized(
            det_positions, track_positions, self.config.max_distance
        )
        
        # Add embedding costs if available
        if (self.config.use_embeddings and 
            any(det.embedding is not None for det in detections) and
            any(len(track.embeddings) > 0 for track in tracks)):
            
            embedding_costs = self._compute_embedding_cost_matrix(detections, tracks)
            # Scale embedding costs to [0, 1]
            if embedding_costs.size > 0:
                embedding_costs = self.embedding_scaler.scale_distances(embedding_costs.flatten()).reshape(embedding_costs.shape)
                # Combine costs
                total_costs = (1 - self.config.embedding_weight) * position_costs + \
                             self.config.embedding_weight * embedding_costs * self.config.max_distance
            else:
                total_costs = position_costs
        else:
            total_costs = position_costs
        
        return total_costs
    
    def _compute_embedding_cost_matrix(self, detections: List[Detection], tracks: List[TrackState]) -> np.ndarray:
        """Compute embedding-based cost matrix."""
        # Get detection embeddings
        det_embeddings = []
        valid_det_indices = []
        for i, det in enumerate(detections):
            if det.embedding is not None:
                det_embeddings.append(det.embedding)
                valid_det_indices.append(i)
        
        if not det_embeddings:
            return np.full((len(detections), len(tracks)), 1.0, dtype=np.float32)
        
        det_embeddings = np.array(det_embeddings, dtype=np.float32)
        
        # Get track embeddings
        track_embeddings_list = []
        track_embedding_counts = []
        
        for track in tracks:
            if len(track.embeddings) > 0:
                track_embs = list(track.embeddings)
                track_embeddings_list.extend(track_embs)
                track_embedding_counts.append(len(track_embs))
            else:
                track_embedding_counts.append(0)
        
        if not track_embeddings_list:
            return np.full((len(detections), len(tracks)), 1.0, dtype=np.float32)
        
        track_embeddings_array = np.array(track_embeddings_list, dtype=np.float32)
        track_counts_array = np.array(track_embedding_counts, dtype=np.int32)
        
        # Compute embedding distances
        valid_distances = compute_embedding_distances_multi_history(
            det_embeddings, track_embeddings_array, track_counts_array,
            method=self.config.embedding_matching_method
        )
        
        # Update scaler statistics
        self.embedding_scaler.update_statistics(valid_distances.flatten())
        
        # Create full cost matrix
        cost_matrix = np.full((len(detections), len(tracks)), 1.0, dtype=np.float32)
        for i, det_idx in enumerate(valid_det_indices):
            cost_matrix[det_idx, :] = valid_distances[i, :]
        
        return cost_matrix
    
    def _update_matched_tracks(self, matched_tracks: List[int], matched_detections: List[int], detections: List[Detection]):
        """Update matched tracks with detections."""
        for track_id, det_idx in zip(matched_tracks, matched_detections):
            track = self.tracks[track_id]
            detection = detections[det_idx]
            
            # Kalman update
            z = np.asarray(detection.position, dtype=np.float32)  # Measurement
            y = z - self.H @ track.state  # Innovation
            S = self.H @ track.covariance @ self.H.T + self.R  # Innovation covariance
            K = track.covariance @ self.H.T @ np.linalg.inv(S)  # Kalman gain
            
            track.state = track.state + K @ y
            track.covariance = (np.eye(4) - K @ self.H) @ track.covariance
            
            # Update track properties
            track.position = track.state[:2]
            track.velocity = track.state[2:]
            track.bbox = detection.bbox if detection.bbox is not None else track.bbox
            track.confidence = detection.confidence
            track.hits += 1
            track.time_since_update = 0
            track.hit_streak += 1
            
            if detection.class_id is not None:
                track.class_id = detection.class_id
            
            # Update embedding history
            if detection.embedding is not None:
                if len(track.embeddings) >= self.config.max_embeddings_per_track:
                    track.embeddings.popleft()
                track.embeddings.append(detection.embedding.copy())
    
    def _handle_unmatched_tracks(self, unmatched_tracks: List[int]):
        """Handle tracks that were not matched with any detection."""
        tracks_to_remove = []
        
        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            track.hit_streak = 0
            
            if track.time_since_update >= self.config.max_age:
                tracks_to_remove.append(track_id)
                # Move to lost tracks for potential re-identification
                if self.config.reid_enabled and len(track.embeddings) > 0:
                    self.lost_tracks[track_id] = track
                    # Clean up old lost tracks
                    lost_to_remove = [tid for tid, t in self.lost_tracks.items() 
                                    if t.time_since_update > self.config.reid_max_frames]
                    for tid in lost_to_remove:
                        del self.lost_tracks[tid]
        
        # Remove expired tracks
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def _process_unmatched_detections(self, unmatched_detections: List[int], detections: List[Detection]):
        """Process detections that were not matched with any track."""
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            
            # Try to match with pending detections
            matched_pending = False
            detection_pos = np.asarray(detection.position)
            for pending in self.pending_detections[:]:
                distance = np.linalg.norm(detection_pos - pending.position)
                if distance < self.config.pending_detection_distance:
                    # Update pending detection
                    frame_gap = self.frame_count - pending.last_seen_frame
                    if frame_gap <= self.config.max_detection_gap:
                        pending.consecutive_frames += 1
                        pending.last_seen_frame = self.frame_count
                        pending.total_detections += 1
                        
                        # Update average position
                        alpha = 0.3
                        pending.average_position = (1 - alpha) * pending.average_position + alpha * detection_pos
                        pending.position = detection_pos
                        pending.bbox = detection.bbox if detection.bbox is not None else pending.bbox
                        pending.confidence = max(pending.confidence, detection.confidence)
                        
                        if detection.embedding is not None:
                            pending.embedding = detection.embedding
                        
                        # Check if ready for track creation
                        if pending.consecutive_frames >= self.config.min_consecutive_detections:
                            self._create_track_from_pending(pending)
                            self.pending_detections.remove(pending)
                        
                        matched_pending = True
                        break
                    else:
                        # Gap too large, remove this pending
                        self.pending_detections.remove(pending)
            
            if not matched_pending:
                # Create new pending detection
                self.pending_detections.append(PendingDetection(
                    position=np.asarray(detection.position).copy(),
                    embedding=detection.embedding.copy() if detection.embedding is not None else None,
                    bbox=detection.bbox.copy() if detection.bbox is not None else np.zeros(4, dtype=np.float32),
                    confidence=detection.confidence,
                    first_seen_frame=self.frame_count,
                    last_seen_frame=self.frame_count
                ))
    
    def _create_track_from_pending(self, pending: PendingDetection):
        """Create a new track from a pending detection."""
        track = TrackState(
            id=self.next_id,
            position=pending.position.copy(),
            bbox=pending.bbox.copy(),
            confidence=pending.confidence
        )
        
        if pending.embedding is not None:
            track.embeddings.append(pending.embedding.copy())
        
        # Initialize Kalman filter
        track.state[:2] = pending.position
        track.hits = pending.total_detections
        track.age = self.frame_count - pending.first_seen_frame + 1
        track.hit_streak = pending.consecutive_frames
        
        self.tracks[self.next_id] = track
        self.next_id += 1
    
    def _attempt_reidentification(self, detections: List[Detection]):
        """Attempt to re-identify lost tracks with current detections."""
        if not self.lost_tracks or not detections:
            return
        
        # Get detections with embeddings
        det_with_emb = [(i, det) for i, det in enumerate(detections) 
                       if det.embedding is not None]
        
        if not det_with_emb:
            return
        
        for track_id, lost_track in list(self.lost_tracks.items()):
            if len(lost_track.embeddings) == 0:
                continue
            
            best_match_idx = None
            best_distance = self.config.reid_max_distance
            best_emb_distance = self.config.reid_embedding_threshold
            
            for det_idx, detection in det_with_emb:
                # Check position distance
                pos_distance = np.linalg.norm(np.asarray(detection.position) - lost_track.position)
                if pos_distance > self.config.reid_max_distance:
                    continue
                
                # Check embedding distance
                track_emb = list(lost_track.embeddings)[-1]  # Use most recent embedding
                emb_distance = cosine_similarity_normalized(detection.embedding, track_emb)
                
                if (emb_distance < best_emb_distance and 
                    pos_distance < best_distance):
                    best_match_idx = det_idx
                    best_distance = pos_distance
                    best_emb_distance = emb_distance
            
            if best_match_idx is not None:
                # Re-activate track
                detection = detections[best_match_idx]
                lost_track.position = detection.position.copy()
                lost_track.bbox = detection.bbox if detection.bbox is not None else lost_track.bbox
                lost_track.confidence = detection.confidence
                lost_track.time_since_update = 0
                lost_track.hits += 1
                lost_track.hit_streak = 1
                
                # Reset Kalman state
                lost_track.state[:2] = detection.position
                lost_track.state[2:] = 0  # Reset velocity
                
                # Update embedding
                if len(lost_track.embeddings) >= self.config.max_embeddings_per_track:
                    lost_track.embeddings.popleft()
                lost_track.embeddings.append(detection.embedding.copy())
                
                # Move back to active tracks
                self.tracks[track_id] = lost_track
                del self.lost_tracks[track_id]
    
    def _get_tracked_objects(self) -> List[TrackedObject]:
        """Convert internal tracks to TrackedObject instances."""
        tracked_objects = []
        
        for track in self.tracks.values():
            # Only return tracks that have been confirmed
            if track.hits >= self.config.min_consecutive_detections or track.hit_streak > 0:
                tracked_obj = TrackedObject(
                    id=track.id,
                    position=track.position.copy(),
                    velocity=track.velocity.copy(),
                    confidence=track.confidence,
                    age=track.age,
                    hits=track.hits,
                    time_since_update=track.time_since_update,
                    state=1 if track.hits >= self.config.min_consecutive_detections else 0,
                    bbox=track.bbox.copy() if track.bbox is not None else None,
                    class_id=track.class_id
                )
                tracked_objects.append(tracked_obj)
        
        return tracked_objects
    
    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.lost_tracks.clear()
        self.pending_detections.clear()
        self.next_id = 1
        self.frame_count = 0
        self.timing_stats.clear()
        self.debug_info.clear()
    
    def get_statistics(self) -> dict:
        """Get tracker statistics."""
        return {
            'frame_count': self.frame_count,
            'active_tracks': len(self.tracks),
            'lost_tracks': len(self.lost_tracks),
            'pending_detections': len(self.pending_detections),
            'next_id': self.next_id,
            'embedding_scaler_stats': self.embedding_scaler.get_statistics(),
            'timing_stats': self.timing_stats.copy()
        }