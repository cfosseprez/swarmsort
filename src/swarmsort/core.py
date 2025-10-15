"""
SwarmSort Core Implementation

This module contains the core SwarmSort multi-object tracking algorithm implementation.
SwarmSort combines Kalman filtering, Hungarian algorithm assignment, and deep learning
embeddings for robust real-time object tracking.

Classes:
    SwarmSortTracker: Main tracking class implementing the full algorithm
"""

# ============================================================================
# STANDARD IMPORTS
# ============================================================================
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from collections import deque
from scipy.optimize import linear_sum_assignment
import time
import gc

# ============================================================================
# LOGGER
# ============================================================================
from loguru import logger

# ============================================================================
# INTERNAL IMPORTS
# ============================================================================
from .data_classes import Detection, TrackedObject
from .config import SwarmSortConfig
from .embedding_scaler import EmbeddingDistanceScaler
from .track_state import PendingDetection, FastTrackState

from .kalman_filters import (
    simple_kalman_update,
    simple_kalman_predict,
    oc_sort_predict,
    oc_sort_update,
    compute_oc_sort_cost_matrix
)

from .cost_computation import (
    cosine_similarity_normalized,
    compute_embedding_distances_with_method,
    compute_cost_matrix_vectorized,
    compute_probabilistic_cost_matrix_vectorized,
    compute_freeze_flags_vectorized,
    compute_deduplication_mask
)

from .assignment import (
    numba_greedy_assignment,
    hungarian_assignment_wrapper,
    hybrid_assignment
)


# ============================================================================
# PERFORMANCE TIMING UTILITIES
# ============================================================================
class Timer:
    """Simple high-resolution timer for performance profiling."""

    def __init__(self):
        self._start_times = {}

    def start(self, key: str) -> None:
        """Start timing for the given key."""
        self._start_times[key] = time.perf_counter()

    def stop(self, key: str, store: dict) -> None:
        """Stop timing for the given key and accumulate the duration."""
        if key in self._start_times:
            duration = time.perf_counter() - self._start_times[key]
            store[key] = store.get(key, 0.0) + duration


# ============================================================================
# MAIN TRACKER CLASS
# ============================================================================
class SwarmSortTracker:
    """
    Main SwarmSort multi-object tracker implementation.

    This class implements a sophisticated tracking algorithm that combines:
    - Kalman filtering for motion prediction
    - Hungarian/Greedy assignment for detection-track matching
    - Deep learning embeddings for appearance-based matching
    - Re-identification for recovering lost tracks
    """

    def __init__(
        self,
        config: Optional[Union[SwarmSortConfig, dict]] = None,
        embedding_type: Optional[str] = None,
        use_gpu: Optional[bool] = None,
        **kwargs
    ):
        """Initialize the SwarmSort tracker with configuration.

        Args:
            config: Configuration object or dictionary
            embedding_type: Type of embedding extractor to use (overrides config)
            use_gpu: Whether to use GPU for embeddings (overrides config)
            **kwargs: Additional keyword arguments (ignored for compatibility)
        """
        # Handle configuration
        if config is None:
            self.config = SwarmSortConfig()
        elif isinstance(config, dict):
            self.config = SwarmSortConfig.from_dict(config)
        else:
            self.config = config

        # Store embedding preferences for later initialization
        self.embedding_type_override = embedding_type
        self.use_gpu = use_gpu if use_gpu is not None else True

        self._initialize_from_config()
        self._setup_tracker_state()
        self._initialize_embeddings()
        self._precompile_numba()

    def _initialize_from_config(self):
        """Initialize tracker parameters from configuration."""
        # Core tracking parameters
        self.max_distance = self.config.max_distance
        self.max_track_age = self.config.max_track_age
        self.min_consecutive_detections = self.config.min_consecutive_detections
        self.max_detection_gap = self.config.max_detection_gap

        # Embedding parameters
        self.do_embeddings = self.config.do_embeddings
        self.embedding_weight = self.config.embedding_weight
        self.max_embeddings_per_track = self.config.max_embeddings_per_track
        self.embedding_matching_method = self.config.embedding_matching_method

        # Assignment strategy
        self.assignment_strategy = self.config.assignment_strategy
        self.greedy_threshold = self.config.greedy_threshold
        self.hungarian_fallback_threshold = self.config.hungarian_fallback_threshold

        # Kalman filter type
        self.kalman_type = self.config.kalman_type

        # ReID parameters
        self.reid_enabled = self.config.reid_enabled
        self.reid_max_distance = self.config.reid_max_distance
        self.reid_embedding_threshold = self.config.reid_embedding_threshold
        self.reid_min_frames_lost = self.config.reid_min_frames_lost

        # Advanced features
        self.collision_freeze_embeddings = self.config.collision_freeze_embeddings
        self.collision_safety_distance = self.config.collision_safety_distance
        self.deduplication_distance = self.config.deduplication_distance

        # Debug options
        self.debug_timings = self.config.debug_timings
        self.debug_embeddings = self.config.debug_embeddings

        # Cost computation
        self.use_probabilistic_costs = self.config.use_probabilistic_costs

    def _setup_tracker_state(self):
        """Initialize tracker state variables."""
        self._tracks: Dict[int, FastTrackState] = {}
        self._pending_detections: Dict[int, PendingDetection] = {}
        self._next_id = 1
        self._next_pending_id = 1
        self._frame_count = 0
        self.timings = {}

    def _initialize_embeddings(self):
        """Initialize embedding-related components."""
        if self.do_embeddings:
            # Determine which embedding type to use
            # Priority: embedding_type_override > config.embedding_function > config.embedding_extractor
            embedding_type = (
                self.embedding_type_override or
                getattr(self.config, 'embedding_function', None) or
                getattr(self.config, 'embedding_extractor', None)
            )

            # Initialize embedding extractor if specified
            if embedding_type:
                try:
                    from .embeddings import get_embedding_extractor
                    self.embedding_extractor = get_embedding_extractor(
                        embedding_type,
                        use_gpu=self.use_gpu
                    )
                    logger.debug(f"Using embedding extractor: {embedding_type}")
                except Exception as e:
                    logger.warning(f"Failed to create embedding extractor '{embedding_type}': {e}")
                    self.embedding_extractor = None
            else:
                self.embedding_extractor = None

            # Initialize embedding scaler
            self.embedding_scaler = EmbeddingDistanceScaler(
                method=getattr(self.config, 'embedding_scaler_method', 'standard')
            )
        else:
            self.embedding_extractor = None
            self.embedding_scaler = None

    def _precompile_numba(self):
        """Pre-compile Numba functions for better performance."""
        if not getattr(self, '_numba_compiled', False):
            try:
                # Create dummy data for compilation
                dummy_det = np.array([[0.0, 0.0]], dtype=np.float32)
                dummy_track = np.array([[0.0, 0.0]], dtype=np.float32)
                dummy_emb = np.zeros((1, 128), dtype=np.float32)
                dummy_cost = np.array([[0.0]], dtype=np.float32)

                # Trigger compilation
                compute_cost_matrix_vectorized(
                    dummy_det, dummy_track, dummy_track,
                    np.array([0]), dummy_emb, 0.5, 100.0, False
                )
                numba_greedy_assignment(dummy_cost, 100.0)

                self._numba_compiled = True
                logger.debug("Numba functions compiled successfully")
            except Exception as e:
                logger.warning(f"Numba compilation warning: {e}")

    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """
        Main update function for the tracker.

        Args:
            detections: List of Detection objects for current frame

        Returns:
            List of TrackedObject instances representing current tracks
        """
        self._frame_count += 1
        timer = Timer() if self.debug_timings else None

        # Handle empty detections
        if not detections:
            return self._handle_empty_frame()

        # Filter low confidence detections
        if timer: timer.start("filter_conf")
        detections = self._filter_detections(detections)
        if timer: timer.stop("filter_conf", self.timings)

        # Deduplication and collision detection
        if timer: timer.start("dedup_and_collision")
        detections = self._deduplicate_detections(detections)
        self._update_collision_states()
        if timer: timer.stop("dedup_and_collision", self.timings)

        # Get active tracks
        tracks = list(self._tracks.values())
        if not tracks:
            return self._handle_no_tracks(detections)

        # Perform assignment
        if timer: timer.start("assignment")
        matches, unmatched_dets, unmatched_tracks = self._perform_assignment(
            detections, tracks, timer
        )
        if timer: timer.stop("assignment", self.timings)

        # Update matched tracks
        if timer: timer.start("update_matched")
        self._update_matched_tracks(matches, detections, tracks)
        if timer: timer.stop("update_matched", self.timings)

        # Handle unmatched
        if timer: timer.start("handle_unmatched")
        self._handle_unmatched_tracks(unmatched_tracks, tracks)
        self._handle_unmatched_detections(unmatched_dets, detections)
        if timer: timer.stop("handle_unmatched", self.timings)

        # ReID attempt
        if self.reid_enabled and unmatched_dets:
            if timer: timer.start("reid")
            self._attempt_reid(detections, unmatched_dets)
            if timer: timer.stop("reid", self.timings)

        # Update pending detections
        if timer: timer.start("update_pending")
        self._update_pending_detections()
        if timer: timer.stop("update_pending", self.timings)

        # Cleanup
        if timer: timer.start("cleanup")
        self._cleanup_tracks()
        if timer: timer.stop("cleanup", self.timings)

        # Get results
        if timer: timer.start("get_results")
        results = self._get_results()
        if timer: timer.stop("get_results", self.timings)

        if self.debug_timings:
            self._log_timings()

        return results

    def _filter_detections(self, detections: List[Detection]) -> List[Detection]:
        """Filter out low confidence detections."""
        threshold = getattr(self.config, 'detection_conf_threshold', 0.0)
        if threshold > 0:
            return [d for d in detections if d.confidence >= threshold]
        return detections

    def _deduplicate_detections(self, detections: List[Detection]) -> List[Detection]:
        """Remove duplicate detections based on proximity."""
        if not detections or self.deduplication_distance <= 0:
            return detections

        n_dets = len(detections)
        positions = np.array([d.position for d in detections], dtype=np.float32)
        confidences = np.array([d.confidence for d in detections], dtype=np.float32)

        # Use Numba function for efficient deduplication
        keep_mask = compute_deduplication_mask(
            positions, confidences, self.deduplication_distance
        )

        return [d for i, d in enumerate(detections) if keep_mask[i]]

    def _update_collision_states(self):
        """Update embedding freeze states based on track proximity."""
        if not self.collision_freeze_embeddings or len(self._tracks) < 2:
            return

        positions = np.array(
            [t.position for t in self._tracks.values()],
            dtype=np.float32
        )

        freeze_flags = compute_freeze_flags_vectorized(
            positions, self.collision_safety_distance
        )

        for (track, should_freeze) in zip(self._tracks.values(), freeze_flags):
            if should_freeze:
                track.freeze_embeddings()
            else:
                track.unfreeze_embeddings()

    def _handle_empty_frame(self) -> List[TrackedObject]:
        """Handle frame with no detections."""
        for track in self._tracks.values():
            track.predict_only(self._frame_count)
        self._cleanup_tracks()
        return self._get_results()

    def _handle_no_tracks(self, detections: List[Detection]) -> List[TrackedObject]:
        """Handle case when there are no active tracks."""
        self._handle_unmatched_detections(list(range(len(detections))), detections)
        self._update_pending_detections()
        return []

    def _perform_assignment(
        self,
        detections: List[Detection],
        tracks: List[FastTrackState],
        timer: Optional[Timer] = None
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Perform detection-track assignment using configured strategy."""
        # Compute cost matrix
        cost_matrix = self._compute_cost_matrix(detections, tracks)

        # Apply assignment strategy
        if self.assignment_strategy == "greedy":
            return numba_greedy_assignment(cost_matrix, self.max_distance)
        elif self.assignment_strategy == "hungarian":
            return hungarian_assignment_wrapper(cost_matrix, self.max_distance)
        elif self.assignment_strategy == "hybrid":
            return hybrid_assignment(
                cost_matrix,
                self.max_distance,
                self.greedy_threshold,
                self.hungarian_fallback_threshold
            )
        else:
            # Default to Hungarian
            return hungarian_assignment_wrapper(cost_matrix, self.max_distance)

    def _compute_cost_matrix(
        self,
        detections: List[Detection],
        tracks: List[FastTrackState]
    ) -> np.ndarray:
        """Compute cost matrix for detection-track assignment."""
        n_dets = len(detections)
        n_tracks = len(tracks)

        if n_dets == 0 or n_tracks == 0:
            return np.full((n_dets, n_tracks), np.inf, dtype=np.float32)

        # Extract positions
        det_positions = np.array([d.position for d in detections], dtype=np.float32)
        track_positions = np.array([t.predicted_position for t in tracks], dtype=np.float32)
        track_last_positions = np.array([t.last_detection_pos for t in tracks], dtype=np.float32)
        track_misses = np.array([t.misses for t in tracks], dtype=np.int32)

        # Compute embedding distances if needed
        scaled_embedding_distances = np.zeros((n_dets, n_tracks), dtype=np.float32)
        do_embeddings = self.do_embeddings and all(
            hasattr(d, 'embedding') and d.embedding is not None for d in detections
        )

        if do_embeddings:
            scaled_embedding_distances = self._compute_embedding_distances(
                detections, tracks
            )

        # Use appropriate cost function
        if self.use_probabilistic_costs:
            # Compute covariances (simplified for now)
            track_covariances = np.tile(np.eye(2, dtype=np.float32) * 10, (n_tracks, 1, 1))

            return compute_probabilistic_cost_matrix_vectorized(
                det_positions, track_positions, track_last_positions,
                track_misses, track_covariances, scaled_embedding_distances,
                self.embedding_weight, self.max_distance, do_embeddings
            )
        else:
            return compute_cost_matrix_vectorized(
                det_positions, track_positions, track_last_positions,
                track_misses, scaled_embedding_distances,
                self.embedding_weight, self.max_distance, do_embeddings
            )

    def _compute_embedding_distances(
        self,
        detections: List[Detection],
        tracks: List[FastTrackState]
    ) -> np.ndarray:
        """Compute embedding distances between detections and tracks."""
        n_dets = len(detections)
        n_tracks = len(tracks)

        # Prepare detection embeddings
        det_embeddings = []
        for det in detections:
            emb = np.asarray(det.embedding, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                if abs(norm - 1.0) > 0.01:
                    emb = emb / norm
            det_embeddings.append(emb)
        det_embeddings = np.array(det_embeddings, dtype=np.float32)

        # Prepare track embeddings
        track_embeddings_flat = []
        track_counts = []

        for track in tracks:
            if len(track.embedding_history) > 0:
                for emb in track.embedding_history:
                    track_embeddings_flat.extend(emb)
                track_counts.append(len(track.embedding_history))
            else:
                # No embeddings for this track
                track_embeddings_flat.extend(np.zeros(128, dtype=np.float32))
                track_counts.append(0)

        track_embeddings_flat = np.array(track_embeddings_flat, dtype=np.float32)
        track_counts = np.array(track_counts, dtype=np.int32)

        # Method mapping
        method_map = {
            "last": 0,
            "average": 1,
            "weighted_average": 2,
            "best_match": 3
        }
        method = method_map.get(self.embedding_matching_method, 1)

        # Compute distances
        distances = compute_embedding_distances_with_method(
            det_embeddings, track_embeddings_flat, track_counts, method
        )

        # Scale distances if scaler is available
        if self.embedding_scaler:
            distances_flat = distances.flatten()
            scaled_flat = self.embedding_scaler.scale_distances(distances_flat)
            self.embedding_scaler.update_statistics(distances_flat)
            return scaled_flat.reshape(distances.shape)

        return distances

    def _update_matched_tracks(
        self,
        matches: List[Tuple[int, int]],
        detections: List[Detection],
        tracks: List[FastTrackState]
    ):
        """Update tracks with matched detections."""
        for det_idx, track_idx in matches:
            if track_idx < len(tracks):
                track = tracks[track_idx]
                det = detections[det_idx]

                track.update_with_detection(
                    position=det.position,
                    embedding=getattr(det, 'embedding', None),
                    bbox=getattr(det, 'bbox', None),
                    frame=self._frame_count,
                    det_conf=det.confidence
                )

                # Update confirmation status
                if not track.confirmed and track.hits >= self.min_consecutive_detections:
                    track.confirmed = True

    def _handle_unmatched_tracks(
        self,
        unmatched_track_indices: List[int],
        tracks: List[FastTrackState]
    ):
        """Handle tracks that weren't matched to any detection."""
        for track_idx in unmatched_track_indices:
            if track_idx < len(tracks):
                track = tracks[track_idx]
                track.predict_only(self._frame_count)

    def _handle_unmatched_detections(
        self,
        unmatched_det_indices: List[int],
        detections: List[Detection]
    ):
        """Handle detections that weren't matched to any track."""
        for det_idx in unmatched_det_indices:
            det = detections[det_idx]

            # Try to match with pending detections
            matched_pending = False
            for pending_id, pending in list(self._pending_detections.items()):
                dist = np.linalg.norm(det.position - pending.position)

                if dist < self.config.pending_detection_distance:
                    # Update existing pending detection
                    pending.update(
                        det.position,
                        getattr(det, 'embedding', None),
                        getattr(det, 'bbox', None),
                        det.confidence
                    )
                    pending.last_seen_frame = self._frame_count
                    matched_pending = True
                    break

            if not matched_pending:
                # Create new pending detection
                pending = PendingDetection(
                    position=det.position,
                    embedding=getattr(det, 'embedding', None),
                    bbox=getattr(det, 'bbox', None),
                    confidence=det.confidence,
                    first_seen_frame=self._frame_count,
                    last_seen_frame=self._frame_count
                )
                self._pending_detections[self._next_pending_id] = pending
                self._next_pending_id += 1

    def _update_pending_detections(self):
        """Promote pending detections to tracks if they meet criteria."""
        to_remove = []

        for pending_id, pending in list(self._pending_detections.items()):
            if pending.is_ready_for_promotion(
                self.min_consecutive_detections,
                self.max_detection_gap,
                self._frame_count
            ):
                # Create new track
                track = FastTrackState(
                    id=self._next_id,
                    position=pending.average_position,
                    kalman_type=self.kalman_type
                )

                # Set embedding parameters
                track.set_embedding_params(
                    self.max_embeddings_per_track,
                    self.embedding_matching_method
                )

                # Add initial embedding if available
                if pending.embedding is not None:
                    track.add_embedding(pending.embedding)

                # Initialize track
                track.update_with_detection(
                    pending.average_position,
                    pending.embedding,
                    pending.bbox,
                    self._frame_count,
                    pending.confidence
                )

                track.hits = pending.total_detections
                track.age = pending.consecutive_frames

                if track.hits >= self.min_consecutive_detections:
                    track.confirmed = True

                self._tracks[self._next_id] = track
                self._next_id += 1
                to_remove.append(pending_id)

            elif (self._frame_count - pending.last_seen_frame) > self.max_detection_gap:
                # Remove stale pending detection
                to_remove.append(pending_id)

        for pending_id in to_remove:
            del self._pending_detections[pending_id]

    def _attempt_reid(
        self,
        detections: List[Detection],
        unmatched_det_indices: List[int]
    ):
        """Attempt to re-identify lost tracks using embeddings."""
        if not self.do_embeddings or not unmatched_det_indices:
            return

        # Get lost tracks eligible for ReID
        lost_tracks_with_ids = [
            (tid, track) for tid, track in self._tracks.items()
            if track.misses >= self.reid_min_frames_lost
            and track.misses < self.max_track_age
            and len(track.embedding_history) > 0
        ]

        if not lost_tracks_with_ids:
            return

        # Get unmatched detections with embeddings
        unmatched_dets_with_emb = [
            (idx, detections[idx]) for idx in unmatched_det_indices
            if hasattr(detections[idx], 'embedding')
            and detections[idx].embedding is not None
        ]

        if not unmatched_dets_with_emb:
            return

        # Prepare data for cost matrix computation
        lost_tracks = [track for _, track in lost_tracks_with_ids]
        unmatched_detections = [det for _, det in unmatched_dets_with_emb]

        # --- 1. Compute ReID Cost Matrix ---
        # This reuses the main cost computation logic but with ReID parameters.
        n_dets = len(unmatched_detections)
        n_tracks = len(lost_tracks)

        if n_dets == 0 or n_tracks == 0:
            return

        det_positions = np.array([d.position for d in unmatched_detections], dtype=np.float32)
        track_positions = np.array([t.predicted_position for t in lost_tracks], dtype=np.float32)
        track_last_positions = np.array([t.last_detection_pos for t in lost_tracks], dtype=np.float32)
        track_misses = np.array([t.misses for t in lost_tracks], dtype=np.int32)

        # Compute embedding distances between unmatched dets and lost tracks
        scaled_embedding_distances = self._compute_embedding_distances(
            unmatched_detections, lost_tracks
        )

        # Use a higher embedding weight for ReID to prioritize appearance
        reid_embedding_weight = min(self.embedding_weight * 1.5, 0.95)

        reid_cost_matrix = compute_cost_matrix_vectorized(
            det_positions, track_positions, track_last_positions,
            track_misses, scaled_embedding_distances,
            reid_embedding_weight, self.reid_max_distance, do_embeddings=True
        )

        # --- 2. Perform Assignment ---
        # Use the same assignment strategy as the main tracker for consistency.
        # The cost matrix is already gated by reid_max_distance.
        reid_matches, _, _ = hungarian_assignment_wrapper(
            reid_cost_matrix, self.reid_max_distance
        )

        # Apply ReID matches
        for det_match_idx, track_match_idx in reid_matches:
            # Map indices back to original detection and track IDs
            det_idx, det = unmatched_dets_with_emb[det_match_idx]
            track_id, track = lost_tracks_with_ids[track_match_idx]

            track = self._tracks[track_id]

            track.update_with_detection(
                position=det.position,
                embedding=det.embedding,
                bbox=getattr(det, 'bbox', None),
                frame=self._frame_count,
                det_conf=det.confidence,
                is_reid=True
            )

    def _cleanup_tracks(self):
        """Remove old or unconfirmed tracks."""
        to_remove = []

        for track_id, track in self._tracks.items():
            if track.misses > self.max_track_age:
                to_remove.append(track_id)
            elif not track.confirmed and track.misses > 3:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self._tracks[track_id]

    def _get_results(self) -> List[TrackedObject]:
        """Get current tracking results."""
        results = []

        for track in self._tracks.values():
            if track.confirmed and track.misses == 0:
                results.append(TrackedObject(
                    id=track.id,
                    position=track.position,
                    velocity=track.velocity,
                    confidence=track.detection_confidence,
                    age=track.age,
                    hits=track.hits,
                    time_since_update=track.misses,
                    state=1 if track.confirmed else 0,  # 1: Confirmed, 0: Tentative
                    bbox=track.bbox
                ))

        return results

    def _log_timings(self):
        """Log timing information for debugging."""
        if self._frame_count % 10 == 0:
            formatted = {k: f"{v*1000:.2f} ms" for k, v in self.timings.items()}
            logger.info(f"[Frame {self._frame_count}] Timings: {formatted}")
            self.timings.clear()

    def reset(self):
        """Reset the tracker to initial state."""
        self._tracks.clear()
        self._pending_detections.clear()
        self._next_id = 1
        self._next_pending_id = 1
        self._frame_count = 0
        self.timings.clear()
        gc.collect()

    @property
    def frame_count(self):
        """Get current frame count."""
        return self._frame_count

    def get_state(self) -> dict:
        """Get current tracker state for debugging."""
        return {
            "frame_count": self._frame_count,
            "n_tracks": len(self._tracks),
            "n_pending": len(self._pending_detections),
            "track_ids": list(self._tracks.keys())
        }