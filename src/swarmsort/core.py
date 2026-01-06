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
    compute_embedding_distances_matmul,
    compute_cost_matrix_vectorized,
    compute_cost_matrix_parallel,
    compute_probabilistic_cost_matrix_vectorized,
    compute_freeze_flags_vectorized,
    compute_neighbor_counts_vectorized,
    compute_deduplication_mask,
    estimate_track_covariances
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
        self.embedding_threshold_adjustment = getattr(self.config, 'embedding_threshold_adjustment', 1.0)
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

        # Embedding freeze density parameters
        self.embedding_freeze_density = getattr(self.config, 'embedding_freeze_density', 1)
        # Use local_density_radius if set, otherwise fall back to collision_safety_distance
        local_density = getattr(self.config, 'local_density_radius', -1.0)
        self.local_density_radius = local_density if local_density > 0 else self.collision_safety_distance

        # Debug options
        self.debug_timings = self.config.debug_timings
        self.debug_embeddings = self.config.debug_embeddings

        # Cost computation
        self.use_probabilistic_costs = self.config.use_probabilistic_costs

        # Performance options
        # parallel_cost_matrix: Use parallel cost matrix computation for better performance
        # Note: May produce slightly non-deterministic results due to floating-point order
        self.parallel_cost_matrix = getattr(self.config, 'parallel_cost_matrix', False)

        # Initialization threshold - minimum confidence to create pending detection
        self.init_conf_threshold = getattr(self.config, 'init_conf_threshold', 0.0)

    def _setup_tracker_state(self):
        """Initialize tracker state variables."""
        self._tracks: Dict[int, FastTrackState] = {}
        self._pending_detections: Dict[int, PendingDetection] = {}
        self._next_id = 1
        self._next_pending_id = 1
        self._frame_count = 0
        self.timings = {}

        # OPTIMIZATION: Reusable array pools to avoid per-frame allocations
        # These grow as needed and are reused across frames
        self._reusable_det_positions: Optional[np.ndarray] = None
        self._reusable_track_positions: Optional[np.ndarray] = None
        self._reusable_track_last_positions: Optional[np.ndarray] = None
        self._reusable_track_misses: Optional[np.ndarray] = None
        self._max_dets_seen = 0
        self._max_tracks_seen = 0

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
                method=getattr(self.config, 'embedding_scaling_method', 'min_robustmax')
            )
        else:
            self.embedding_extractor = None
            self.embedding_scaler = None

    def _precompile_numba(self):
        """Pre-compile Numba functions for better performance."""
        if not getattr(self, '_numba_compiled', False):
            try:
                # Create dummy data for compilation
                emb_dim = getattr(self.config, 'default_embedding_dimension', 128)
                dummy_det = np.array([[0.0, 0.0]], dtype=np.float32)
                dummy_track = np.array([[0.0, 0.0]], dtype=np.float32)
                dummy_emb = np.zeros((1, emb_dim), dtype=np.float32)
                dummy_cost = np.array([[0.0]], dtype=np.float32)

                # Trigger compilation
                compute_cost_matrix_vectorized(
                    dummy_det, dummy_track, dummy_track,
                    np.array([0]), dummy_emb, 0.5, 100.0, False, 3, 1.0
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

        Raises:
            TypeError: If detections is not a list
            ValueError: If any detection has invalid data
        """
        # Input validation
        if detections is not None and not isinstance(detections, list):
            raise TypeError(f"detections must be a list, got {type(detections).__name__}")

        self._frame_count += 1
        timer = Timer() if self.debug_timings else None

        # Handle empty detections
        if not detections:
            return self._handle_empty_frame()

        # Validate individual detections
        self._validate_detections(detections)

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

        # CRITICAL: Predict ALL track positions BEFORE assignment
        # This ensures cost matrix uses current-frame predictions
        if timer: timer.start("prediction")
        for track in tracks:
            track.predict_position(self._frame_count)
        if timer: timer.stop("prediction", self.timings)

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
        if self.reid_enabled and len(unmatched_dets) > 0:
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

    def _validate_detections(self, detections: List[Detection]) -> None:
        """
        Validate detection inputs.

        Args:
            detections: List of Detection objects to validate

        Raises:
            TypeError: If detection is not a Detection object
            ValueError: If detection has invalid position or contains non-finite values
        """
        for i, det in enumerate(detections):
            # Duck typing check - allow any object with required attributes
            # This supports Detection objects from different modules (e.g., swarmtracker)
            if not hasattr(det, 'position') or not hasattr(det, 'confidence'):
                raise TypeError(
                    f"Detection {i} must have 'position' and 'confidence' attributes, got {type(det).__name__}"
                )

            # Position validation
            if det.position is None:
                raise ValueError(f"Detection {i}: position cannot be None")

            if not isinstance(det.position, np.ndarray):
                # Auto-convert list/tuple to numpy array
                try:
                    det.position = np.asarray(det.position, dtype=np.float32)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Detection {i}: position must be array-like, got {type(det.position).__name__}"
                    )

            # Auto-convert position to shape (2,) if possible
            if det.position.shape != (2,):
                if det.position.size == 2:
                    # Flatten compatible shapes like (1, 2), (2, 1), (2,)
                    det.position = det.position.flatten().astype(np.float32)
                else:
                    raise ValueError(
                        f"Detection {i}: position must have 2 elements, got shape {det.position.shape}"
                    )

            if not np.isfinite(det.position).all():
                raise ValueError(
                    f"Detection {i}: position contains non-finite values: {det.position}"
                )

            # Confidence validation
            if not 0 <= det.confidence <= 1:
                raise ValueError(
                    f"Detection {i}: confidence must be in [0, 1], got {det.confidence}"
                )

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
        """Update embedding freeze states based on track proximity with hysteresis.

        Uses embedding_freeze_density to determine when to freeze:
        - Freeze when neighbor_count >= embedding_freeze_density
        - Unfreeze when neighbor_count < max(0, embedding_freeze_density - 1)

        The hysteresis (freeze_density - 1 for unfreeze) prevents oscillation
        when tracks are on the boundary of the collision zone.
        """
        if not self.collision_freeze_embeddings or len(self._tracks) < 2:
            return

        positions = np.array(
            [t.position for t in self._tracks.values()],
            dtype=np.float32
        )

        # Count neighbors within local_density_radius
        neighbor_counts = compute_neighbor_counts_vectorized(
            positions, self.local_density_radius
        )

        # Hysteresis thresholds
        freeze_threshold = self.embedding_freeze_density
        # Unfreeze requires fewer neighbors than freeze (hysteresis prevents oscillation)
        unfreeze_threshold = max(0, self.embedding_freeze_density - 1)

        for (track, count) in zip(self._tracks.values(), neighbor_counts):
            if count >= freeze_threshold:
                # Too many neighbors - freeze embeddings
                track.freeze_embeddings()
            elif count <= unfreeze_threshold and track.embedding_frozen:
                # Few enough neighbors and currently frozen - safe to unfreeze
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
        # Return any newly promoted tracks (don't always return empty!)
        return self._get_results()

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
            # hungarian_fallback_threshold is a multiplier, convert to absolute threshold
            hungarian_threshold = self.max_distance * self.hungarian_fallback_threshold
            return hybrid_assignment(
                cost_matrix,
                self.max_distance,
                self.greedy_threshold,
                hungarian_threshold
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

        # OPTIMIZATION: Use reusable array pools to avoid per-frame allocations
        # Grow arrays if needed, but never shrink (avoids repeated allocations)
        if self._reusable_det_positions is None or self._reusable_det_positions.shape[0] < n_dets:
            new_size = max(n_dets, self._max_dets_seen, 50)  # Minimum 50 to reduce early reallocations
            self._reusable_det_positions = np.empty((new_size, 2), dtype=np.float32)
            self._max_dets_seen = new_size

        if self._reusable_track_positions is None or self._reusable_track_positions.shape[0] < n_tracks:
            new_size = max(n_tracks, self._max_tracks_seen, 50)
            self._reusable_track_positions = np.empty((new_size, 2), dtype=np.float32)
            self._reusable_track_last_positions = np.empty((new_size, 2), dtype=np.float32)
            self._reusable_track_misses = np.empty(new_size, dtype=np.int32)
            self._max_tracks_seen = new_size

        # Fill reusable arrays (no new allocation)
        for i, d in enumerate(detections):
            self._reusable_det_positions[i] = d.position
        for i, t in enumerate(tracks):
            self._reusable_track_positions[i] = t.predicted_position
            self._reusable_track_last_positions[i] = t.last_detection_pos
            self._reusable_track_misses[i] = t.misses

        # Use views into reusable arrays
        det_positions = self._reusable_det_positions[:n_dets]
        track_positions = self._reusable_track_positions[:n_tracks]
        track_last_positions = self._reusable_track_last_positions[:n_tracks]
        track_misses = self._reusable_track_misses[:n_tracks]

        # Compute embedding distances if needed
        # Changed from all() to any() - allows partial embeddings
        scaled_embedding_distances = np.zeros((n_dets, n_tracks), dtype=np.float32)
        has_embedding_mask = np.array([
            hasattr(d, 'embedding') and d.embedding is not None
            for d in detections
        ], dtype=bool)
        any_embeddings = self.do_embeddings and np.any(has_embedding_mask)

        if any_embeddings:
            scaled_embedding_distances = self._compute_embedding_distances(
                detections, tracks, has_embedding_mask, det_positions, track_positions
            )

        # do_embeddings flag for cost matrix - only True if we computed embeddings
        do_embeddings = any_embeddings

        # Get configurable parameters with defaults
        miss_threshold = getattr(self.config, 'prediction_miss_threshold', 3)

        # Use appropriate cost function
        if self.use_probabilistic_costs:
            # Extract track velocities for covariance estimation
            track_velocities = np.array([t.velocity for t in tracks], dtype=np.float32)

            # Get covariance estimation parameters (fallbacks match config defaults)
            base_variance = getattr(self.config, 'base_position_variance', 15.0)
            velocity_scale = getattr(self.config, 'velocity_variance_scale', 2.0)
            velocity_isotropic_threshold = getattr(self.config, 'velocity_isotropic_threshold', 0.1)
            cov_inflation_rate = getattr(self.config, 'time_covariance_inflation', 0.1)

            # Estimate base covariances from velocity (miss-based inflation applied in cost function)
            track_covariances = estimate_track_covariances(
                track_velocities, base_variance, velocity_scale, velocity_isotropic_threshold
            )

            # Get probabilistic-specific config parameters (fallbacks match config defaults)
            gating_multiplier = getattr(self.config, 'probabilistic_gating_multiplier', 1.5)
            mahal_normalization = getattr(self.config, 'mahalanobis_normalization', 5.0)
            singular_cov_threshold = getattr(self.config, 'singular_covariance_threshold', 1e-6)

            return compute_probabilistic_cost_matrix_vectorized(
                det_positions, track_positions, track_last_positions,
                track_misses, track_covariances, scaled_embedding_distances,
                self.embedding_weight, self.max_distance, do_embeddings,
                miss_threshold, gating_multiplier, mahal_normalization, cov_inflation_rate,
                self.embedding_threshold_adjustment, singular_cov_threshold
            )
        else:
            # Select cost function based on parallel mode setting
            cost_func = (
                compute_cost_matrix_parallel if self.parallel_cost_matrix
                else compute_cost_matrix_vectorized
            )
            return cost_func(
                det_positions, track_positions, track_last_positions,
                track_misses, scaled_embedding_distances,
                self.embedding_weight, self.max_distance, do_embeddings,
                miss_threshold, self.embedding_threshold_adjustment
            )

    def _compute_embedding_distances(
        self,
        detections: List[Detection],
        tracks: List[FastTrackState],
        has_embedding_mask: np.ndarray = None,
        det_positions: np.ndarray = None,
        track_positions: np.ndarray = None
    ) -> np.ndarray:
        """Compute embedding distances between detections and tracks.

        For detections without embeddings (has_embedding_mask[i] = False),
        the embedding distance is set to normalized position distance so that
        the combined cost equals position-only cost:
            cost = (1-w)*pos + w*(pos/max)*max = pos

        Args:
            detections: List of Detection objects
            tracks: List of FastTrackState objects
            has_embedding_mask: Boolean mask indicating which detections have embeddings
            det_positions: Detection positions [N_det, 2] (for fallback computation)
            track_positions: Track predicted positions [N_track, 2] (for fallback computation)

        Returns:
            Scaled embedding distance matrix [N_det, N_track]
        """
        n_dets = len(detections)
        n_tracks = len(tracks)

        # Handle backward compatibility - if mask not provided, assume all have embeddings
        if has_embedding_mask is None:
            has_embedding_mask = np.ones(n_dets, dtype=bool)

        # Get indices of detections with embeddings
        dets_with_emb = np.where(has_embedding_mask)[0]
        dets_without_emb = np.where(~has_embedding_mask)[0]

        # Log if we have partial embeddings
        if len(dets_without_emb) > 0 and len(dets_with_emb) > 0:
            logger.debug(
                f"Partial embeddings: {len(dets_with_emb)}/{n_dets} detections have embeddings"
            )

        # Get embedding dimension from first detection with embedding, or config default
        emb_dim = getattr(self.config, 'default_embedding_dimension', 128)
        if len(dets_with_emb) > 0:
            first_emb = detections[dets_with_emb[0]].embedding
            if first_emb is not None:
                emb_dim = len(first_emb)

        # Prepare detection embeddings - only for those that have them
        det_embeddings = np.zeros((n_dets, emb_dim), dtype=np.float32)
        for i in dets_with_emb:
            det_embeddings[i] = np.asarray(detections[i].embedding, dtype=np.float32)[:emb_dim]

        # Compute all norms at once (vectorized)
        norms = np.linalg.norm(det_embeddings, axis=1, keepdims=True)

        # Normalize where needed: norm > 0 and not already unit length
        # Avoid division by zero by using np.where
        needs_norm = (norms > 1e-8) & (np.abs(norms - 1.0) > 0.01)
        safe_norms = np.where(norms > 1e-8, norms, 1.0)  # Avoid div by zero
        det_embeddings = np.where(needs_norm, det_embeddings / safe_norms, det_embeddings)

        # Prepare track embeddings using vectorized operations
        # OPTIMIZATION: Avoid np.vstack per track - directly copy embeddings

        # First pass: count total embeddings and per-track counts
        track_counts = np.zeros(n_tracks, dtype=np.int32)
        total_embeddings = 0
        for i, track in enumerate(tracks):
            count = len(track.embedding_history)
            track_counts[i] = count if count > 0 else 0
            total_embeddings += count if count > 0 else 1  # Reserve space for zero embedding

        # Pre-allocate flat array (zeros handles empty tracks automatically)
        track_embeddings_flat = np.zeros(total_embeddings * emb_dim, dtype=np.float32)

        # Second pass: directly copy each embedding without intermediate vstack
        offset = 0
        for i, track in enumerate(tracks):
            n_track_embs = len(track.embedding_history)
            if n_track_embs > 0:
                # Directly iterate and copy each embedding (avoids vstack allocation)
                for emb in track.embedding_history:
                    emb_arr = np.asarray(emb, dtype=np.float32).ravel()
                    track_embeddings_flat[offset:offset + emb_dim] = emb_arr[:emb_dim]
                    offset += emb_dim
            else:
                # Zero embedding for tracks without history (already zeros)
                offset += emb_dim

        # Method mapping
        method_map = {
            "last": 0,
            "average": 1,
            "weighted_average": 2,
            "best_match": 3
        }
        method = method_map.get(self.embedding_matching_method, 1)

        # Compute distances
        # OPTIMIZATION: Use fast matrix multiplication path for method=0 (last embedding)
        # This is 2-3x faster than the Numba loop for large matrices
        if method == 0:
            distances = compute_embedding_distances_matmul(
                det_embeddings, track_embeddings_flat, track_counts
            )
        else:
            distances = compute_embedding_distances_with_method(
                det_embeddings, track_embeddings_flat, track_counts, method
            )

        # Scale distances if scaler is available
        if self.embedding_scaler:
            distances_flat = distances.flatten()
            scaled_flat = self.embedding_scaler.scale_distances(distances_flat)
            self.embedding_scaler.update_statistics(distances_flat)
            distances = scaled_flat.reshape(distances.shape)

        # For detections without embeddings, set embedding distance = pos_dist / max_distance
        # This makes the combined cost equal position distance:
        # cost = (1-w)*pos + w*(pos/max)*max = (1-w)*pos + w*pos = pos
        if len(dets_without_emb) > 0 and det_positions is not None and track_positions is not None:
            for i in dets_without_emb:
                for j in range(n_tracks):
                    # Compute position distance
                    pos_dist = np.linalg.norm(det_positions[i] - track_positions[j])
                    # Normalize to [0, 1] range (same as scaled embedding distances)
                    distances[i, j] = min(pos_dist / self.max_distance, 1.0)

        return distances

    def _update_matched_tracks(
        self,
        matches: List[Tuple[int, int]],
        detections: List[Detection],
        tracks: List[FastTrackState]
    ):
        """Update tracks with matched detections."""
        store_scores = getattr(self.config, 'store_embedding_scores', False)

        for det_idx, track_idx in matches:
            if track_idx < len(tracks):
                track = tracks[track_idx]
                det = detections[det_idx]

                # Compute and store embedding similarity score if enabled
                if store_scores and self.do_embeddings:
                    det_emb = getattr(det, 'embedding', None)
                    if det_emb is not None and len(track.embedding_history) > 0:
                        # Compute cosine similarity with track's representative embedding
                        track_emb = track.get_representative_embedding()
                        if track_emb is not None:
                            # Normalize embeddings
                            det_norm = det_emb / (np.linalg.norm(det_emb) + 1e-8)
                            track_norm = track_emb / (np.linalg.norm(track_emb) + 1e-8)
                            # Cosine similarity (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)
                            similarity = float(np.dot(det_norm, track_norm))
                            track.embedding_score_history.append(similarity)

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
        """Handle tracks that weren't matched to any detection.

        Note: predict_position() was already called for ALL tracks BEFORE assignment,
        so we only need to update counters here (no double prediction).
        """
        for track_idx in unmatched_track_indices:
            if track_idx < len(tracks):
                track = tracks[track_idx]
                # Position already predicted before assignment - just update counters
                track.age += 1
                track.misses += 1
                track.lost_frames += 1

    def _get_detection_confidence(self, detection: Detection) -> float:
        """Extract confidence from detection object."""
        if hasattr(detection, 'confidence'):
            return float(detection.confidence)
        return 1.0

    def _handle_unmatched_detections(
        self,
        unmatched_det_indices: List[int],
        detections: List[Detection]
    ):
        """Handle detections that weren't matched to any track.

        IMPORTANT: This function now includes safeguards against creating
        duplicate tracks by:
        1. Filtering detections by init_conf_threshold
        2. Skipping detections that are close to EXISTING CONFIRMED tracks
           (these should have matched but failed - don't create duplicates)
        """
        if len(unmatched_det_indices) == 0:
            return

        # Get existing confirmed track positions for spatial filtering
        confirmed_tracks = [t for t in self._tracks.values() if t.confirmed]

        # OPTIMIZATION: Pre-compute confirmed track positions as arrays (once, outside loop)
        dedup_threshold = self.deduplication_distance
        has_confirmed_tracks = len(confirmed_tracks) > 0

        if has_confirmed_tracks:
            confirmed_pred_positions = np.array(
                [t.predicted_position for t in confirmed_tracks], dtype=np.float32
            )
            confirmed_last_positions = np.array(
                [t.last_detection_pos for t in confirmed_tracks], dtype=np.float32
            )

        for det_idx in unmatched_det_indices:
            det = detections[det_idx]

            # Check confidence threshold for track initialization
            det_conf = self._get_detection_confidence(det)
            if det_conf < self.init_conf_threshold:
                continue  # Skip low-confidence detections

            det_pos = det.position.flatten()[:2] if hasattr(det.position, 'flatten') else det.position
            det_pos_arr = np.asarray(det_pos, dtype=np.float32)

            # CRITICAL: Skip detections that are very close to existing confirmed tracks
            # These should have matched but failed - creating pending would cause duplicates
            # OPTIMIZATION: Vectorized distance computation to all confirmed tracks at once
            near_existing_track = False
            if has_confirmed_tracks:
                # Compute distances to all confirmed tracks in one operation
                dists_to_pred = np.linalg.norm(confirmed_pred_positions - det_pos_arr, axis=1)
                dists_to_last = np.linalg.norm(confirmed_last_positions - det_pos_arr, axis=1)
                min_dists = np.minimum(dists_to_pred, dists_to_last)

                # Check if any track is within threshold
                if np.any(min_dists < dedup_threshold):
                    near_existing_track = True

            if near_existing_track:
                continue  # Skip - this detection is very close to an existing track

            # Try to match with pending detections
            # OPTIMIZATION: Vectorized distance computation to all pending at once
            matched_pending = False
            pending_distance = self.config.pending_detection_distance

            if self._pending_detections:
                # Pre-compute pending positions array (only when there are pending detections)
                pending_ids = list(self._pending_detections.keys())
                pending_positions = np.array(
                    [self._pending_detections[pid].position for pid in pending_ids],
                    dtype=np.float32
                )

                # Vectorized distance to all pending
                dists_to_pending = np.linalg.norm(pending_positions - det_pos_arr, axis=1)

                # Find closest within threshold
                within_threshold_mask = dists_to_pending < pending_distance
                if np.any(within_threshold_mask):
                    # Get the closest pending within threshold
                    closest_idx = np.argmin(dists_to_pending)
                    if dists_to_pending[closest_idx] < pending_distance:
                        pending_id = pending_ids[closest_idx]
                        pending = self._pending_detections[pending_id]
                        # Update existing pending detection
                        pending.update(
                            det_pos,
                            getattr(det, 'embedding', None),
                            getattr(det, 'bbox', None),
                            det_conf
                        )
                        pending.last_seen_frame = self._frame_count
                        matched_pending = True

            if not matched_pending:
                # Create new pending detection
                pending = PendingDetection(
                    position=det_pos.astype(np.float32) if hasattr(det_pos, 'astype') else np.array(det_pos, dtype=np.float32),
                    embedding=getattr(det, 'embedding', None),
                    bbox=getattr(det, 'bbox', None),
                    confidence=det_conf,
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
                score_history_len = getattr(self.config, 'embedding_score_history_length', 5)
                track.set_embedding_params(
                    self.max_embeddings_per_track,
                    self.embedding_matching_method,
                    score_history_len
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
        """Attempt to re-identify lost tracks using embeddings.

        IMPORTANT: ReID only matches detections that are NOT close to any active track.
        This prevents "stealing" detections that likely belong to active tracks but
        temporarily failed to match due to noise or occlusion.
        """
        if not self.do_embeddings or len(unmatched_det_indices) == 0:
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

        # --- CRITICAL: Filter out detections close to ACTIVE tracks ---
        # Active tracks are those that were recently matched (misses < reid_min_frames_lost)
        # These tracks might have temporarily failed to match due to noise/occlusion
        # We should NOT allow ReID to "steal" their detections
        active_tracks = [
            track for track in self._tracks.values()
            if track.misses < self.reid_min_frames_lost
        ]

        if active_tracks:
            active_positions = np.array(
                [t.predicted_position for t in active_tracks], dtype=np.float32
            )

            # OPTIMIZATION: Vectorized filtering - compute all pairwise distances at once
            # Extract all detection positions as array
            det_positions = np.array(
                [det.position for _, det in unmatched_dets_with_emb], dtype=np.float32
            )

            # Compute all pairwise distances: [n_dets, n_active_tracks]
            # Using broadcasting: det_positions[:, None, :] - active_positions[None, :, :]
            all_distances = np.linalg.norm(
                det_positions[:, None, :] - active_positions[None, :, :], axis=2
            )

            # Find minimum distance to any active track for each detection
            min_distances = np.min(all_distances, axis=1)

            # Create mask: True for detections far enough from all active tracks
            safe_mask = min_distances > self.max_distance

            # Apply filter using mask
            unmatched_dets_with_emb = [
                item for item, is_safe in zip(unmatched_dets_with_emb, safe_mask) if is_safe
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
        reid_boost = getattr(self.config, 'reid_embedding_weight_boost', 1.5)
        reid_embedding_weight = min(self.embedding_weight * reid_boost, 0.95)

        reid_cost_matrix = compute_cost_matrix_vectorized(
            det_positions, track_positions, track_last_positions,
            track_misses, scaled_embedding_distances,
            reid_embedding_weight, self.reid_max_distance, do_embeddings=True,
            miss_threshold=3, embedding_threshold_adjustment=self.embedding_threshold_adjustment
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
        store_scores = getattr(self.config, 'store_embedding_scores', False)

        for track in self._tracks.values():
            if track.confirmed and track.misses == 0:
                # Compute average embedding score if available
                embedding_score = None
                if store_scores and len(track.embedding_score_history) > 0:
                    embedding_score = float(np.mean(list(track.embedding_score_history)))

                results.append(TrackedObject(
                    id=track.id,
                    position=track.position,
                    velocity=track.velocity,
                    confidence=track.detection_confidence,
                    age=track.age,
                    hits=track.hits,
                    time_since_update=track.misses,
                    state=1 if track.confirmed else 0,  # 1: Confirmed, 0: Tentative
                    bbox=track.bbox,
                    predicted_position=track.predicted_position.copy() if track.predicted_position is not None else (track.position + track.velocity),
                    embedding_score=embedding_score
                ))

        return results

    def _log_timings(self):
        """Log timing information for debugging."""
        if self._frame_count % 10 == 0:
            formatted = {k: f"{v*1000:.2f} ms" for k, v in self.timings.items()}
            logger.info(f"[Frame {self._frame_count}] Timings: {formatted}")
            self.timings.clear()

    def reset(self):
        """Reset the tracker to initial state.

        This clears all tracks, pending detections, and resets the embedding
        scaler statistics. Use when starting a new video or scene.
        """
        self._tracks.clear()
        self._pending_detections.clear()
        self._next_id = 1
        self._next_pending_id = 1
        self._frame_count = 0
        self.timings.clear()

        # Reset reusable array pools (allow them to be garbage collected)
        self._reusable_det_positions = None
        self._reusable_track_positions = None
        self._reusable_track_last_positions = None
        self._reusable_track_misses = None
        self._max_dets_seen = 0
        self._max_tracks_seen = 0

        # Reset embedding scaler statistics
        if self.embedding_scaler:
            self.embedding_scaler.reset()

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