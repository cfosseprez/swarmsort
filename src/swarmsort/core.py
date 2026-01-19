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
    oc_sort_predict,
    oc_sort_update,
    compute_oc_sort_cost_matrix
)

from .cost_computation import (
    cosine_similarity_normalized,
    compute_embedding_distances_with_method,
    compute_embedding_distances_matmul,
    compute_embedding_distances_sparse,
    compute_cost_matrix_vectorized,
    compute_cost_matrix_parallel,
    compute_probabilistic_cost_matrix_vectorized,
    compute_freeze_flags_vectorized,
    compute_neighbor_counts_vectorized,
    compute_deduplication_mask,
    estimate_track_covariances,
    compute_sparse_cost_matrix,
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
        self.kalman_velocity_damping = self.config.kalman_velocity_damping

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

        # Uncertainty penalty (0 = disabled, no overhead)
        self.uncertainty_weight = getattr(self.config, 'uncertainty_weight', 0.0)
        self.uncertainty_window = getattr(self.config, 'uncertainty_window', 10)

        # Performance options
        # parallel_cost_matrix: Use parallel cost matrix computation for better performance
        # Note: May produce slightly non-deterministic results due to floating-point order
        self.parallel_cost_matrix = getattr(self.config, 'parallel_cost_matrix', False)

        # Sparse computation threshold - use grid-based spatial indexing for large-scale scenarios
        self.sparse_computation_threshold = getattr(self.config, 'sparse_computation_threshold', 300)

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

        # OPTIMIZATION: Pre-allocate array pools to avoid per-frame allocations
        # Pre-allocate with reasonable default size to avoid early reallocation spikes
        # These grow as needed if object count exceeds initial allocation
        initial_pool_size = getattr(self.config, 'initial_pool_size', 150)
        self._reusable_det_positions = np.empty((initial_pool_size, 2), dtype=np.float32)
        self._reusable_track_positions = np.empty((initial_pool_size, 2), dtype=np.float32)
        self._reusable_track_last_positions = np.empty((initial_pool_size, 2), dtype=np.float32)
        self._reusable_track_misses = np.empty(initial_pool_size, dtype=np.int32)
        self._max_dets_seen = initial_pool_size
        self._max_tracks_seen = initial_pool_size

    def _initialize_embeddings(self):
        """Initialize embedding-related components.

        Embedding modes:
        1. Built-in extractor: Set embedding_function to 'cupytexture', etc.
           SwarmSort will compute embeddings from image crops.
        2. External embeddings: Set embedding_function=None or 'external'.
           You must provide embeddings in Detection.embedding field.
        3. No embeddings: Set do_embeddings=False.
        """
        if self.do_embeddings:
            # Determine which embedding type to use
            # Priority: embedding_type_override > config.embedding_function > config.embedding_extractor
            embedding_type = (
                self.embedding_type_override or
                getattr(self.config, 'embedding_function', None) or
                getattr(self.config, 'embedding_extractor', None)
            )

            # Check if user wants external embeddings (provided in Detection objects)
            if embedding_type is None or embedding_type.lower() == 'external':
                logger.info("Using external embeddings mode - embeddings must be provided in Detection.embedding")
                self.embedding_extractor = None
            else:
                # Try to initialize built-in embedding extractor
                try:
                    from .embeddings import get_embedding_extractor
                    self.embedding_extractor = get_embedding_extractor(
                        embedding_type,
                        use_gpu=self.use_gpu
                    )
                    logger.info(f"Using built-in embedding extractor: {embedding_type}")
                except Exception as e:
                    # Check if this looks like an external embedding type (not a built-in)
                    builtin_types = ['cupytexture', 'cupytexture_color', 'cupytexture_mega', 'cupyshape']
                    if embedding_type.lower() not in builtin_types:
                        logger.warning(
                            f"Embedding '{embedding_type}' is not a built-in type. "
                            f"Available built-in: {builtin_types}. "
                            f"Assuming external embeddings will be provided in Detection.embedding."
                        )
                        self.embedding_extractor = None
                    else:
                        # Built-in type failed to load - this is an error
                        raise RuntimeError(
                            f"Failed to create built-in embedding extractor '{embedding_type}': {e}. "
                            f"If providing external embeddings, set embedding_function='external' or None."
                        )

            # Initialize embedding scaler
            self.embedding_scaler = EmbeddingDistanceScaler(
                method=getattr(self.config, 'embedding_scaling_method', 'min_robustmax')
            )
            # Warmup the scaler to avoid mode transition spike at min_samples
            if hasattr(self.embedding_scaler, 'warmup'):
                self.embedding_scaler.warmup()
        else:
            self.embedding_extractor = None
            self.embedding_scaler = None

    def _precompile_numba(self):
        """Pre-compile Numba functions for better performance.

        Uses realistic data sizes to ensure all code paths are compiled upfront,
        avoiding JIT compilation spikes during actual tracking.
        """
        if not getattr(self, '_numba_compiled', False):
            try:
                # Use realistic data sizes to trigger full compilation
                # Small dummy data (1x1) doesn't exercise all code paths
                warmup_size = getattr(self.config, 'initial_pool_size', 150)
                n_warmup_dets = min(warmup_size, 100)
                n_warmup_tracks = min(warmup_size, 100)
                emb_dim = getattr(self.config, 'default_embedding_dimension', 128)

                # Create realistic-sized dummy data
                dummy_det = np.random.randn(n_warmup_dets, 2).astype(np.float32) * 100
                dummy_track = np.random.randn(n_warmup_tracks, 2).astype(np.float32) * 100
                dummy_track_last = np.random.randn(n_warmup_tracks, 2).astype(np.float32) * 100
                dummy_misses = np.zeros(n_warmup_tracks, dtype=np.int32)
                dummy_emb = np.random.randn(n_warmup_tracks, emb_dim).astype(np.float32)
                dummy_cost = np.random.randn(n_warmup_dets, n_warmup_tracks).astype(np.float32)

                # Trigger compilation of cost matrix functions
                compute_cost_matrix_vectorized(
                    dummy_det, dummy_track, dummy_track_last,
                    dummy_misses, dummy_emb, 0.5, 100.0, False, 3, 1.0
                )

                # Trigger compilation of assignment functions
                numba_greedy_assignment(dummy_cost, 100.0)

                # Trigger compilation of freeze flag computation if available
                try:
                    compute_freeze_flags_vectorized(dummy_track, 25.0, 1)
                except Exception:
                    pass  # Optional function

                # Trigger compilation of neighbor count if available
                try:
                    compute_neighbor_counts_vectorized(dummy_track, 33.0)
                except Exception:
                    pass  # Optional function

                self._numba_compiled = True
                logger.debug(f"Numba functions compiled successfully (warmup size: {n_warmup_dets}x{n_warmup_tracks})")
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
        detailed_timing = self.debug_timings

        # Compute cost matrix
        if detailed_timing:
            t0 = time.perf_counter()

        cost_matrix = self._compute_cost_matrix(detections, tracks)

        if detailed_timing:
            t_cost = time.perf_counter() - t0
            t0 = time.perf_counter()

        # Apply assignment strategy
        if self.assignment_strategy == "greedy":
            result = numba_greedy_assignment(cost_matrix, self.max_distance)
        elif self.assignment_strategy == "hungarian":
            result = hungarian_assignment_wrapper(cost_matrix, self.max_distance)
        elif self.assignment_strategy == "hybrid":
            # hungarian_fallback_threshold is a multiplier, convert to absolute threshold
            hungarian_threshold = self.max_distance * self.hungarian_fallback_threshold
            result = hybrid_assignment(
                cost_matrix,
                self.max_distance,
                self.greedy_threshold,
                hungarian_threshold
            )
        else:
            # Default to Hungarian
            result = hungarian_assignment_wrapper(cost_matrix, self.max_distance)

        if detailed_timing:
            t_assign = time.perf_counter() - t0
            n_dets, n_tracks = cost_matrix.shape
            print(f"  [Assignment] {self.assignment_strategy} | cost_matrix={t_cost*1000:.2f}ms | assign={t_assign*1000:.2f}ms | matches={len(result[0])}")

        return result

    def _compute_cost_matrix(
        self,
        detections: List[Detection],
        tracks: List[FastTrackState]
    ) -> np.ndarray:
        """Compute cost matrix for detection-track assignment."""
        n_dets = len(detections)
        n_tracks = len(tracks)

        # Detailed timing for performance profiling
        detailed_timing = self.debug_timings
        if detailed_timing:
            t_start = time.perf_counter()
            cost_timings = {}

        if n_dets == 0 or n_tracks == 0:
            return np.full((n_dets, n_tracks), np.inf, dtype=np.float32)

        # OPTIMIZATION: Use reusable array pools to avoid per-frame allocations
        # Grow arrays if needed, but never shrink (avoids repeated allocations)
        if detailed_timing:
            t0 = time.perf_counter()

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

        if detailed_timing:
            cost_timings['array_pool'] = time.perf_counter() - t0
            t0 = time.perf_counter()

        # Fill reusable arrays (no new allocation)
        for i, d in enumerate(detections):
            self._reusable_det_positions[i] = d.position
        for i, t in enumerate(tracks):
            self._reusable_track_positions[i] = t.predicted_position
            self._reusable_track_last_positions[i] = t.last_detection_pos
            self._reusable_track_misses[i] = t.misses

        if detailed_timing:
            cost_timings['fill_arrays'] = time.perf_counter() - t0
            t0 = time.perf_counter()

        # Use views into reusable arrays
        det_positions = self._reusable_det_positions[:n_dets]
        track_positions = self._reusable_track_positions[:n_tracks]
        track_last_positions = self._reusable_track_last_positions[:n_tracks]
        track_misses = self._reusable_track_misses[:n_tracks]

        # Check embedding availability
        has_embedding_mask = np.array([
            hasattr(d, 'embedding') and d.embedding is not None
            for d in detections
        ], dtype=bool)
        any_embeddings = self.do_embeddings and np.any(has_embedding_mask)

        if detailed_timing:
            cost_timings['embedding_mask'] = time.perf_counter() - t0
            t0 = time.perf_counter()

        # Get configurable parameters with defaults
        miss_threshold = getattr(self.config, 'prediction_miss_threshold', 3)

        # Compute uncertainty ratios only if uncertainty_weight > 0 (no overhead when disabled)
        if self.uncertainty_weight > 0.0:
            track_uncertainty_ratios = np.array(
                [t.get_recent_miss_ratio() for t in tracks], dtype=np.float32
            )
        else:
            # Pass empty array when disabled (Numba requires consistent types)
            track_uncertainty_ratios = np.empty(0, dtype=np.float32)

        if detailed_timing:
            cost_timings['uncertainty_ratios'] = time.perf_counter() - t0
            t0 = time.perf_counter()

        # =====================================================================
        # SPARSE MODE: Compute sparse pairs FIRST, then sparse embedding distances
        # This is 20-30x faster for large-scale (1000+ objects) scenarios
        # =====================================================================
        sparse_pairs = None
        if not self.use_probabilistic_costs:
            sparse_pairs = self._compute_sparse_pairs(
                det_positions, track_positions, self.max_distance * 1.5
            )

        if detailed_timing and sparse_pairs is not None:
            cost_timings['sparse_pairs'] = time.perf_counter() - t0
            n_pairs = len(sparse_pairs[0])
            t0 = time.perf_counter()

        # Use sparse cost matrix if candidate pairs were found
        if sparse_pairs is not None:
            sparse_det_indices, sparse_track_indices = sparse_pairs

            # Compute embedding distances ONLY for sparse pairs (huge speedup!)
            # For 1000 objects: 36K pairs vs 880K pairs = 24x fewer aggregations
            scaled_embedding_distances = np.zeros((n_dets, n_tracks), dtype=np.float32)
            if any_embeddings:
                scaled_embedding_distances = self._compute_embedding_distances_sparse(
                    detections, tracks, has_embedding_mask,
                    sparse_det_indices, sparse_track_indices,
                    det_positions, track_positions
                )

            if detailed_timing:
                cost_timings['embedding_distances'] = time.perf_counter() - t0
                t0 = time.perf_counter()

            result = compute_sparse_cost_matrix(
                det_positions, track_positions, track_last_positions,
                track_misses, scaled_embedding_distances,
                self.embedding_weight, self.max_distance, any_embeddings,
                miss_threshold, self.embedding_threshold_adjustment,
                sparse_det_indices, sparse_track_indices,
                track_uncertainty_ratios, self.uncertainty_weight
            )

            if detailed_timing:
                cost_timings['cost_func'] = time.perf_counter() - t0
                cost_timings['total'] = time.perf_counter() - t_start
                self._print_cost_matrix_timings(
                    cost_timings, n_dets, n_tracks,
                    f'sparse({n_pairs}/{n_dets*n_tracks})'
                )

            return result

        # =====================================================================
        # NON-SPARSE MODE: Full embedding distance computation
        # =====================================================================
        scaled_embedding_distances = np.zeros((n_dets, n_tracks), dtype=np.float32)
        if any_embeddings:
            scaled_embedding_distances = self._compute_embedding_distances(
                detections, tracks, has_embedding_mask, det_positions, track_positions
            )

        if detailed_timing:
            cost_timings['embedding_distances'] = time.perf_counter() - t0
            t0 = time.perf_counter()

        # do_embeddings flag for cost matrix - only True if we computed embeddings
        do_embeddings = any_embeddings

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

            if detailed_timing:
                cost_timings['prob_prep'] = time.perf_counter() - t0
                t0 = time.perf_counter()

            result = compute_probabilistic_cost_matrix_vectorized(
                det_positions, track_positions, track_last_positions,
                track_misses, track_covariances, scaled_embedding_distances,
                self.embedding_weight, self.max_distance, do_embeddings,
                miss_threshold, gating_multiplier, mahal_normalization, cov_inflation_rate,
                self.embedding_threshold_adjustment, singular_cov_threshold,
                track_uncertainty_ratios, self.uncertainty_weight
            )

            if detailed_timing:
                cost_timings['cost_func'] = time.perf_counter() - t0
                cost_timings['total'] = time.perf_counter() - t_start
                self._print_cost_matrix_timings(cost_timings, n_dets, n_tracks, 'probabilistic')

            return result
        else:
            # Select cost function based on parallel mode setting
            cost_func = (
                compute_cost_matrix_parallel if self.parallel_cost_matrix
                else compute_cost_matrix_vectorized
            )
            func_name = 'parallel' if self.parallel_cost_matrix else 'vectorized'

            result = cost_func(
                det_positions, track_positions, track_last_positions,
                track_misses, scaled_embedding_distances,
                self.embedding_weight, self.max_distance, do_embeddings,
                miss_threshold, self.embedding_threshold_adjustment,
                track_uncertainty_ratios, self.uncertainty_weight
            )

            if detailed_timing:
                cost_timings['cost_func'] = time.perf_counter() - t0
                cost_timings['total'] = time.perf_counter() - t_start
                self._print_cost_matrix_timings(cost_timings, n_dets, n_tracks, func_name)

            return result

    def _print_cost_matrix_timings(self, timings: dict, n_dets: int, n_tracks: int, mode: str):
        """Print detailed cost matrix computation timings."""
        total_ms = timings['total'] * 1000
        breakdown = []
        for key, val in timings.items():
            if key != 'total':
                ms = val * 1000
                pct = (val / timings['total']) * 100 if timings['total'] > 0 else 0
                breakdown.append(f"{key}={ms:.2f}ms({pct:.0f}%)")

        print(f"  [CostMatrix] {mode} | {n_dets}x{n_tracks} | total={total_ms:.2f}ms | {' '.join(breakdown)}")

    def _compute_sparse_pairs(
        self,
        det_positions: np.ndarray,
        track_positions: np.ndarray,
        max_search_radius: float
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Compute sparse candidate pairs using spatial grid indexing.

        For large-scale scenarios (many detections and tracks), this uses grid-based
        spatial indexing to find only nearby detection-track pairs, reducing complexity
        from O(n*m) to O(n*k) where k is the average number of nearby tracks.

        Args:
            det_positions: Detection positions [N_det, 2]
            track_positions: Track predicted positions [N_track, 2]
            max_search_radius: Maximum distance to consider (grid cell size)

        Returns:
            Tuple of (det_indices, track_indices) for candidate pairs,
            or None if sparse mode should not be used (too few objects or too many pairs)
        """
        n_dets = len(det_positions)
        n_tracks = len(track_positions)

        # Only use spatial indexing when it's beneficial (overhead is worth it)
        if n_dets < self.sparse_computation_threshold or n_tracks < self.sparse_computation_threshold:
            return None

        # Build spatial grid index for tracks
        grid_size = max_search_radius
        if grid_size <= 0:
            return None

        track_grid = {}

        # Place tracks in grid cells
        for j in range(n_tracks):
            x, y = track_positions[j]
            grid_x = int(x / grid_size)
            grid_y = int(y / grid_size)
            key = (grid_x, grid_y)
            if key not in track_grid:
                track_grid[key] = []
            track_grid[key].append(j)

        # Find candidate pairs by checking 3x3 neighborhood of each detection
        det_indices = []
        track_indices = []

        for i in range(n_dets):
            x, y = det_positions[i]
            grid_x = int(x / grid_size)
            grid_y = int(y / grid_size)

            # Check neighboring grid cells (3x3 neighborhood)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    key = (grid_x + dx, grid_y + dy)
                    if key in track_grid:
                        for j in track_grid[key]:
                            det_indices.append(i)
                            track_indices.append(j)

        if not det_indices:
            return None

        # Only use sparse mode if it reduces computation significantly (< 50% of full matrix)
        n_pairs = len(det_indices)
        if n_pairs >= n_dets * n_tracks * 0.5:
            return None

        return np.array(det_indices, dtype=np.int32), np.array(track_indices, dtype=np.int32)

    def _compute_embedding_distances(
        self,
        detections: List[Detection],
        tracks: List[FastTrackState],
        has_embedding_mask: np.ndarray = None,
        det_positions: np.ndarray = None,
        track_positions: np.ndarray = None
    ) -> np.ndarray:
        """Compute embedding distances between detections and tracks.

        OPTIMIZED VERSION with buffer reuse and vectorized operations.

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
        from scipy.spatial.distance import cdist

        n_dets = len(detections)
        n_tracks = len(tracks)

        # Handle backward compatibility - if mask not provided, assume all have embeddings
        if has_embedding_mask is None:
            has_embedding_mask = np.ones(n_dets, dtype=bool)

        # Get indices of detections with embeddings
        dets_with_emb = np.where(has_embedding_mask)[0]
        dets_without_emb = np.where(~has_embedding_mask)[0]

        # Get embedding dimension from first detection with embedding, or config default
        emb_dim = getattr(self.config, 'default_embedding_dimension', 128)
        if len(dets_with_emb) > 0:
            first_emb = detections[dets_with_emb[0]].embedding
            if first_emb is not None:
                emb_dim = len(first_emb)

        # === OPTIMIZATION: Reusable buffer for detection embeddings ===
        if not hasattr(self, '_emb_dist_det_buffer') or self._emb_dist_det_buffer.shape[0] < n_dets or self._emb_dist_det_buffer.shape[1] != emb_dim:
            self._emb_dist_det_buffer = np.zeros((max(n_dets, 100), emb_dim), dtype=np.float32)

        det_embeddings = self._emb_dist_det_buffer[:n_dets, :emb_dim]
        det_embeddings[:] = 0.0

        # Batch extract detection embeddings
        for i in dets_with_emb:
            emb = detections[i].embedding
            if emb is not None:
                det_embeddings[i] = np.asarray(emb, dtype=np.float32)[:emb_dim]

        # Vectorized normalization
        norms = np.linalg.norm(det_embeddings, axis=1, keepdims=True)
        mask = (norms > 1e-8).flatten()
        det_embeddings[mask] /= norms[mask]

        # === Track embedding extraction with buffer reuse ===
        track_counts = np.array([len(t.embedding_history) for t in tracks], dtype=np.int32)
        total_embs = sum(max(c, 1) for c in track_counts)

        # Reusable buffer for track embeddings
        if not hasattr(self, '_emb_dist_track_buffer') or len(self._emb_dist_track_buffer) < total_embs * emb_dim:
            self._emb_dist_track_buffer = np.zeros(max(total_embs * emb_dim, 10000), dtype=np.float32)

        track_embeddings_flat = self._emb_dist_track_buffer[:total_embs * emb_dim]
        track_embeddings_flat[:] = 0.0

        # Copy track embeddings - try vectorized stack first, fallback to loop
        offset = 0
        for track in tracks:
            hist = track.embedding_history
            n_hist = len(hist)
            if n_hist > 0:
                try:
                    # Vectorized: stack all embeddings at once
                    stacked = np.vstack(list(hist))
                    flat_len = n_hist * emb_dim
                    track_embeddings_flat[offset:offset + flat_len] = stacked.ravel()[:flat_len]
                    offset += flat_len
                except Exception:
                    # Fallback: copy one by one
                    for emb in hist:
                        emb_arr = np.asarray(emb, dtype=np.float32).ravel()
                        track_embeddings_flat[offset:offset + emb_dim] = emb_arr[:emb_dim]
                        offset += emb_dim
            else:
                offset += emb_dim

        # Method mapping
        method_map = {
            "last": 0,
            "average": 1,
            "weighted_average": 2,
            "best_match": 3,
            "median": 4
        }
        method = method_map.get(self.embedding_matching_method, 1)

        # Compute distances
        # OPTIMIZATION: Use fast matrix multiplication path for method=0 (last embedding)
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

        # === OPTIMIZATION: Vectorized fallback for dets without embeddings ===
        if len(dets_without_emb) > 0 and det_positions is not None and track_positions is not None:
            pos_dists = cdist(det_positions[dets_without_emb], track_positions)
            distances[dets_without_emb, :] = np.clip(pos_dists / self.max_distance, 0, 1)

        return distances

    def _compute_embedding_distances_sparse(
        self,
        detections: List[Detection],
        tracks: List[FastTrackState],
        has_embedding_mask: np.ndarray,
        sparse_det_indices: np.ndarray,
        sparse_track_indices: np.ndarray,
        det_positions: np.ndarray = None,
        track_positions: np.ndarray = None
    ) -> np.ndarray:
        """Compute embedding distances ONLY for sparse candidate pairs.

        FAST SPARSE VERSION - computes embedding distances only for spatially
        close detection-track pairs, not the full N×M matrix.

        For 1000 objects with 36K sparse pairs out of 880K total:
        - Full computation: ~100ms (880K aggregations)
        - Sparse computation: ~4ms (36K aggregations) = 24x speedup!

        Args:
            detections: List of Detection objects
            tracks: List of FastTrackState objects
            has_embedding_mask: Boolean mask indicating which detections have embeddings
            sparse_det_indices: Detection indices for sparse pairs
            sparse_track_indices: Track indices for sparse pairs
            det_positions: Detection positions [N_det, 2] (for fallback)
            track_positions: Track predicted positions [N_track, 2] (for fallback)

        Returns:
            Sparse embedding distance matrix [N_det, N_track]
            Non-sparse entries are set to 1.0 (max distance)
        """
        from scipy.spatial.distance import cdist

        n_dets = len(detections)
        n_tracks = len(tracks)

        # Get embedding dimension
        emb_dim = getattr(self.config, 'default_embedding_dimension', 128)
        dets_with_emb = np.where(has_embedding_mask)[0]
        if len(dets_with_emb) > 0:
            first_emb = detections[dets_with_emb[0]].embedding
            if first_emb is not None:
                emb_dim = len(first_emb)

        # === Extract detection embeddings (reuse buffer) ===
        if not hasattr(self, '_emb_dist_det_buffer') or self._emb_dist_det_buffer.shape[0] < n_dets or self._emb_dist_det_buffer.shape[1] != emb_dim:
            self._emb_dist_det_buffer = np.zeros((max(n_dets, 100), emb_dim), dtype=np.float32)

        det_embeddings = self._emb_dist_det_buffer[:n_dets, :emb_dim]
        det_embeddings[:] = 0.0

        for i in dets_with_emb:
            emb = detections[i].embedding
            if emb is not None:
                det_embeddings[i] = np.asarray(emb, dtype=np.float32)[:emb_dim]

        # Normalize
        norms = np.linalg.norm(det_embeddings, axis=1, keepdims=True)
        mask = (norms > 1e-8).flatten()
        det_embeddings[mask] /= norms[mask]

        # === Extract track embeddings (NO padding for empty tracks) ===
        track_counts = np.array([len(t.embedding_history) for t in tracks], dtype=np.int32)
        total_embs = int(np.sum(track_counts))  # Exact count, no padding

        if total_embs == 0:
            return np.ones((n_dets, n_tracks), dtype=np.float32)

        # Reusable buffer
        if not hasattr(self, '_emb_dist_track_buffer') or len(self._emb_dist_track_buffer) < total_embs * emb_dim:
            self._emb_dist_track_buffer = np.zeros(max(total_embs * emb_dim, 10000), dtype=np.float32)

        track_embeddings_flat = self._emb_dist_track_buffer[:total_embs * emb_dim]
        track_embeddings_flat[:] = 0.0

        # Copy track embeddings - NO padding for empty tracks
        offset = 0
        for track in tracks:
            hist = track.embedding_history
            n_hist = len(hist)
            if n_hist > 0:
                try:
                    stacked = np.vstack(list(hist))
                    flat_len = n_hist * emb_dim
                    track_embeddings_flat[offset:offset + flat_len] = stacked.ravel()[:flat_len]
                    offset += flat_len
                except Exception:
                    for emb in hist:
                        emb_arr = np.asarray(emb, dtype=np.float32).ravel()
                        track_embeddings_flat[offset:offset + emb_dim] = emb_arr[:emb_dim]
                        offset += emb_dim
            # NOTE: No "else: offset += emb_dim" - no padding for empty tracks

        # Method mapping
        method_map = {
            "last": 0,
            "average": 1,
            "weighted_average": 2,
            "best_match": 3,
            "median": 4
        }
        method = method_map.get(self.embedding_matching_method, 1)

        # === SPARSE COMPUTATION: Only aggregate for sparse pairs ===
        distances = compute_embedding_distances_sparse(
            det_embeddings, track_embeddings_flat, track_counts,
            sparse_det_indices, sparse_track_indices,
            method, n_dets, n_tracks
        )

        # Scale distances if scaler is available (VECTORIZED - not per-pair loop)
        if self.embedding_scaler:
            # Extract all sparse distances at once
            sparse_distances = distances[sparse_det_indices, sparse_track_indices]
            # Scale in batch (single call, not 4000+ individual calls)
            scaled_sparse = self.embedding_scaler.scale_distances(sparse_distances)
            self.embedding_scaler.update_statistics(sparse_distances)
            # Put back
            distances[sparse_det_indices, sparse_track_indices] = scaled_sparse

        # Fallback for dets without embeddings (VECTORIZED)
        dets_without_emb = np.where(~has_embedding_mask)[0]
        if len(dets_without_emb) > 0 and det_positions is not None and track_positions is not None:
            # Find sparse pairs where detection doesn't have embedding (numpy vectorized)
            fallback_mask = np.isin(sparse_det_indices, dets_without_emb)
            if np.any(fallback_mask):
                fb_det_idx = sparse_det_indices[fallback_mask]
                fb_track_idx = sparse_track_indices[fallback_mask]
                # Compute position distances vectorized
                pos_dists = np.sqrt(
                    (det_positions[fb_det_idx, 0] - track_positions[fb_track_idx, 0])**2 +
                    (det_positions[fb_det_idx, 1] - track_positions[fb_track_idx, 1])**2
                )
                distances[fb_det_idx, fb_track_idx] = np.clip(pos_dists / self.max_distance, 0, 1)

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
                    kalman_type=self.kalman_type,
                    velocity_damping=self.kalman_velocity_damping
                )

                # Set embedding parameters
                score_history_len = getattr(self.config, 'embedding_score_history_length', 5)
                track.set_embedding_params(
                    self.max_embeddings_per_track,
                    self.embedding_matching_method,
                    score_history_len
                )

                # Set uncertainty window if enabled
                if self.uncertainty_weight > 0.0:
                    track.set_uncertainty_window(self.uncertainty_window)

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

        OPTIMIZED VERSION:
        - Uses representative embeddings (single per track) instead of full history
        - Uses greedy assignment O(n²) instead of Hungarian O(n³)
        - Uses matrix multiplication for fast cosine similarity

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
        active_tracks = [
            track for track in self._tracks.values()
            if track.misses < self.reid_min_frames_lost
        ]

        if active_tracks:
            active_positions = np.array(
                [t.predicted_position for t in active_tracks], dtype=np.float32
            )
            det_positions_filter = np.array(
                [det.position for _, det in unmatched_dets_with_emb], dtype=np.float32
            )

            # Vectorized distance computation
            all_distances = np.linalg.norm(
                det_positions_filter[:, None, :] - active_positions[None, :, :], axis=2
            )
            min_distances = np.min(all_distances, axis=1)
            safe_mask = min_distances > self.max_distance

            unmatched_dets_with_emb = [
                item for item, is_safe in zip(unmatched_dets_with_emb, safe_mask) if is_safe
            ]

            if not unmatched_dets_with_emb:
                return

        n_dets = len(unmatched_dets_with_emb)
        n_tracks = len(lost_tracks_with_ids)

        if n_dets == 0 or n_tracks == 0:
            return

        # --- Use same embedding matching method as main assignment ---
        # Extract detection embeddings
        det_embeddings = np.array([
            det.embedding for _, det in unmatched_dets_with_emb
        ], dtype=np.float32)

        # Normalize detection embeddings
        det_norms = np.linalg.norm(det_embeddings, axis=1, keepdims=True)
        det_norms = np.where(det_norms > 1e-8, det_norms, 1.0)
        det_embeddings = det_embeddings / det_norms

        # Extract ALL track embeddings (full history, same as main assignment)
        emb_dim = det_embeddings.shape[1]
        track_counts = np.array([len(track.embedding_history) for _, track in lost_tracks_with_ids], dtype=np.int32)
        total_embs = int(np.sum(track_counts))

        track_embeddings_flat = np.zeros(total_embs * emb_dim, dtype=np.float32)
        offset = 0
        for _, track in lost_tracks_with_ids:
            hist = track.embedding_history
            n_hist = len(hist)
            if n_hist > 0:
                stacked = np.vstack(list(hist))
                # Normalize each embedding
                norms = np.linalg.norm(stacked, axis=1, keepdims=True)
                norms = np.where(norms > 1e-8, norms, 1.0)
                stacked = stacked / norms
                flat_len = n_hist * emb_dim
                track_embeddings_flat[offset:offset + flat_len] = stacked.ravel()[:flat_len]
                offset += flat_len

        # Use same matching method as main assignment (best_match, average, etc.)
        method_map = {"last": 0, "average": 1, "weighted_average": 2, "best_match": 3, "median": 4}
        method = method_map.get(self.embedding_matching_method, 3)  # default to best_match

        raw_embedding_distances = compute_embedding_distances_with_method(
            det_embeddings, track_embeddings_flat, track_counts, method
        )

        # Scale embedding distances
        if self.embedding_scaler:
            scaled_flat = self.embedding_scaler.scale_distances(raw_embedding_distances.flatten())
            scaled_embedding_distances = scaled_flat.reshape(n_dets, n_tracks)
        else:
            scaled_embedding_distances = raw_embedding_distances

        # Get position data
        det_positions = np.array([det.position for _, det in unmatched_dets_with_emb], dtype=np.float32)
        track_positions = np.array([t.predicted_position for _, t in lost_tracks_with_ids], dtype=np.float32)
        track_last_positions = np.array([t.last_detection_pos for _, t in lost_tracks_with_ids], dtype=np.float32)
        track_misses = np.array([t.misses for _, t in lost_tracks_with_ids], dtype=np.int32)

        # Compute ReID cost matrix
        reid_boost = getattr(self.config, 'reid_embedding_weight_boost', 1.5)
        reid_embedding_weight = min(self.embedding_weight * reid_boost, 0.95)

        reid_cost_matrix = compute_cost_matrix_vectorized(
            det_positions, track_positions, track_last_positions,
            track_misses, scaled_embedding_distances,
            reid_embedding_weight, self.reid_max_distance, do_embeddings=True,
            miss_threshold=3, embedding_threshold_adjustment=self.embedding_threshold_adjustment
        )

        # --- OPTIMIZATION: Use greedy assignment O(n²) instead of Hungarian O(n³) ---
        # ReID typically has few candidates, greedy is sufficient and much faster
        reid_matches, _, _ = numba_greedy_assignment(
            reid_cost_matrix, self.reid_max_distance
        )

        # Apply ReID matches
        for det_match_idx, track_match_idx in reid_matches:
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
        """Log timing information for debugging.

        Prints to stdout every frame when debug_timings is enabled.
        Uses print() instead of logger to ensure visibility regardless of logging config.
        """
        formatted = {k: f"{v*1000:.2f} ms" for k, v in self.timings.items()}
        print(f"[Frame {self._frame_count}] Timings: {formatted}")
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

        # Re-initialize array pools to initial size (preserves warmup allocation)
        # This avoids re-triggering allocation spikes on new videos
        initial_pool_size = getattr(self.config, 'initial_pool_size', 150)
        self._reusable_det_positions = np.empty((initial_pool_size, 2), dtype=np.float32)
        self._reusable_track_positions = np.empty((initial_pool_size, 2), dtype=np.float32)
        self._reusable_track_last_positions = np.empty((initial_pool_size, 2), dtype=np.float32)
        self._reusable_track_misses = np.empty(initial_pool_size, dtype=np.int32)
        self._max_dets_seen = initial_pool_size
        self._max_tracks_seen = initial_pool_size

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

    def get_statistics(self) -> dict:
        """Get tracker statistics for benchmarking and analysis.

        Returns:
            Dictionary containing:
                - next_id: Next track ID to be assigned (= total tracks created)
                - active_tracks: Number of currently active tracks
                - confirmed_tracks: Number of confirmed tracks
                - pending_detections: Number of pending detections
                - frame_count: Total frames processed
        """
        confirmed = sum(1 for t in self._tracks.values() if t.confirmed)
        return {
            "next_id": self._next_id,
            "active_tracks": len(self._tracks),
            "confirmed_tracks": confirmed,
            "pending_detections": len(self._pending_detections),
            "frame_count": self._frame_count
        }