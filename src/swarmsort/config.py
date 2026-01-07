"""
SwarmSort Configuration System

This module provides the configuration classes and utilities for SwarmSort.
The main SwarmSortConfig class contains all tunable parameters for the tracking
algorithm, with sensible defaults and validation.

Classes:
    BaseConfig: Base configuration class with YAML loading capabilities
    SwarmSortConfig: Main configuration class for SwarmSort tracker parameters
"""
# ============================================================================
# STANDARD IMPORTS
# ============================================================================
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Literal, Type, TypeVar, Tuple, List
from pathlib import Path
import yaml
from loguru import logger

T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    """Base configuration class with YAML loading capabilities."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                result[key] = value
        return result

    @classmethod
    def from_dict(cls: Type[T], config_dict: Dict[str, Any]) -> T:
        """Create configuration from dictionary."""
        # Filter out unknown fields
        valid_fields = {
            f.name for f in cls.__dataclass_fields__.values() if not f.name.startswith("_")
        }
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)

    @classmethod
    def from_yaml(cls: Type[T], yaml_path: str) -> T:
        """Load configuration from YAML file."""
        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    def save_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def merge_config_with_priority(default_config: Any,
                             runtime_config: Optional[Dict] = None,
                             yaml_config_location=None,
                             yaml_config_name: str = "swarmsort_config.yaml",
                             verbose_parameters=False):
    """
    Merge configuration from multiple sources with priority:
    runtime_config > yaml_config > default_config
    """
    # Start with hardcoded defaults
    config = default_config()

    # Override with YAML config (yaml > hardcoded)
    try:
        loaded_config = load_local_yaml_config(yaml_config_name, caller_file=yaml_config_location)
        for k, v in loaded_config.items():
            if hasattr(config, k):
                setattr(config, k, v)
            else:
                logger.warning(f"Ignoring unknown YAML config key: {k}")
    except Exception as e:
        logger.debug(f"Could not load config from {yaml_config_name}, using defaults: {e}")

    # Override with runtime config (runtime > yaml > hardcoded)
    if runtime_config:
        for k, v in runtime_config.items():
            if hasattr(config, k):
                setattr(config, k, v)
            else:
                logger.warning(f"Ignoring unknown runtime config key: {k}")

    # Recompute auto-parameters after all overrides are applied
    # This handles cases where YAML or runtime sets values to -1.0 (sentinel)
    # or changes max_distance which affects dependent parameters
    if hasattr(config, '_compute_auto_parameters'):
        config._compute_auto_parameters()

    if verbose_parameters:
        config_dict = config.to_dict()
        lines = [f"     {key} = {value}" for key, value in config_dict.items()]
        param_str = f"Creating {default_config.__name__} with parameters:\n" + "\n".join(lines)
        logger.info(param_str)

    return config


def load_local_yaml_config(yaml_filename: str, caller_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Try to load a local YAML configuration file.
    
    Searches in:
    - Same directory as caller module (if provided)  
    - Same directory as this module
    - Parent directories
    """
    search_paths = []

    # Add caller's directory first if provided
    if caller_file:
        caller_path = Path(caller_file).parent
        search_paths.extend([
            caller_path / yaml_filename,
            caller_path.parent / yaml_filename,
        ])

    # Add original search paths
    search_paths.extend([
        Path(__file__).parent / "data" / yaml_filename,  # swarmsort/data/
        Path(__file__).parent / yaml_filename,            # swarmsort/
        Path(__file__).parent.parent / yaml_filename,     # src/
    ])

    for yaml_path in search_paths:
        try:
            if yaml_path.exists():
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    local_config = yaml.safe_load(f)

                if isinstance(local_config, dict):
                    print(f"Successfully loaded local YAML config from: {yaml_path}")
                    return local_config
                else:
                    print(f"Warning: Local config file {yaml_path} does not contain a valid dictionary")

        except yaml.YAMLError as e:
            print(f"Warning: Error parsing YAML file {yaml_path}: {e}")
        except Exception as e:
            print(f"Warning: Error reading local config file {yaml_path}: {e}")

    print("Local config could not be loaded from any location, using hardcoded defaults")
    return {}


@dataclass
class SwarmSortConfig(BaseConfig):
    """
    Configuration for SwarmSort multi-object tracker.

    SwarmSort is a real-time tracking algorithm that follows multiple objects across video frames.
    It combines motion prediction (where objects will be) with appearance matching (what objects
    look like) to maintain consistent identity assignments even when objects temporarily disappear
    or cross paths.

    This configuration controls all aspects of the tracking behavior. Default values are tuned
    for general tracking scenarios but should be adjusted based on your specific use case.

    Quick Start Guide:
    ------------------
    For tracking fast-moving objects: increase max_distance
    For crowded scenes: decrease max_distance, increase min_consecutive_detections
    For high-quality detections: increase detection_conf_threshold
    For appearance-based tracking: ensure do_embeddings=True, adjust embedding_weight
    """

    # ============================================================================
    # CORE TRACKING PARAMETERS
    # ============================================================================

    max_distance: float = 80.0
    """Maximum pixel distance for matching detections to tracks.

    This is THE most important parameter. It defines how far an object can move between
    frames and still be considered the same object.

    - INCREASE (200-300) for: fast-moving objects, low frame rates, zoomed-out views
    - DECREASE (50-100) for: slow objects, high frame rates, crowded scenes, zoomed-in views

    Rule of thumb: Set to the maximum pixels an object typically moves per frame.

    Example: If objects move 100 pixels/frame, set to 150 to handle variations.
    """

    detection_conf_threshold: float = 0
    """Minimum confidence score to accept a detection (0.0 to 1.0).

    Filters out low-confidence detections from your detector (YOLO, etc.) before tracking.
    This is a GLOBAL filter applied to ALL detections.

    - 0.0 = Accept all detections (default - let tracker handle noise)
    - 0.3-0.5 = Moderate filtering (good for decent detectors)
    - 0.7-0.9 = Aggressive filtering (only very confident detections)

    Note: Set based on your detector's confidence distribution. Check histogram of
    confidence scores to pick appropriate threshold.
    """

    max_track_age: int = 30
    """Maximum frames a track can exist without any detection before deletion.

    Controls how long to keep tracking an object after it disappears (occlusion, 
    leaving frame, detection failure).

    - 10-20 frames = Quick deletion (good for fast-changing scenes)
    - 30-50 frames = Balanced (default, handles brief occlusions)
    - 60-120 frames = Persistent (good for long occlusions, more false positives)

    At 30 FPS: 30 frames = 1 second of occlusion tolerance.
    """

    # ============================================================================
    # MOTION PREDICTION SETTINGS
    # ============================================================================

    kalman_type: Literal["simple", "oc"] = "simple"
    """Motion prediction algorithm type.

    - "simple": Classic Kalman filter with constant velocity model
      Pros: Smooth, predictable motion, good for linear movement
      Cons: Overshoots on sudden stops/turns

    - "oc": OC-SORT style observation-centric (no prediction during occlusion)
      Pros: Better for erratic motion, no drift during occlusion
      Cons: Less smooth, may lose fast-moving occluded objects

    Use "simple" for vehicles/pedestrians, "oc" for animals/sports/erratic motion.
    """

    # ============================================================================
    # UNCERTAINTY & COLLISION HANDLING
    # ============================================================================

    uncertainty_weight: float = 0.0
    """Weight for uncertainty penalties in cost computation (0.0 to 1.0).

    Adds a cost penalty based on recent miss ratio, making unreliable tracks
    less likely to steal matches from reliable ones.

    Formula: cost = base_cost * (1 + uncertainty_weight * recent_miss_ratio)

    - 0.0 = Disabled (no overhead, treat all tracks equally)
    - 0.2-0.3 = Light uncertainty (small preference for reliable tracks)
    - 0.5-0.7 = Strong uncertainty (heavily favor reliable tracks)
    """

    uncertainty_window: int = 10
    """Number of recent frames to consider for uncertainty calculation.

    Tracks the hit/miss ratio over the last N frames.

    - 5-10 = Short memory (quick adaptation)
    - 10-20 = Medium memory (balanced)
    """

    local_density_radius: float = -1.0  # Default: max_distance / 2
    """Radius in pixels to check for nearby tracks (for density computation).

    Used to detect crowded areas where ID switches are more likely.
    When many tracks are within this radius, the tracker becomes more conservative.

    Typically set to max_distance/2 or max_distance/3.
    """
    # Note: -1.0 means auto-compute from max_distance in __post_init__

    collision_freeze_embeddings: bool = True
    """Freeze appearance updates when objects are too close together.

    When enabled, prevents appearance features from being corrupted during
    collisions or severe occlusions. The tracker "remembers" what each object
    looked like before they got too close.

    - True = Safer tracking in crowds (recommended)
    - False = Always update appearance (may cause ID switches in crowds)
    """

    embedding_freeze_density: int = 1
    """Number of nearby tracks to trigger embedding freeze.

    When this many (or more) tracks are within local_density_radius,
    stop updating appearance features to prevent corruption.

    - 1 = Freeze when ANY other track is nearby (most conservative)
    - 2-3 = Freeze only in moderately crowded areas
    - 5+ = Freeze only in very dense crowds
    """
    deduplication_distance: float = 0.0
    """Minimum distance between detections to be considered separate objects.

    Detections closer than this are merged to prevent duplicate tracks.
    Should be set based on your object size and detector characteristics.

    - 5-10 pixels = Tight deduplication for small objects
    - 10-20 pixels = Standard deduplication (default)
    - 20-50 pixels = Loose deduplication for large objects
    """

    collision_safety_distance: float = -1.0
    """Distance at which to consider objects in collision for embedding freezing.

    When objects are closer than this, their embeddings stop updating to prevent
    appearance confusion during occlusion. This is typically larger than
    deduplication_distance but smaller than max_distance.

    Set to -1.0 for auto-compute (max_distance * 0.4).

    - 20-30 pixels = Early freeze for safety
    - 30-50 pixels = Standard collision distance
    - 50+ pixels = Late freeze, allows more updates
    """
    # Note: -1.0 means auto-compute from max_distance in __post_init__

    # ============================================================================
    # APPEARANCE MATCHING (EMBEDDINGS) SETTINGS
    # ============================================================================

    do_embeddings: bool = True
    """Enable appearance-based matching using visual features.

    When True: Uses both motion AND appearance to match objects
    When False: Uses only motion/position (faster but less accurate)

    Embeddings help when:
    - Objects look different (clothing, color, size)
    - Objects cross paths or occlude each other
    - Multiple similar objects need to be distinguished

    Disable only if all objects look identical or for maximum speed.
    """

    embedding_weight: float = 1.0
    """Relative importance of appearance penalty (0.0 to ~2.0).

    Controls how much embedding distance adds to the position-based cost:
    Cost = position_distance + embedding_weight × embedding_distance × max_distance

    - 0.0 = Position only (appearance ignored even if do_embeddings=True)
    - 0.5 = Embedding adds up to 50% of max_distance as penalty
    - 1.0 = Embedding adds up to 100% of max_distance as penalty (default)
    - 2.0 = Embedding adds up to 200% of max_distance as penalty

    Increase for distinct-looking objects, decrease for similar-looking objects.
    """

    embedding_threshold_adjustment: float = 1.0
    """Threshold adjustment factor for embedding contribution to assignment gating.

    With additive cost formula, total cost can exceed max_distance. This parameter
    adjusts the effective assignment threshold to account for embedding contribution:

    effective_max_distance = max_distance × (1 + embedding_weight × embedding_threshold_adjustment)

    Example with max_distance=80, embedding_weight=1.0, embedding_threshold_adjustment=1.0:
    - effective_max_distance = 80 × (1 + 1.0 × 1.0) = 160
    - Allows costs up to 160 (80 position + 80 max embedding contribution)

    - 0.0 = No threshold adjustment (may reject valid matches with high embedding cost)
    - 1.0 = Full adjustment for embedding contribution (default, recommended)
    - 0.5 = Partial adjustment (more strict on appearance)
    """

    max_embeddings_per_track: int = 15
    """Maximum number of appearance samples to store per track.

    Each track keeps a history of appearance features to handle appearance changes
    (rotation, lighting, partial occlusion).

    - 1 = Only most recent (fast, no appearance history)
    - 5-10 = Short history (good for stable appearance)
    - 15-30 = Long history (handles appearance variation, uses more memory)

    More samples = better appearance model but slower matching.
    """

    embedding_function: str = "cupytexture"
    """Algorithm for extracting appearance features.

    Built-in options (require CuPy for GPU):
    - "cupytexture": GPU-accelerated texture features (fast, good quality)
    - "cupytexture_color": Texture + color histogram features
    - "cupytexture_mega": Advanced features (slower, best quality)
    - "cupyshape": Shape-based features

    External embeddings:
    - "external" or None: Use embeddings provided in Detection.embedding field.
      This allows using custom models (MobileNet, ResNet, etc.).
      You must attach embeddings to each Detection before calling tracker.update().

    Example with external embeddings:
        detection = Detection(
            position=np.array([x, y]),
            bbox=np.array([x1, y1, x2, y2]),
            embedding=my_model.extract_features(crop)  # Your custom embedding
        )
    """

    embedding_matching_method: Literal["average", "weighted_average", "best_match", "median"] = "median"
    """How to match current appearance against track history.

    - "average": Compare against mean of all stored appearances
      Simple, stable, but slow to adapt to changes

    - "weighted_average": Recent appearances count more
      Good balance of stability and adaptability (recommended)

    - "best_match": Find best matching historical appearance
      Handles appearance changes well but more computationally expensive

    - "median": Median distance to all stored appearances
      More robust to outliers than average, good for noisy embeddings
    """

    store_embedding_scores: bool = False
    """Store embedding match scores for each track update.

    When True: Stores the cosine similarity scores from recent matches.
    Useful for debugging, visualization, and confidence estimation.

    This has minimal performance impact as scores are computed anyway during matching.
    """

    embedding_score_history_length: int = 5
    """Number of recent embedding match scores to keep per track.

    Only used when store_embedding_scores=True.
    Higher values give more stable averages but use slightly more memory.
    """

    # ============================================================================
    # ASSIGNMENT ALGORITHM SETTINGS
    # ============================================================================

    sparse_computation_threshold: int = 300
    """Use the many object optimized sparse computation"""

    use_probabilistic_costs: bool = False
    """Use probabilistic fusion for cost computation.

    - False: Simple distance-based costs (faster, usually sufficient)
    - True: Bayesian fusion considering uncertainties (more sophisticated)

    Probabilistic costs can help in complex scenarios but add computation overhead.
    Most users should keep this False.
    """

    assignment_strategy: Literal["hungarian", "greedy", "hybrid"] = "hungarian"
    """Algorithm for matching detections to tracks.

    - "hungarian": Globally optimal assignment (best accuracy, O(n³) complexity)
      Use when: <50 objects, accuracy is critical

    - "greedy": Fast local assignment (good accuracy, O(n²) complexity)  
      Use when: >100 objects, speed is critical

    - "hybrid": Greedy for obvious matches, Hungarian for ambiguous (recommended)
      Best balance of speed and accuracy for most scenarios
    """

    greedy_threshold: float = -1.0  # Default: max_distance / 5
    """Distance threshold for confident matches in hybrid/greedy mode.

    Matches closer than this distance are assigned immediately without
    considering other possibilities. Should be much smaller than max_distance.

    - max_distance/6 = Very conservative (only super obvious matches)
    - max_distance/4 = Balanced (default)
    - max_distance/2 = Aggressive (may cause errors in crowds)
    """
    # Note: -1.0 means auto-compute from max_distance in __post_init__

    greedy_confidence_boost: float = 1.0
    """Confidence multiplier for greedy matches (not currently used)."""

    hungarian_fallback_threshold: float = 1.0
    """Multiplier for max_distance in Hungarian phase of hybrid assignment.

    After greedy assignment, Hungarian considers matches up to
    max_distance * hungarian_fallback_threshold.

    - 1.0 = Same threshold (consistent behavior)
    - 1.5 = More permissive in Hungarian (catches difficult matches)
    - 0.8 = More restrictive (fewer but more confident matches)
    """

    # ============================================================================
    # RE-IDENTIFICATION (REID) SETTINGS
    # ============================================================================

    reid_enabled: bool = True
    """Enable re-identification of lost tracks.

    ReID attempts to re-connect tracks that were lost (due to occlusion,
    detection failure) with new detections using appearance matching.

    Helps maintain consistent IDs through temporary disappearances.
    Disable if objects never reappear or appearance is unreliable.
    """

    reid_max_distance: float = -1.0  # Default: max_distance * 1.5
    """Maximum distance for re-identification matching.

    Lost tracks can be matched to detections up to this distance away.
    Larger than max_distance because objects may have moved far during occlusion.

    - max_distance * 1.0 = Conservative (only nearby reappearances)
    - max_distance * 1.5 = Balanced (default)
    - max_distance * 2.0 = Aggressive (may cause false re-identifications)
    """
    # Note: -1.0 means auto-compute from max_distance in __post_init__

    reid_embedding_threshold: float = 0.5
    """Maximum embedding distance for ReID (0.0 to 1.0, lower = stricter).

    Lost tracks are only matched if appearance similarity is better than this.

    - 0.1-0.2 = Very strict (only nearly identical appearance)
    - 0.3-0.4 = Balanced (default, some appearance change allowed)
    - 0.5-0.7 = Permissive (allows significant appearance change)

    Lower values = fewer but more accurate re-identifications.
    """

    reid_min_frames_lost: int = 3
    """Minimum frames a track must be lost before attempting ReID.

    Prevents immediate re-identification that can cause ID swaps.
    Allows the tracker to wait and see if the object reappears naturally.

    - 0 = Immediate ReID (may cause ID swaps)
    - 1 = Wait one frame (minimal delay)
    - 2-3 = Wait a few frames (recommended, prevents most ID swaps)
    - 5+ = Conservative delay (very safe but may miss quick reappearances)
    """

    reid_embedding_weight_boost: float = 1.5
    """Multiplier for embedding_weight during ReID (1.0 to 2.0).

    During ReID, appearance matching is more important than spatial matching
    because lost tracks may have moved far. This boosts embedding weight.

    Final ReID embedding weight = min(embedding_weight * boost, 0.95)

    - 1.0 = Same as normal matching (no boost)
    - 1.5 = 50% more emphasis on appearance (default, recommended)
    - 2.0 = Double emphasis on appearance (very appearance-focused)
    """

    # ============================================================================
    # TRACK INITIALIZATION SETTINGS
    # ============================================================================

    init_conf_threshold: float = 0.0
    """Minimum confidence to start tracking an object (0.0 to 1.0).

    This is a SECOND filter specifically for track creation.
    Detections must pass BOTH detection_conf_threshold (to be processed)
    AND init_conf_threshold (to create new tracks).

    - 0.0 = Create tracks from any detection (after min_consecutive_detections)
    - 0.3-0.5 = Only track reasonably confident detections
    - 0.7+ = Only track very confident detections

    Use higher values to reduce false positive tracks.
    """

    min_consecutive_detections: int = 10
    """Number of consecutive detections required to confirm a track.

    New tracks start as "tentative" and become "confirmed" after being
    detected in this many consecutive frames. Prevents tracking noise/artifacts.

    - 1 = Immediate tracking (fast response, more false positives)
    - 2-3 = Quick confirmation (balanced)
    - 5-10 = Careful confirmation (slow response, very few false positives)

    Increase in noisy environments or with unreliable detectors.
    """

    max_detection_gap: int = 1
    """Maximum frame gap allowed during track initialization.

    While building confidence for a new track, detections can be missing
    for up to this many frames without resetting the count.

    - 0 = No gaps allowed (very strict)
    - 1-2 = Allow brief gaps (default, handles detector flickering)
    - 3-5 = Allow longer gaps (for difficult detection scenarios)
    """

    pending_detection_distance: float = -1.0  # Default: max_distance
    """Maximum distance to associate detections during initialization.

    Before a track is confirmed, detections must be within this distance
    to be considered the same pending object.

    Usually same as max_distance, but can be smaller for stricter initialization.
    """
    # Note: -1.0 means auto-compute from max_distance in __post_init__

    # ============================================================================
    # EMBEDDING SCALING SETTINGS (ADVANCED)
    # ============================================================================

    embedding_scaling_method: str = "min_robustmax"
    """Method for normalizing embedding distances to [0,1] range.

    Raw embedding distances have arbitrary scale. These methods normalize
    them to match the scale of spatial distances:

    - "min_robustmax": Asymmetric scaling with true min and robust max
    - "robust_minmax": Symmetric robust scaling using percentiles
    - Others: Various statistical methods (see embedding_scaler.py)

    Most users should keep the default.
    """

    embedding_scaling_update_rate: float = 0.05
    """Learning rate for updating scaling statistics (0.0 to 1.0).

    Controls how quickly the scaler adapts to changing embedding distributions.

    - 0.01 = Very slow adaptation (stable but slow to adjust)
    - 0.05 = Moderate adaptation (default)
    - 0.1-0.2 = Fast adaptation (responsive but may be unstable)
    """

    embedding_scaling_min_samples: int = 200
    """Minimum samples before embedding scaling is activated.

    The scaler needs to see enough embedding distances to compute reliable
    statistics. Before this, a simple fallback scaling is used.

    - 100-200 = Quick activation (may be less accurate initially)
    - 500-1000 = Careful activation (more accurate but takes longer)
    """

    default_embedding_dimension: int = 128
    """Default embedding dimension when embeddings are not yet available.

    Used for array pre-allocation and Numba JIT compilation warmup.
    Should match your embedding extractor's output dimension.

    - 128 = Standard (most re-identification networks)
    - 256 = High-dimensional embeddings
    - 512+ = Very high-dimensional (more memory overhead)
    """

    # ============================================================================
    # DEBUG SETTINGS
    # ============================================================================

    debug_embeddings: bool = False
    """Print detailed embedding information for debugging.

    Outputs embedding statistics, distances, and scaling information.
    Useful for tuning embedding parameters but very verbose.
    """

    plot_embeddings: bool = False
    """Generate embedding visualization plots (requires matplotlib).

    Creates visual representations of embedding space and distances.
    Helpful for understanding embedding behavior but slows tracking.
    """

    debug_timings: bool = True
    """Print detailed timing information for performance analysis.

    Shows time spent in each component of the tracking pipeline.
    Use to identify performance bottlenecks.
    """

    # ============================================================================
    # ADVANCED KALMAN FILTER TUNING
    # ============================================================================

    kalman_velocity_damping: float = 0.95
    """Velocity damping factor applied during Kalman prediction (0.0 to 1.0).

    Controls how quickly velocity decays when no measurement is available.
    - 1.0 = No damping (velocity persists indefinitely)
    - 0.95 = Slight damping (default, handles noise)
    - 0.8 = Strong damping (velocity decays quickly)

    Lower values help prevent overshoot during occlusions.
    """

    kalman_update_alpha: float = 0.7
    """Measurement weight in Kalman update (0.0 to 1.0).

    Controls the blend between prediction and measurement:
    - new_pos = alpha * measurement + (1 - alpha) * prediction

    - 1.0 = Trust measurement completely (no filtering)
    - 0.7 = Balanced (default, good for noisy detections)
    - 0.5 = Heavy filtering (smoother but slower response)
    """

    kalman_velocity_weight: float = 0.2
    """Weight for velocity consistency term in OC-SORT cost (0.0 to 1.0).

    Controls how much velocity consistency affects the assignment cost.
    Higher values prefer matches that maintain velocity direction.
    """

    prediction_miss_threshold: int = 3
    """Number of missed frames before using last position instead of prediction.

    When a track has missed this many frames, the tracker uses its last
    known position rather than the predicted position for matching.
    This prevents prediction drift during extended occlusions.
    """

    # ============================================================================
    # ADVANCED PROBABILISTIC COST TUNING
    # ============================================================================

    mahalanobis_normalization: float = 5.0
    """Normalization factor for Mahalanobis distance (used in probabilistic mode).

    Scales Mahalanobis distance to be comparable with max_distance.
    LOWER values make the probabilistic gating more permissive.

    The normalized distance = mahal_dist * mahalanobis_normalization.
    - 3-5 = Permissive (good for new tracks with low velocity estimates)
    - 5-10 = Moderate (default, balances precision and tolerance)
    - 10-20 = Strict (requires good velocity model)
    """

    probabilistic_gating_multiplier: float = 1.5
    """Multiplier for max_distance in probabilistic gating.

    Euclidean pre-filter threshold = max_distance * this value.
    Allows Mahalanobis gating to consider matches beyond strict max_distance.
    """

    time_covariance_inflation: float = 0.2
    """Rate at which covariance inflates per missed frame (0.0 to 1.0).

    Each missed frame, covariance is multiplied by (1 + this * misses).
    Higher values make uncertainty grow faster during occlusion.
    """

    base_position_variance: float = 25.0
    """Base position variance for covariance estimation in probabilistic mode.

    This is the minimum uncertainty in position regardless of velocity.
    Higher values make the tracker more tolerant of position errors.

    - 5-10 = Tight uncertainty (good for high-quality detections)
    - 15-25 = Moderate uncertainty (default, good for typical tracking)
    - 25-50 = Loose uncertainty (for noisy detections or fast motion)
    """

    velocity_variance_scale: float = 0.5
    """Scale factor for velocity contribution to position uncertainty.

    In probabilistic mode, tracks moving faster have higher uncertainty
    in their predicted position. This controls how much velocity affects
    the covariance: variance_along_motion = base + scale * velocity_magnitude.

    - 0.0 = Velocity doesn't affect uncertainty
    - 1.0-2.0 = Moderate velocity effect (default)
    - 3.0+ = Strong velocity effect
    """

    velocity_isotropic_threshold: float = 2.0
    """Velocity threshold (pixels/frame) below which covariance is isotropic.

    When a track's velocity magnitude is below this threshold, the covariance
    is circular (same uncertainty in all directions). Above this threshold,
    the covariance becomes elliptical (more uncertainty in direction of motion).

    - 0.1 = Very slow tracks only get isotropic covariance (default)
    - 1.0 = Tracks moving < 1 pixel/frame get isotropic covariance
    - 5.0 = Only very fast tracks get anisotropic covariance
    """

    singular_covariance_threshold: float = 1e-6
    """Threshold for detecting singular (degenerate) covariance matrices.

    If the covariance matrix determinant is below this value, the tracker
    falls back to Euclidean distance instead of Mahalanobis distance.
    This prevents numerical instability from matrix inversion.

    - 1e-6 = Default, works for most cases
    - 1e-8 = More aggressive, may have numerical issues
    - 1e-4 = More conservative, falls back to Euclidean more often
    """

    def __post_init__(self):
        """
        Automatically compute dependent parameters if they are not set.
        This ensures that if `max_distance` is changed, related parameters
        are updated accordingly.
        """
        self._compute_auto_parameters()

    def _compute_auto_parameters(self):
        """Compute auto-parameters based on max_distance.

        This method can be called after config merging to recompute
        dependent parameters when max_distance changes.

        Parameters with value -1.0 are sentinel values meaning "auto-compute".
        """
        if self.local_density_radius == -1.0:
            self.local_density_radius = self.max_distance / 3
        if self.greedy_threshold == -1.0:
            self.greedy_threshold = self.max_distance / 3
        if self.reid_max_distance == -1.0:
            self.reid_max_distance = self.max_distance * 1.5
        if self.pending_detection_distance == -1.0:
            self.pending_detection_distance = self.max_distance
        if self.collision_safety_distance == -1.0:
            self.collision_safety_distance = self.max_distance * 0.25


    def validate(self) -> None:
        """Validate configuration parameters.

        Checks that all parameters are within valid ranges and compatible
        with each other. Raises ValueError if configuration is invalid.
        """
        # Core parameters
        if self.max_distance <= 0:
            raise ValueError("max_distance must be positive")

        if not 0 <= self.init_conf_threshold <= 1:
            raise ValueError("init_conf_threshold must be between 0 and 1")

        if self.max_track_age < 1:
            raise ValueError("max_track_age must be at least 1")

        if not 0 <= self.detection_conf_threshold <= 1:
            raise ValueError("detection_conf_threshold must be between 0 and 1")

        # Embedding parameters
        if self.do_embeddings and self.embedding_weight < 0:
            raise ValueError("embedding_weight must be non-negative")

        if self.embedding_weight > 10.0:
            raise ValueError("embedding_weight must be <= 10.0 (very high values cause numerical issues)")

        if self.max_embeddings_per_track < 1:
            raise ValueError("max_embeddings_per_track must be at least 1")

        if self.embedding_matching_method not in ["average", "weighted_average", "best_match", "median"]:
            raise ValueError("Invalid embedding_matching_method")

        if self.min_consecutive_detections < 1:
            raise ValueError("min_consecutive_detections must be at least 1")

        # Distance parameter relationships
        if self.deduplication_distance >= self.collision_safety_distance:
            raise ValueError(
                f"deduplication_distance ({self.deduplication_distance}) must be < "
                f"collision_safety_distance ({self.collision_safety_distance})"
            )

        if self.collision_safety_distance >= self.max_distance:
            raise ValueError(
                f"collision_safety_distance ({self.collision_safety_distance}) must be < "
                f"max_distance ({self.max_distance})"
            )

        if self.local_density_radius > self.max_distance:
            raise ValueError(
                f"local_density_radius ({self.local_density_radius}) must be <= "
                f"max_distance ({self.max_distance})"
            )

        if self.greedy_threshold >= self.max_distance:
            raise ValueError(
                f"greedy_threshold ({self.greedy_threshold}) must be < "
                f"max_distance ({self.max_distance})"
            )

        # ReID parameters
        if not 0 <= self.reid_embedding_threshold <= 1:
            raise ValueError("reid_embedding_threshold must be between 0 and 1")

        # Kalman parameters
        if not 0 <= self.kalman_velocity_damping <= 1:
            raise ValueError("kalman_velocity_damping must be between 0 and 1")

        if not 0 <= self.kalman_update_alpha <= 1:
            raise ValueError("kalman_update_alpha must be between 0 and 1")

        if not 0 <= self.kalman_velocity_weight <= 1:
            raise ValueError("kalman_velocity_weight must be between 0 and 1")

        if self.prediction_miss_threshold < 1:
            raise ValueError("prediction_miss_threshold must be at least 1")

        # Probabilistic cost parameters
        if self.mahalanobis_normalization <= 0:
            raise ValueError("mahalanobis_normalization must be positive")

        if self.probabilistic_gating_multiplier <= 0:
            raise ValueError("probabilistic_gating_multiplier must be positive")

        if not 0 <= self.time_covariance_inflation <= 1:
            raise ValueError("time_covariance_inflation must be between 0 and 1")

        # Assignment strategy
        if self.assignment_strategy not in ["hungarian", "greedy", "hybrid"]:
            raise ValueError(f"Invalid assignment_strategy: {self.assignment_strategy}")

        if self.kalman_type not in ["simple", "oc"]:
            raise ValueError(f"Invalid kalman_type: {self.kalman_type}")


def load_config(config_path: Optional[str] = None) -> SwarmSortConfig:
    """
    Load SwarmSort configuration from a yaml file.

    Args:
        config_path: Path to YAML configuration file. If None, uses defaults.

    Returns:
        SwarmSortConfig instance
    """
    if config_path is None:
        config = SwarmSortConfig()
    else:
        config = SwarmSortConfig.from_yaml(config_path)

    config.validate()
    return config


def validate_config(config: SwarmSortConfig) -> Tuple[bool, List[str]]:
    """
    Validate configuration at runtime with detailed error messages.

    This is a convenience function for runtime validation that collects
    all errors instead of raising on the first one.

    Args:
        config: SwarmSortConfig instance to validate

    Returns:
        Tuple of (is_valid, list_of_error_messages)

    Example:
        >>> config = SwarmSortConfig(max_distance=-10)
        >>> is_valid, errors = validate_config(config)
        >>> if not is_valid:
        ...     for error in errors:
        ...         print(f"Config error: {error}")
    """
    errors: List[str] = []

    # Core parameters
    if config.max_distance <= 0:
        errors.append("max_distance must be positive")

    if not 0 <= config.init_conf_threshold <= 1:
        errors.append("init_conf_threshold must be between 0 and 1")

    if config.max_track_age < 1:
        errors.append("max_track_age must be at least 1")

    if not 0 <= config.detection_conf_threshold <= 1:
        errors.append("detection_conf_threshold must be between 0 and 1")

    # Embedding parameters
    if config.do_embeddings and config.embedding_weight < 0:
        errors.append("embedding_weight must be non-negative when embeddings enabled")

    if config.embedding_weight > 10.0:
        errors.append("embedding_weight must be <= 10.0 (very high values cause numerical issues)")

    if config.max_embeddings_per_track < 1:
        errors.append("max_embeddings_per_track must be at least 1")

    if config.embedding_matching_method not in ["average", "weighted_average", "best_match", "median"]:
        errors.append(f"Invalid embedding_matching_method: {config.embedding_matching_method}")

    if config.min_consecutive_detections < 1:
        errors.append("min_consecutive_detections must be at least 1")

    # Distance parameter relationships
    if config.deduplication_distance >= config.collision_safety_distance:
        errors.append(
            f"deduplication_distance ({config.deduplication_distance}) must be < "
            f"collision_safety_distance ({config.collision_safety_distance})"
        )

    if config.collision_safety_distance >= config.max_distance:
        errors.append(
            f"collision_safety_distance ({config.collision_safety_distance}) must be < "
            f"max_distance ({config.max_distance})"
        )

    if config.local_density_radius > config.max_distance:
        errors.append(
            f"local_density_radius ({config.local_density_radius}) must be <= "
            f"max_distance ({config.max_distance})"
        )

    if config.greedy_threshold >= config.max_distance:
        errors.append(
            f"greedy_threshold ({config.greedy_threshold}) must be < "
            f"max_distance ({config.max_distance})"
        )

    # ReID parameters
    if not 0 <= config.reid_embedding_threshold <= 1:
        errors.append("reid_embedding_threshold must be between 0 and 1")

    # Kalman parameters
    if not 0 <= config.kalman_velocity_damping <= 1:
        errors.append("kalman_velocity_damping must be between 0 and 1")

    if not 0 <= config.kalman_update_alpha <= 1:
        errors.append("kalman_update_alpha must be between 0 and 1")

    if not 0 <= config.kalman_velocity_weight <= 1:
        errors.append("kalman_velocity_weight must be between 0 and 1")

    if config.prediction_miss_threshold < 1:
        errors.append("prediction_miss_threshold must be at least 1")

    # Probabilistic cost parameters
    if config.mahalanobis_normalization <= 0:
        errors.append("mahalanobis_normalization must be positive")

    if config.probabilistic_gating_multiplier <= 0:
        errors.append("probabilistic_gating_multiplier must be positive")

    if not 0 <= config.time_covariance_inflation <= 1:
        errors.append("time_covariance_inflation must be between 0 and 1")

    # Assignment strategy
    if config.assignment_strategy not in ["hungarian", "greedy", "hybrid"]:
        errors.append(f"Invalid assignment_strategy: {config.assignment_strategy}")

    if config.kalman_type not in ["simple", "oc"]:
        errors.append(f"Invalid kalman_type: {config.kalman_type}")

    return len(errors) == 0, errors
