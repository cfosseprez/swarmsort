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
from typing import Dict, Any, Optional, Literal, Type, TypeVar
from pathlib import Path
import yaml

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
                print(f"Warning: Ignoring unknown YAML config key: {k}")
    except Exception as e:
        print(f"Warning: Could not load config from {yaml_config_name}, using defaults: {e}")

    # Override with runtime config (runtime > yaml > hardcoded)
    if runtime_config:
        for k, v in runtime_config.items():
            if hasattr(config, k):
                setattr(config, k, v)
            else:
                print(f"Warning: Ignoring unknown runtime config key: {k}")

    if verbose_parameters:
        config_dict = config.to_dict()
        lines = [f"     {key} = {value}" for key, value in config_dict.items()]
        param_str = f"Creating {default_config.__name__} with parameters:\n" + "\n".join(lines)
        print(param_str)

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
        Path(__file__).parent / yaml_filename,
        Path(__file__).parent.parent / yaml_filename,
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

    max_distance: float = 150.0
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

    uncertainty_weight: float = 0.33
    """Weight for uncertainty penalties in cost computation (0.0 to 1.0).

    Makes the tracker less confident about tracks in difficult situations:
    - Tracks that have missed recent detections
    - Tracks in crowded areas
    - Tracks with poor detection history

    - 0.0 = Disabled (treat all tracks equally)
    - 0.2-0.3 = Light uncertainty (small preference for reliable tracks)
    - 0.5-0.7 = Strong uncertainty (heavily favor reliable tracks)

    Helps prevent ID switches in challenging scenarios.
    """

    local_density_radius: float = max_distance/2  # max_distance/2
    """Radius in pixels to check for nearby tracks (for density computation).

    Used to detect crowded areas where ID switches are more likely.
    When many tracks are within this radius, the tracker becomes more conservative.

    Typically set to max_distance/2 or max_distance/3.
    """

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
    """Relative importance of appearance vs motion (0.0 to ~2.0).

    Controls the balance between position-based and appearance-based matching:

    - 0.0 = Position only (appearance ignored even if do_embeddings=True)
    - 0.5 = Position is 2x more important than appearance
    - 1.0 = Equal weight (default, balanced approach)
    - 2.0 = Appearance is 2x more important than position

    Increase for distinct-looking objects, decrease for similar-looking objects.
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

    Options depend on your installation:
    - "cupytexture": GPU-accelerated texture features (fast, good quality)
    - "cupytexture_color": Texture + color histogram features
    - "mega_cupytexture": Advanced features (slower, best quality)

    GPU options require CuPy. Falls back to CPU if unavailable.
    """

    embedding_matching_method: Literal["average", "weighted_average", "best_match"] = "weighted_average"
    """How to match current appearance against track history.

    - "average": Compare against mean of all stored appearances
      Simple, stable, but slow to adapt to changes

    - "weighted_average": Recent appearances count more
      Good balance of stability and adaptability (recommended)

    - "best_match": Find best matching historical appearance
      Handles appearance changes well but more computationally expensive
    """

    # ============================================================================
    # ASSIGNMENT ALGORITHM SETTINGS
    # ============================================================================

    sparse_computation_threshold: int = 300
    """Use the many ovject optimized sparse computation"""

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

    greedy_threshold: float = max_distance/5  # max_distance/4
    """Distance threshold for confident matches in hybrid/greedy mode.

    Matches closer than this distance are assigned immediately without
    considering other possibilities. Should be much smaller than max_distance.

    - max_distance/6 = Very conservative (only super obvious matches)
    - max_distance/4 = Balanced (default)
    - max_distance/2 = Aggressive (may cause errors in crowds)
    """

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

    reid_max_distance: float = 225.0  # max_distance*1.5
    """Maximum distance for re-identification matching.

    Lost tracks can be matched to detections up to this distance away.
    Larger than max_distance because objects may have moved far during occlusion.

    - max_distance * 1.0 = Conservative (only nearby reappearances)
    - max_distance * 1.5 = Balanced (default)
    - max_distance * 2.0 = Aggressive (may cause false re-identifications)
    """

    reid_embedding_threshold: float = 0.5
    """Maximum embedding distance for ReID (0.0 to 1.0, lower = stricter).

    Lost tracks are only matched if appearance similarity is better than this.

    - 0.1-0.2 = Very strict (only nearly identical appearance)
    - 0.3-0.4 = Balanced (default, some appearance change allowed)
    - 0.5-0.7 = Permissive (allows significant appearance change)

    Lower values = fewer but more accurate re-identifications.
    """

    reid_min_frames_lost: int = 2
    """Minimum frames a track must be lost before attempting ReID.

    Prevents immediate re-identification that can cause ID swaps.
    Allows the tracker to wait and see if the object reappears naturally.

    - 0 = Immediate ReID (may cause ID swaps)
    - 1 = Wait one frame (minimal delay)
    - 2-3 = Wait a few frames (recommended, prevents most ID swaps)
    - 5+ = Conservative delay (very safe but may miss quick reappearances)
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

    min_consecutive_detections: int = 6
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

    pending_detection_distance: float = max_distance  # Same as max_distance
    """Maximum distance to associate detections during initialization.

    Before a track is confirmed, detections must be within this distance
    to be considered the same pending object.

    Usually same as max_distance, but can be smaller for stricter initialization.
    """

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

    debug_timings: bool = False
    """Print detailed timing information for performance analysis.

    Shows time spent in each component of the tracking pipeline.
    Use to identify performance bottlenecks.
    """

    def validate(self) -> None:
        """Validate configuration parameters.

        Checks that all parameters are within valid ranges and compatible
        with each other. Raises ValueError if configuration is invalid.
        """
        if self.max_distance <= 0:
            raise ValueError("max_distance must be positive")

        if not 0 <= self.init_conf_threshold <= 1:
            raise ValueError("init_conf_threshold must be between 0 and 1")

        if self.max_track_age < 1:
            raise ValueError("max_track_age must be at least 1")

        if not 0 <= self.detection_conf_threshold <= 1:
            raise ValueError("detection_conf_threshold must be between 0 and 1")

        if self.do_embeddings and self.embedding_weight < 0:
            raise ValueError("embedding_weight must be non-negative")

        if self.max_embeddings_per_track < 1:
            raise ValueError("max_embeddings_per_track must be at least 1")

        if self.embedding_matching_method not in ["average", "weighted_average", "best_match"]:
            raise ValueError("Invalid embedding_matching_method")

        if self.min_consecutive_detections < 1:
            raise ValueError("min_consecutive_detections must be at least 1")


def load_config(config_path: Optional[str] = None) -> SwarmSortConfig:
    """
    Load SwarmSort configuration.

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
