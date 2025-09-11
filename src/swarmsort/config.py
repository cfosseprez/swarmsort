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
    Configuration for SwarmSort tracker.

    This contains all parameters needed to configure the tracking algorithm,
    including distance thresholds, embedding parameters, and behavior settings.
    """

    # Core tracking parameters
    max_distance: float = 150.0  # Maximum distance for association
    detection_conf_threshold: float = 0  # Minimum confidence for detections (general filter)
    max_track_age: int = 30  # Maximum frames a track can exist without detection before deletion
    
    # Kalman filter type
    kalman_type: Literal["simple", "oc"] = "simple"  # Kalman filter type: simple or OC-SORT style
    
    # Uncertainty-based cost system for smart collision handling
    uncertainty_weight: float = 0.15  # Weight for uncertainty penalties (0 = disabled, typical 0.2-0.5)
    local_density_radius: float = max_distance/2  # Radius for computing local track density
    
    # Embedding freeze (simplified density-based)
    collision_freeze_embeddings: bool = True  # Freeze embedding updates in dense areas
    embedding_freeze_density: int = 1  # Freeze embeddings when ≥N tracks within radius

    # Embedding parameters
    do_embeddings: bool = True  # Whether to compute and use embedding features
    embedding_weight: float = 1  # Weight for embedding similarity in cost function
    max_embeddings_per_track: int = 15  # Maximum embeddings stored per track
    embedding_function: str = "cupytexture"
    embedding_matching_method: Literal[
        "average", "weighted_average", "best_match"
    ] = "weighted_average"

    # Cost computation method
    use_probabilistic_costs: bool = False  # Use probabilistic fusion vs simple costs

    # Assignment strategy parameters
    assignment_strategy: Literal["hungarian", "greedy", "hybrid"] = "hybrid"
    greedy_threshold: float = max_distance/4  # Distance threshold for greedy assignment
    greedy_confidence_boost: float = 1  # Confidence multiplier for greedy matches
    hungarian_fallback_threshold: float = 1  # Multiplier of max_distance for Hungarian fallback

    # Re-identification (ReID) parameters
    reid_enabled: bool = True  # Enable re-identification of lost tracks
    reid_max_distance: float = max_distance*1.5  # Maximum distance for ReID
    reid_embedding_threshold: float = 0.3  # Embedding threshold for ReID (lower more permissive)

    # Track initialization parameters
    init_conf_threshold: float = 0.  # Minimum confidence for track initialization (initialization filter)
    min_consecutive_detections: int = 3  # Minimum consecutive detections to create track
    max_detection_gap: int = 2  # Maximum gap between detections for same pending track
    pending_detection_distance: float = max_distance  # Distance threshold for pending detection matching

    # Embedding distance scaling
    embedding_scaling_method: str = "min_robustmax"  # "robust_minmax" "min_robustmax" Method for scaling embedding distances
    embedding_scaling_update_rate: float = 0.05  # Update rate for online scaling statistics
    embedding_scaling_min_samples: int = 200  # Minimum samples before scaling is active

    # Debug options
    debug_embeddings: bool = False  # Enable embedding debugging output
    plot_embeddings: bool = False  # Generate embedding visualization plots
    debug_timings: bool = False  # Enable timing debug output

    def validate(self) -> None:
        """Validate configuration parameters."""
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
