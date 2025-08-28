"""
Configuration system for SwarmSort.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Literal, Type, TypeVar
from pathlib import Path
import yaml

T = TypeVar('T', bound='BaseConfig')


@dataclass
class BaseConfig:
    """Base configuration class with YAML loading capabilities."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls: Type[T], config_dict: Dict[str, Any]) -> T:
        """Create configuration from dictionary."""
        # Filter out unknown fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values() if not f.name.startswith('_')}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)
    
    @classmethod
    def from_yaml(cls: Type[T], yaml_path: str) -> T:
        """Load configuration from YAML file."""
        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    def save_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


@dataclass
class SwarmSortConfig(BaseConfig):
    """
    Configuration for SwarmSort tracker.
    
    This contains all parameters needed to configure the tracking algorithm,
    including distance thresholds, embedding parameters, and behavior settings.
    """
    
    # Core tracking parameters
    max_distance: float = 80.0  # Maximum distance for association
    high_score_threshold: float = 0.8  # Threshold for high-confidence detections
    max_age: int = 20  # Maximum frames to keep a track alive without detections
    detection_conf_threshold: float = 0.3  # Minimum confidence for detections
    
    # Embedding parameters
    use_embeddings: bool = True  # Whether to use embedding features
    embedding_weight: float = 0.3  # Weight for embedding similarity in cost function
    max_embeddings_per_track: int = 15  # Maximum embeddings stored per track
    embedding_matching_method: Literal['average', 'weighted_average', 'best_match'] = 'weighted_average'
    
    # Cost computation method
    use_probabilistic_costs: bool = False  # Use probabilistic fusion vs simple costs
    
    # Re-identification (ReID) parameters
    reid_enabled: bool = True  # Enable re-identification of lost tracks
    reid_max_distance: float = 150.0  # Maximum distance for ReID
    reid_embedding_threshold: float = 0.4  # Embedding threshold for ReID
    reid_max_frames: int = 10  # Maximum frames to keep lost tracks for ReID
    
    # Track initialization parameters
    min_consecutive_detections: int = 3  # Minimum consecutive detections to create track
    max_detection_gap: int = 2  # Maximum gap between detections for same pending track
    pending_detection_distance: float = 50.0  # Distance threshold for pending detection matching
    
    # Duplicate detection removal
    duplicate_detection_threshold: float = 25.0  # Distance threshold for duplicate removal
    
    # Embedding distance scaling
    embedding_scaling_method: str = 'min_robustmax'  # Method for scaling embedding distances
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
        
        if not 0 <= self.high_score_threshold <= 1:
            raise ValueError("high_score_threshold must be between 0 and 1")
        
        if self.max_age < 1:
            raise ValueError("max_age must be at least 1")
        
        if not 0 <= self.detection_conf_threshold <= 1:
            raise ValueError("detection_conf_threshold must be between 0 and 1")
        
        if self.use_embeddings and self.embedding_weight < 0:
            raise ValueError("embedding_weight must be non-negative")
        
        if self.max_embeddings_per_track < 1:
            raise ValueError("max_embeddings_per_track must be at least 1")
        
        if self.embedding_matching_method not in ['average', 'weighted_average', 'best_match']:
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