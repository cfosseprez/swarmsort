#!/usr/bin/env python3
"""
SwarmTracker Pipeline Integration Adapter

This module provides seamless integration between the standalone SwarmSort implementation
and the SwarmTracker pipeline, ensuring 100% compatibility with the old integration.

Features:
- Maintains identical interface to old SwarmSort integration
- Preserves all tracking algorithm behavior
- Compatible with SwarmTracker's BaseTrackerModel interface
- Returns proper TrackingResult format
- Supports all original configuration options
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, NamedTuple
from dataclasses import dataclass, asdict

from .core import SwarmSortTracker as StandaloneSwarmSortTracker, SwarmSortConfig
from .data_classes import Detection as StandaloneDetection, TrackedObject

# ============================================================================
# SWARMTRACKER PIPELINE COMPATIBILITY
# ============================================================================


class TrackingResult(NamedTuple):
    """Standardized return type for tracking operations - SwarmTracker compatible."""

    tracked_objects: List[Any]
    bounding_boxes: Optional[np.ndarray] = None
    result_image: Optional[np.ndarray] = None


class Detection:
    """SwarmTracker compatible Detection class"""

    def __init__(self, position=None, bbox=None, embedding=None, confidence=1.0, **kwargs):
        self.position = np.asarray(position, dtype=np.float32) if position is not None else None
        self.bbox = np.asarray(bbox, dtype=np.float32) if bbox is not None else None
        self.embedding = np.asarray(embedding, dtype=np.float32) if embedding is not None else None
        self.confidence = float(confidence)

        # Support additional attributes for compatibility
        for key, value in kwargs.items():
            setattr(self, key, value)

    def points(self):
        """Compatibility property - same as position"""
        return self.position if self.position is not None else np.array([0, 0], dtype=np.float32)


def create_tracked_object_fast(track_id, position, bbox=None, confidence=1.0, **kwargs):
    """Create tracked object compatible with SwarmTracker pipeline"""
    obj = TrackedObject(id=track_id, position=position, bbox=bbox, confidence=confidence)

    # Add additional attributes for compatibility
    for key, value in kwargs.items():
        setattr(obj, key, value)

    return obj


# ============================================================================
# PIPELINE ADAPTER CLASSES
# ============================================================================


class RawTrackerSwarmSORT:
    """Raw SwarmSORT tracker - tracking only, compatible with SwarmTracker pipeline"""

    def __init__(
        self,
        tracker_config: Optional[Union[Dict, SwarmSortConfig]] = None,
        runtime_config: Optional[Any] = None,
        yaml_config_location: Optional[str] = None,
        **kwargs,
    ):
        """Initialize with SwarmTracker pipeline compatibility"""

        # Handle config merging like the old implementation
        if isinstance(tracker_config, dict):
            # Handle backward compatibility for parameter rename
            tracker_config = dict(tracker_config)  # Make a copy
            if "use_embeddings" in tracker_config and "do_embeddings" not in tracker_config:
                tracker_config["do_embeddings"] = tracker_config.pop("use_embeddings")
            config = SwarmSortConfig(**tracker_config)
        elif isinstance(tracker_config, SwarmSortConfig):
            config = tracker_config
        else:
            config = SwarmSortConfig()

        # Apply runtime config overrides if provided
        if runtime_config is not None:
            if hasattr(runtime_config, "get_modified_params"):
                modified_params = runtime_config.get_modified_params()
                for key, value in modified_params.items():
                    if hasattr(config, key):
                        setattr(config, key, value)

        # Initialize the standalone tracker
        self.swarmsort_tracker = StandaloneSwarmSortTracker(config)

        # Store config for compatibility
        self.config = config

        # Timing compatibility
        self.timings = {}

    def track(
        self, detections, frame: np.ndarray, verbose: bool = True, **kwargs
    ) -> TrackingResult:
        """
        Track objects using SwarmSORT - SwarmTracker pipeline compatible

        Args:
            detections: Detection results (can be Detection objects, lists, etc.)
            frame: Input frame (for compatibility - not used in tracking)
            verbose: Enable verbose output
            **kwargs: Additional arguments for compatibility

        Returns:
            TrackingResult: Standardized tracking result
        """

        # Convert detections to standalone format
        standalone_detections = self._convert_detections(detections)

        # Perform tracking
        tracked_objects = self.swarmsort_tracker.update(standalone_detections)

        # Convert back to SwarmTracker compatible format
        tracked_detections = []
        bboxes = []

        for obj in tracked_objects:
            # Create compatible tracked object
            tracked_obj = create_tracked_object_fast(
                track_id=obj.id,
                position=obj.position,
                bbox=obj.bbox,
                confidence=obj.confidence,
                age=obj.age,
                matched=obj.matched,
            )
            tracked_detections.append(tracked_obj)

            # Collect bboxes
            if obj.bbox is not None:
                bboxes.append(obj.bbox)

        # Convert bboxes to array
        bbox = np.array(bboxes) if bboxes else None

        # Store timings for compatibility
        if hasattr(self.swarmsort_tracker, "timings"):
            self.timings = self.swarmsort_tracker.timings

        return TrackingResult(
            tracked_objects=tracked_detections,
            bounding_boxes=bbox,
            result_image=None,  # No visualization by default
        )

    def _convert_detections(self, detections) -> List[StandaloneDetection]:
        """Convert various detection formats to standalone Detection objects"""
        if not detections:
            return []

        standalone_detections = []

        for det in detections:
            if hasattr(det, "position") or hasattr(det, "points"):
                # Already a Detection-like object
                position = getattr(det, "position", None)
                if position is None and hasattr(det, "points"):
                    position = det.points() if callable(det.points) else det.points

                bbox = getattr(det, "bbox", None)
                embedding = getattr(det, "embedding", None)
                confidence = getattr(det, "confidence", 1.0)

            elif isinstance(det, (list, tuple, np.ndarray)):
                # Assume [x1, y1, x2, y2, confidence, class_id] format
                det_array = np.asarray(det)
                if len(det_array) >= 4:
                    bbox = det_array[:4].astype(np.float32)
                    position = np.array(
                        [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32
                    )
                    confidence = float(det_array[4]) if len(det_array) > 4 else 1.0
                    embedding = None
                else:
                    continue  # Skip invalid detection
            else:
                continue  # Skip unknown format

            # Create standalone detection
            standalone_det = StandaloneDetection(
                position=position, bbox=bbox, embedding=embedding, confidence=confidence
            )
            standalone_detections.append(standalone_det)

        return standalone_detections

    def get_statistics(self) -> Dict:
        """Get tracker statistics - compatibility method"""
        return self.swarmsort_tracker.get_statistics()

    def reset(self):
        """Reset tracker state - compatibility method"""
        self.swarmsort_tracker.reset()


# ============================================================================
# FACTORY FUNCTIONS FOR SWARMTRACKER INTEGRATION
# ============================================================================


def create_swarmsort_tracker(runtime_config=None, yaml_config_location=None):
    """
    Create SwarmSORT tracker with full config handling - SwarmTracker compatible

    This function maintains the exact same interface as the old integration,
    ensuring seamless compatibility with the SwarmTracker pipeline.
    """
    try:
        from .core import SwarmSortTracker
        from .config import SwarmSortConfig
        from .embeddings import is_gpu_available
        
        # Handle config merging exactly like the old implementation
        config_dict = {}
        
        if runtime_config is not None and hasattr(runtime_config, "get_modified_params"):
            modified_params = runtime_config.get_modified_params()
            config_dict.update(modified_params)
        
        # Skip the SwarmSort config merging system that's failing
        # Just use the runtime config directly from SwarmTracker
        if runtime_config is not None:
            if hasattr(runtime_config, 'to_dict'):
                # It's a Config object
                config_dict = runtime_config.to_dict()
            elif hasattr(runtime_config, '__dict__'):
                # It's an object with attributes
                config_dict = {k: v for k, v in runtime_config.__dict__.items() if not k.startswith('_')}
        
        print(f"Using config from SwarmTracker: {config_dict}")
        
        # Determine GPU availability and embedding type
        use_gpu = is_gpu_available() and config_dict.get('use_gpu', True)
        embedding_type = config_dict.get('embedding_function', 'cupytexture')
        
        # Validate and adjust embedding type for GPU availability
        if not use_gpu and embedding_type in ['cupytexture', 'mega_cupytexture', 'cupytexture_color']:
            print(f"GPU not available, switching from {embedding_type} to histogram")
            embedding_type = 'histogram'
        elif use_gpu and embedding_type == 'histogram':
            # Upgrade to GPU embedding if available
            embedding_type = 'cupytexture'
            print("Upgrading to GPU-accelerated cupytexture embedding")

        # Create SwarmSort config with all available parameters from SwarmTracker
        swarmsort_config = SwarmSortConfig(
            # Use values from config with proper defaults
            max_distance=float(config_dict.get('max_distance', 80.0)),
            max_track_age=int(config_dict.get('max_track_age', 20)),
            detection_conf_threshold=float(config_dict.get('detection_conf_threshold', 0.0)),
            do_embeddings=bool(config_dict.get('do_embeddings', True)),
            embedding_weight=float(config_dict.get('embedding_weight', 0.25)),
            reid_enabled=bool(config_dict.get('reid_enabled', True)),
            reid_max_distance=float(config_dict.get('reid_max_distance', 120.0)),
            reid_embedding_threshold=float(config_dict.get('reid_embedding_threshold', 0.3)),
            min_consecutive_detections=int(config_dict.get('min_consecutive_detections', 5)),
            max_detection_gap=int(config_dict.get('max_detection_gap', 2)),
            pending_detection_distance=float(config_dict.get('pending_detection_distance', 80.0)),
            use_probabilistic_costs=bool(config_dict.get('use_probabilistic_costs', False)),
            # Add embedding function configuration
            embedding_function=config_dict.get('embedding_function', 'cupytexture'),
        )
        
        print(f"Created SwarmSortConfig: max_distance={swarmsort_config.max_distance}, "
              f"embedding_weight={swarmsort_config.embedding_weight}, "
              f"reid_enabled={swarmsort_config.reid_enabled}")

        # Create and return the new SwarmSort package tracker
        return SwarmSortTracker(
            config=swarmsort_config,
            embedding_type=embedding_type,
            use_gpu=use_gpu
        )
        
    except ImportError as e:
        # Fallback to old adapter approach if SwarmSort package components fail
        print(f"Warning: Fallback to adapter mode: {e}")
        config_dict = {}
        if runtime_config is not None and hasattr(runtime_config, "get_modified_params"):
            modified_params = runtime_config.get_modified_params()
            config_dict.update(modified_params)

        return RawTrackerSwarmSORT(
            tracker_config=config_dict,
            runtime_config=runtime_config,
            yaml_config_location=yaml_config_location,
        )


# ============================================================================
# LEGACY COMPATIBILITY ALIASES
# ============================================================================

# For backward compatibility with old imports
FastMultiHypothesisTracker = RawTrackerSwarmSORT


def create_tracker(*args, **kwargs):
    """Legacy compatibility function"""
    return create_swarmsort_tracker(*args, **kwargs)
