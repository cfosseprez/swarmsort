"""
SwarmSort Data Classes

This module defines the core data structures used throughout SwarmSort for
representing detections and tracked objects.

Classes:
    Detection: Input detection with position, confidence, and optional features
    TrackedObject: Output tracked object with full tracking state information

Note:
    PendingDetection is defined in track_state.py to keep internal tracking
    state management consolidated in one place.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Detection:
    """Input detection representing a single object observation.

    A Detection represents a single object detection from a computer vision model,
    containing its position, confidence score, and optional features like bounding box
    and embedding vector for appearance-based matching.

    Attributes:
        position (np.ndarray): 2D position [x, y] in world/image coordinates
        confidence (float): Detection confidence score, typically in range [0, 1]
        bbox (np.ndarray, optional): Bounding box as [x1, y1, x2, y2]
        embedding (np.ndarray, optional): Feature vector for appearance matching
        class_id (int, optional): Object class identifier
        id (str, optional): Unique detection identifier

    Example:
        >>> import numpy as np
        >>> detection = Detection(
        ...     position=np.array([100.0, 150.0], dtype=np.float32),
        ...     confidence=0.95,
        ...     bbox=np.array([90, 140, 110, 160], dtype=np.float32)
        ... )
    """

    position: np.ndarray
    confidence: float
    bbox: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    class_id: Optional[int] = None
    id: Optional[str] = None


@dataclass
class TrackedObject:
    """Output tracked object representing a track's current state.

    A TrackedObject represents the current state of a tracked object, including
    its position, motion, confidence, and tracking statistics. This is the main
    output type returned by SwarmSortTracker.update().

    Attributes:
        id (int): Unique track identifier, assigned when track is created
        position (np.ndarray): Current 2D position [x, y] estimate
        velocity (np.ndarray): Current velocity [vx, vy] estimate from Kalman filter
        confidence (float): Most recent detection confidence associated with this track
        age (int): Number of frames since track was created
        hits (int): Total number of successful detection associations
        time_since_update (int): Frames since last successful detection association
        state (int): Track state (0: Tentative, 1: Confirmed, 2: Deleted)
        bbox (np.ndarray, optional): Most recent bounding box [x1, y1, x2, y2]
        class_id (int, optional): Object class identifier

    Example:
        >>> # TrackedObject is typically created by the tracker
        >>> tracked_objects = tracker.update(detections)
        >>> for obj in tracked_objects:
        ...     print(f"Track {obj.id} at position {obj.position}")
        ...     print(f"  Velocity: {obj.velocity}")
        ...     print(f"  Age: {obj.age}, Hits: {obj.hits}")
    """

    id: int
    position: np.ndarray
    velocity: np.ndarray
    confidence: float
    age: int
    hits: int
    time_since_update: int
    state: int  # e.g., 0: Tentative, 1: Confirmed, 2: Deleted
    bbox: Optional[np.ndarray] = None
    class_id: Optional[int] = None
    predicted_position: Optional[np.ndarray] = None  # Predicted position for visualization
    embedding_score: Optional[float] = None  # Average embedding match score (cosine similarity)
