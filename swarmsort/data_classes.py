import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Deque
from collections import deque


@dataclass
class Detection:
    """A standalone data class for a single detection."""
    position: np.ndarray
    confidence: float
    bbox: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    class_id: Optional[int] = None
    id: Optional[str] = None


@dataclass
class TrackedObject:
    """A standalone data class for a single tracked object."""
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


@dataclass
class PendingDetection:
    """Internal class to manage detections that are not yet tracks."""
    position: np.ndarray
    confidence: float
    first_seen_frame: int
    last_seen_frame: int
    consecutive_frames: int = 1
    total_detections: int = 1
    embedding: Optional[np.ndarray] = None
    bbox: Optional[np.ndarray] = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    average_position: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.average_position is None:
            self.average_position = self.position.copy()

    def __eq__(self, other):
        """
        Custom equality check to handle NumPy arrays for list.remove().
        """
        if not isinstance(other, PendingDetection):
            return NotImplemented

        # Compare non-array attributes first for a quick exit
        if (self.first_seen_frame != other.first_seen_frame or
            self.last_seen_frame != other.last_seen_frame or
            self.consecutive_frames != other.consecutive_frames or
            self.total_detections != other.total_detections or
            not np.isclose(self.confidence, other.confidence)):
            return False

        # Compare numpy arrays safely
        if not np.array_equal(self.position, other.position):
            return False

        # Handle optional bbox
        if (self.bbox is None) != (other.bbox is None) or \
           (self.bbox is not None and not np.array_equal(self.bbox, other.bbox)):
            return False

        # Handle optional embedding
        if (self.embedding is None) != (other.embedding is None) or \
           (self.embedding is not None and not np.array_equal(self.embedding, other.embedding)):
            return False

        return True