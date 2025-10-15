"""
SwarmSort Track State Management

This module contains the core data structures for managing track states in the
SwarmSort tracking system. It includes both active track states and pending
detections that are waiting to be confirmed as tracks.

Classes:
    PendingDetection: Detection waiting to become a confirmed track
    FastTrackState: State representation of an active tracked object
"""

# ============================================================================
# STANDARD IMPORTS
# ============================================================================
from dataclasses import dataclass, field
from typing import Optional, Literal
from collections import deque
import numpy as np


# ============================================================================
# PENDING DETECTION CLASS
# ============================================================================
@dataclass
class PendingDetection:
    """Represents a detection waiting to become a track."""

    position: np.ndarray
    embedding: Optional[np.ndarray] = None
    bbox: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.float32)
    )
    class_id: Optional[int] = None
    confidence: float = 1.0
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    consecutive_frames: int = 1
    total_detections: int = 1
    average_position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))

    def __post_init__(self):
        self.average_position = self.position.copy()

    def update(self, position: np.ndarray, embedding: Optional[np.ndarray] = None,
               bbox: Optional[np.ndarray] = None, confidence: float = 1.0):
        """Update pending detection with new observation."""
        self.total_detections += 1
        self.consecutive_frames += 1

        # Update average position
        self.average_position = (
            self.average_position * (self.total_detections - 1) + position
        ) / self.total_detections

        # Update latest position
        self.position = position

        # Update embedding if provided
        if embedding is not None:
            self.embedding = embedding

        # Update bbox if provided
        if bbox is not None:
            self.bbox = bbox

        # Update confidence (keep max)
        self.confidence = max(self.confidence, confidence)

    def is_ready_for_promotion(
        self, min_consecutive: int, max_gap: int, current_frame: int
    ) -> bool:
        """Check if pending detection should become a track."""
        gap = current_frame - self.last_seen_frame
        return self.consecutive_frames >= min_consecutive and gap <= max_gap


# ============================================================================
# FAST TRACK STATE CLASS
# ============================================================================
@dataclass
class FastTrackState:
    """Enhanced track state with N-embedding history and kalman_type support."""

    id: int
    class_id: Optional[int] = None

    position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    predicted_position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    bbox: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))

    # Kalman state (for "simple" type)
    kalman_state: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))

    last_detection_pos: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    last_detection_frame: int = 0

    # Observation history for both types
    observation_history: deque = field(default_factory=lambda: deque(maxlen=5))
    observation_frames: deque = field(default_factory=lambda: deque(maxlen=5))

    # OC-SORT specific arrays (for "oc" type)
    observation_history_array: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.float32)
    )
    observation_frames_array: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )

    # Track type
    kalman_type: str = "simple"

    # Embedding freeze tracking
    embedding_frozen: bool = False
    last_safe_embedding: Optional[np.ndarray] = None

    # Embedding history with configurable size
    embedding_history: deque = field(default_factory=lambda: deque(maxlen=5))
    embedding_method: Literal["average", "best_match", "weighted_average", "last"] = "average"

    # Cache for average embedding
    _cached_avg_embedding: Optional[np.ndarray] = None
    _cache_valid: bool = False

    # Cache for multi-embedding computation
    _cached_representative_embedding: Optional[np.ndarray] = None
    _representative_cache_valid: bool = False

    # Keep for backward compatibility
    avg_embedding: Optional[np.ndarray] = None
    embedding_update_count: int = 0

    age: int = 0
    hits: int = 0
    misses: int = 0
    confirmed: bool = False
    detection_confidence: float = 0.0
    confidence_score: float = 0.5

    lost_frames: int = 0

    def __post_init__(self):
        self.kalman_state[:2] = self.position
        self.last_detection_pos = self.position.copy()
        self.predicted_position = self.position.copy()

        # Initialize OC-SORT arrays properly if needed
        if self.kalman_type == "oc":
            if self.observation_history_array.shape[0] == 0:
                self.observation_history_array = np.zeros((0, 2), dtype=np.float32)
            if self.observation_frames_array.shape[0] == 0:
                self.observation_frames_array = np.zeros(0, dtype=np.int32)

    def get_observation_prediction(self, current_frame: int, max_history: int = 5) -> np.ndarray:
        """Get observation-based prediction using recent detection history."""
        if len(self.observation_history) < 2:
            return self.predicted_position

        # Simple linear extrapolation from last two observations
        pos1 = self.observation_history[-2]
        pos2 = self.observation_history[-1]
        frame1 = self.observation_frames[-2]
        frame2 = self.observation_frames[-1]

        if frame2 == frame1:
            return pos2.copy()

        dt = current_frame - frame2
        velocity = (pos2 - pos1) / (frame2 - frame1)
        predicted = pos2 + velocity * dt

        return predicted.astype(np.float32)

    def update_observation_history(self, position: np.ndarray, frame: int):
        """Update observation history for observation-based prediction."""
        self.observation_history.append(position.copy())
        self.observation_frames.append(frame)

    def set_embedding_params(
            self,
            max_embeddings: int = 5,
            method: Literal["average", "best_match", "weighted_average", "last"] = "average",
    ):
        """Configure embedding storage parameters."""
        self.embedding_history = deque(maxlen=max_embeddings)
        self.embedding_method = method
        self._cache_valid = False

    def add_embedding(self, embedding: np.ndarray):
        """Add new embedding to history with safe normalization."""
        if self.embedding_frozen:
            return

        if embedding is not None:
            embedding = np.asarray(embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                # Only normalize if not already normalized
                if abs(norm - 1.0) > 0.01:
                    normalized_emb = embedding / (norm + 1e-8)
                else:
                    normalized_emb = embedding

                if self.last_safe_embedding is None:
                    self.last_safe_embedding = normalized_emb.copy()

                self.embedding_history.append(normalized_emb.copy())
                self.embedding_update_count += 1

                self._cache_valid = False
                self._representative_cache_valid = False

                self._update_avg_embedding()

    def _update_avg_embedding(self):
        """Update avg_embedding with caching."""
        if len(self.embedding_history) > 0:
            if self.embedding_method == "average":
                if not self._cache_valid:
                    self._cached_avg_embedding = np.mean(list(self.embedding_history), axis=0)
                    self._cache_valid = True
                self.avg_embedding = self._cached_avg_embedding
            elif self.embedding_method == "weighted_average":
                weights = np.arange(1, len(self.embedding_history) + 1, dtype=np.float32)
                weights = weights / weights.sum()
                self.avg_embedding = np.average(list(self.embedding_history), axis=0, weights=weights)
            else:
                self.avg_embedding = self.embedding_history[-1]

    def get_representative_embedding(self) -> Optional[np.ndarray]:
        """Get representative embedding based on configured method."""
        if len(self.embedding_history) == 0:
            return None

        if self.embedding_method == "last":
            return self.embedding_history[-1]
        elif self.embedding_method == "average":
            return np.mean(list(self.embedding_history), axis=0)
        elif self.embedding_method == "weighted_average":
            weights = np.arange(1, len(self.embedding_history) + 1, dtype=np.float32)
            weights = weights / weights.sum()
            return np.average(list(self.embedding_history), axis=0, weights=weights)
        else:
            return self.embedding_history[-1]

    def freeze_embeddings(self):
        """Freeze embeddings when collision detected."""
        if not self.embedding_frozen:
            self.embedding_frozen = True
            if len(self.embedding_history) > 0 and self.last_safe_embedding is None:
                self.last_safe_embedding = self.embedding_history[-1].copy()

    def unfreeze_embeddings(self):
        """Unfreeze embeddings when collision resolved."""
        if self.embedding_frozen:
            self.embedding_frozen = False

    def update_with_detection(
        self,
        position: np.ndarray,
        embedding: Optional[np.ndarray] = None,
        bbox: Optional[np.ndarray] = None,
        frame: int = 0,
        det_conf: float = 0.0,
        is_reid: bool = False,
    ):
        """Update track state with new detection."""
        # Import here to avoid circular dependency
        from .kalman_filters import simple_kalman_update, simple_kalman_predict

        self.position = position.astype(np.float32)
        self.last_detection_pos = position.copy()
        self.last_detection_frame = frame
        self.detection_confidence = det_conf

        self.update_observation_history(position, frame)

        if self.kalman_type == "simple":
            self.kalman_state = simple_kalman_update(self.kalman_state, position, is_reid=is_reid)
            self.velocity = self.kalman_state[2:].copy()
            self.predicted_position = self.kalman_state[:2].copy()
        elif self.kalman_type == "oc":
            new_observation = position.reshape(1, 2).astype(np.float32)
            new_frame = np.array([frame], dtype=np.int32)

            if len(self.observation_history_array) == 0:
                self.observation_history_array = new_observation
                self.observation_frames_array = new_frame
            else:
                self.observation_history_array = np.vstack(
                    [self.observation_history_array, new_observation]
                )
                self.observation_frames_array = np.append(
                    self.observation_frames_array, frame
                )

            if len(self.observation_history_array) >= 2:
                dt = self.observation_frames_array[-1] - self.observation_frames_array[-2]
                if dt > 0:
                    self.velocity[0] = (self.observation_history_array[-1, 0] -
                                       self.observation_history_array[-2, 0]) / dt
                    self.velocity[1] = (self.observation_history_array[-1, 1] -
                                       self.observation_history_array[-2, 1]) / dt

        if embedding is not None:
            self.add_embedding(embedding)

        if bbox is not None:
            self.bbox = np.asarray(bbox, dtype=np.float32)

        self.hits += 1
        self.age += 1
        self.misses = 0
        self.lost_frames = 0

    def predict_only(self, current_frame: int = None):
        """Prediction step - behavior depends on kalman_type."""
        # Import here to avoid circular dependency
        from .kalman_filters import simple_kalman_predict, oc_sort_predict

        if self.kalman_type == "simple":
            self.kalman_state = simple_kalman_predict(self.kalman_state)
            self.velocity = self.kalman_state[2:].copy()
            self.predicted_position = self.kalman_state[:2].copy()
        elif self.kalman_type == "oc":
            if hasattr(self, 'observation_history_array') and len(self.observation_history_array) > 0:
                frame_to_use = current_frame if current_frame is not None else (
                    self.observation_frames_array[-1] + 1 if len(self.observation_frames_array) > 0 else 0
                )
                pred_state = oc_sort_predict(
                    self.observation_history_array,
                    self.observation_frames_array,
                    frame_to_use
                )
                self.predicted_position = pred_state[:2].copy()
            else:
                self.predicted_position = self.position.copy()

        self.age += 1
        self.misses += 1
        self.lost_frames += 1

    def get_predicted_position(self, current_frame: int) -> np.ndarray:
        """Get predicted position based on kalman_type."""
        # Import here to avoid circular dependency
        from .kalman_filters import oc_sort_predict

        if self.kalman_type == "simple":
            return self.predicted_position
        elif self.kalman_type == "oc":
            pred_state = oc_sort_predict(
                self.observation_history_array,
                self.observation_frames_array,
                current_frame
            )
            return pred_state[:2]
        else:
            return self.predicted_position

    def add_embedding_fast(self, embedding: np.ndarray, pre_normalized: bool = True):
        """Fast embedding addition without expensive checks."""
        if self.embedding_frozen or embedding is None:
            return

        if not pre_normalized:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        if self.last_safe_embedding is None:
            self.last_safe_embedding = embedding.copy()

        if len(self.embedding_history) > 0:
            self._cache_valid = False
            self._representative_cache_valid = False

        self.embedding_history.append(embedding)
        self.embedding_update_count += 1

        self._update_avg_embedding()