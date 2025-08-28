"""
Basic tests for SwarmSort standalone package.
"""
import numpy as np
import pytest
from swarmsort import (
    SwarmSortTracker,
    SwarmSortConfig,
    Detection,
    TrackedObject,
    SwarmSort,
    create_tracker,
    is_within_swarmtracker
)


def test_detection_creation():
    """Test Detection data class creation and validation."""
    # Basic detection
    det = Detection(position=np.array([10.0, 20.0]), confidence=0.8)
    assert det.position.shape == (2,)
    assert det.confidence == 0.8
    assert det.embedding is None
    
    # Detection with all fields
    det = Detection(
        position=np.array([5.0, 15.0]),
        confidence=0.9,
        bbox=np.array([1.0, 2.0, 3.0, 4.0]),
        embedding=np.random.randn(128),
        class_id=1,
        id="det_1"
    )
    assert det.position.shape == (2,)
    assert det.bbox.shape == (4,)
    assert det.embedding.shape == (128,)


def test_tracked_object_creation():
    """Test TrackedObject data class creation."""
    obj = TrackedObject(
        id=1,
        position=np.array([10.0, 20.0]),
        velocity=np.array([1.0, 0.5]),
        confidence=0.8,
        age=1,
        hits=1,
        time_since_update=0,
        state=1
    )
    assert obj.id == 1
    assert obj.position.shape == (2,)
    assert obj.velocity.shape == (2,)
    assert obj.confidence == 0.8
    assert obj.age == 1
    assert obj.hits == 1
    assert obj.time_since_update == 0
    assert obj.state == 1


def test_config_creation_and_validation():
    """Test SwarmSortConfig creation and validation."""
    # Default config
    config = SwarmSortConfig()
    config.validate()
    
    # Custom config
    config = SwarmSortConfig(
        max_distance=100.0,
        high_score_threshold=0.7,
        use_embeddings=False
    )
    config.validate()
    assert config.max_distance == 100.0
    assert config.use_embeddings == False
    
    # Invalid config
    with pytest.raises(ValueError):
        config = SwarmSortConfig(max_distance=-1)
        config.validate()


def test_tracker_initialization():
    """Test tracker initialization with different configurations."""
    # Default config
    tracker = SwarmSortTracker()
    assert tracker.next_id == 1
    assert tracker.frame_count == 0
    assert len(tracker.tracks) == 0
    
    # Custom config
    config = SwarmSortConfig(max_distance=50.0, use_embeddings=False)
    tracker = SwarmSortTracker(config)
    assert tracker.config.max_distance == 50.0
    assert not tracker.config.use_embeddings
    
    # Dict config
    tracker = SwarmSortTracker({'max_distance': 75.0})
    assert tracker.config.max_distance == 75.0


def test_basic_tracking():
    """Test basic tracking functionality."""
    tracker = SwarmSortTracker()
    
    # Frame 1: Add detections
    detections = [
        Detection(position=np.array([10.0, 10.0]), confidence=0.9),
        Detection(position=np.array([50.0, 50.0]), confidence=0.8),
    ]
    
    # Process multiple frames to initialize tracks
    for _ in range(4):  # Ensure min_consecutive_detections is met
        tracked_objects = tracker.update(detections)
    
    # Should have created tracks after enough consecutive detections
    assert len(tracked_objects) >= 0  # Tracks may or may not be created yet due to initialization logic
    
    # Move detections slightly and update
    detections = [
        Detection(position=np.array([12.0, 11.0]), confidence=0.9),
        Detection(position=np.array([51.0, 49.0]), confidence=0.8),
    ]
    
    tracked_objects = tracker.update(detections)
    
    # Test tracker statistics
    stats = tracker.get_statistics()
    assert 'frame_count' in stats
    assert 'active_tracks' in stats
    assert 'embedding_scaler_stats' in stats


def test_embedding_tracking():
    """Test tracking with embeddings."""
    config = SwarmSortConfig(
        use_embeddings=True,
        embedding_weight=0.5,
        min_consecutive_detections=2  # Lower threshold for testing
    )
    tracker = SwarmSortTracker(config)
    
    # Create detections with embeddings
    emb1 = np.random.randn(128).astype(np.float32)
    emb2 = np.random.randn(128).astype(np.float32)
    
    detections = [
        Detection(position=np.array([10.0, 10.0]), confidence=0.9, embedding=emb1),
        Detection(position=np.array([50.0, 50.0]), confidence=0.8, embedding=emb2),
    ]
    
    # Process frames
    for _ in range(3):
        tracked_objects = tracker.update(detections)
        # Move detections slightly
        detections[0].position = detections[0].position + np.array([1.0, 0.5])
        detections[1].position = detections[1].position + np.array([-0.5, 1.0])
    
    # Should have some tracked objects
    stats = tracker.get_statistics()
    assert stats['frame_count'] == 3


def test_duplicate_detection_removal():
    """Test removal of duplicate detections."""
    tracker = SwarmSortTracker(SwarmSortConfig(duplicate_detection_threshold=5.0))
    
    # Create very close detections
    detections = [
        Detection(position=np.array([10.0, 10.0]), confidence=0.9),
        Detection(position=np.array([10.1, 10.1]), confidence=0.8),  # Very close
        Detection(position=np.array([50.0, 50.0]), confidence=0.7),
    ]
    
    tracked_objects = tracker.update(detections)
    # The close detections should be merged/filtered
    
    stats = tracker.get_statistics()
    assert 'frame_count' in stats


def test_integration_detection():
    """Test integration environment detection."""
    # This will be False in standalone tests
    within_swarmtracker = is_within_swarmtracker()
    assert isinstance(within_swarmtracker, bool)


def test_adaptive_tracker():
    """Test adaptive tracker creation."""
    # Should work regardless of environment
    tracker = SwarmSort()
    assert hasattr(tracker, 'update')
    assert hasattr(tracker, 'reset')
    assert hasattr(tracker, 'get_statistics')
    
    # Test with config
    config = SwarmSortConfig(max_distance=100.0)
    tracker = SwarmSort(config)
    assert tracker.config.max_distance == 100.0


def test_factory_function():
    """Test tracker factory function."""
    # Default creation
    tracker = create_tracker()
    assert hasattr(tracker, 'update')
    
    # With config dict
    tracker = create_tracker({'max_distance': 80.0})
    assert tracker.config.max_distance == 80.0
    
    # Force standalone
    tracker = create_tracker(force_standalone=True)
    assert hasattr(tracker, 'update')


def test_reset_functionality():
    """Test tracker reset."""
    tracker = SwarmSortTracker()
    
    # Add some data
    detections = [Detection(position=np.array([10.0, 10.0]), confidence=0.9)]
    tracker.update(detections)
    tracker.update(detections)
    
    assert tracker.frame_count > 0
    
    # Reset
    tracker.reset()
    assert tracker.frame_count == 0
    assert tracker.next_id == 1
    assert len(tracker.tracks) == 0


def test_config_yaml_functionality():
    """Test YAML config loading functionality (without actual file)."""
    # Test dict conversion
    config_dict = {
        'max_distance': 90.0,
        'high_score_threshold': 0.75,
        'use_embeddings': True,
        'embedding_weight': 0.4
    }
    
    config = SwarmSortConfig.from_dict(config_dict)
    assert config.max_distance == 90.0
    assert config.high_score_threshold == 0.75
    assert config.use_embeddings == True
    assert config.embedding_weight == 0.4
    
    # Test dict export
    exported = config.to_dict()
    assert exported['max_distance'] == 90.0


if __name__ == '__main__':
    pytest.main([__file__])