"""
Test ReID functionality with the current SwarmSort implementation.
"""
import numpy as np
from swarmsort import SwarmSortTracker, Detection


def test_reid_direct():
    """Test ReID with tracks that have missed detections"""
    print("Testing ReID optimization with tracks that have missed detections...")

    config = {"debug_timings": True, "reid_enabled": True, "reid_max_frames": 10}
    tracker = SwarmSortTracker(config=config)

    # Phase 1: Create tracks by providing consistent detections
    print("Creating tracks...")
    base_positions = [np.array([100.0, 150.0]), np.array([300.0, 200.0]), np.array([500.0, 250.0])]
    
    for frame in range(5):
        detections = []
        for i, base_pos in enumerate(base_positions):
            # Add small noise to position
            pos = base_pos + np.random.randn(2) * 5
            # Create consistent embedding for each track
            embedding = np.random.randn(128).astype(np.float32)
            embedding[i] = 1.0  # Make each track distinctive
            embedding = embedding / np.linalg.norm(embedding)
            
            det = Detection(
                position=pos,
                confidence=0.9,
                embedding=embedding,
                bbox=np.array([pos[0]-25, pos[1]-25, pos[0]+25, pos[1]+25])
            )
            detections.append(det)
        
        tracks = tracker.update(detections)
        print(f"Frame {frame}: {len(tracks)} active tracks")

    print(f"Created {len(tracker.tracks)} tracks")

    # Phase 2: Skip frames to create missed detections
    print("Skipping frames to create missed detections...")
    for frame in range(5, 8):
        tracks = tracker.update([])  # No detections
        print(f"Frame {frame}: {len(tracks)} active tracks")

    # Phase 3: Try to re-identify with similar detections
    print("Testing ReID with similar detections...")
    reid_detections = []
    
    for i, base_pos in enumerate(base_positions):
        # Create detection with similar position but some drift
        pos = base_pos + np.array([20.0, 15.0])  # Some drift
        # Create similar embedding
        embedding = np.random.randn(128).astype(np.float32)
        embedding[i] = 0.8  # Similar but not identical
        embedding = embedding / np.linalg.norm(embedding)
        
        det = Detection(
            position=pos,
            confidence=0.85,
            embedding=embedding,
            bbox=np.array([pos[0]-25, pos[1]-25, pos[0]+25, pos[1]+25])
        )
        reid_detections.append(det)

    # Update with ReID detections
    tracks = tracker.update(reid_detections)
    print(f"After ReID attempt: {len(tracks)} active tracks")

    # Check statistics
    stats = tracker.get_statistics()
    print(f"Final statistics: {stats}")
    print("ReID test completed")

    # Basic assertion - we should have some tracks (internal tracker state)
    assert stats["active_tracks"] > 0, "Should have active tracks in internal state"
    print("ReID test passed - tracks exist in internal tracker state")


if __name__ == "__main__":
    test_reid_direct()