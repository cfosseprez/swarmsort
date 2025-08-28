#!/usr/bin/env python3
"""Direct ReID test with manually created lost tracks"""

import numpy as np
import time
from swarmsort import SwarmSortTracker, Detection
from swarmsort.core import FastTrackState


def create_detection_with_embedding(position, bbox, confidence, class_id, emb_dim=128):
    """Create detection with random embedding"""
    embedding = np.random.randn(emb_dim).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)  # normalize
    return Detection(
        position=np.array(position, dtype=np.float32),
        bbox=np.array(bbox, dtype=np.float32),
        confidence=confidence,
        class_id=class_id,
        embedding=embedding,
    )


def test_reid_direct():
    """Test ReID with manually created lost tracks"""
    print("Testing ReID optimization with manually created lost tracks...")

    config = {"debug_timings": True, "reid_enabled": True, "reid_max_frames": 10}
    tracker = SwarmSortTracker(config=config)

    # Manually create some lost tracks with embeddings
    print("Creating lost tracks...")

    for i in range(3):
        track = FastTrackState(
            id=i + 1, position=np.array([100 + i * 200, 150 + i * 50], dtype=np.float32)
        )
        track.set_embedding_params(5, "best_match")

        # Add some embeddings to the track
        for j in range(3):
            emb = np.random.randn(128).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            track.add_embedding(emb)

        track.confirmed = True
        track.hits = 5
        track.misses = 2
        track.lost_frames = 3
        track.last_detection_frame = 7

        tracker.lost_tracks[track.id] = track

    tracker.frame_count = 10  # Set current frame

    print(f"Created {len(tracker.lost_tracks)} lost tracks")

    # Create detections that might match the lost tracks
    reid_detections = []

    # Create detections with similar positions to the lost tracks
    for i, (track_id, lost_track) in enumerate(tracker.lost_tracks.items()):
        # Use similar position but with some noise
        pos = lost_track.position + np.random.randn(2) * 10
        bbox = np.array([pos[0] - 25, pos[1] - 25, pos[0] + 25, pos[1] + 25])

        # Create detection with similar embedding
        if len(lost_track.embedding_history) > 0:
            base_emb = lost_track.embedding_history[-1].copy()
            # Add small noise to make it similar but not identical
            noisy_emb = base_emb + np.random.randn(len(base_emb)) * 0.05
            noisy_emb = noisy_emb / np.linalg.norm(noisy_emb)
        else:
            noisy_emb = np.random.randn(128).astype(np.float32)
            noisy_emb = noisy_emb / np.linalg.norm(noisy_emb)

        det = Detection(position=pos, bbox=bbox, confidence=0.8, class_id=i, embedding=noisy_emb)
        reid_detections.append(det)

        if i >= 2:  # Only create 3 detections
            break

    print(f"\nTesting ReID with {len(reid_detections)} detections...")
    start = time.perf_counter()
    result = tracker.update(reid_detections)
    end = time.perf_counter()

    print(f"ReID frame: {len(result)} tracks recovered")
    print(f"Active tracks: {len(tracker.tracks)}")
    print(f"Lost tracks remaining: {len(tracker.lost_tracks)}")
    print(f"Total time: {(end - start) * 1000:.2f} ms")

    # Print detailed ReID timing breakdown
    if hasattr(tracker, "timings"):
        print("\nDetailed ReID timing breakdown:")
        reid_timings = {k: v for k, v in tracker.timings.items() if k.startswith("reid_")}
        if reid_timings:
            total_reid_time = 0
            for key, value in reid_timings.items():
                print(f"  {key}: {value}")
                # Extract numeric value
                try:
                    numeric_val = float(value.split()[0])
                    total_reid_time += numeric_val
                except:
                    pass
            print(f"  Total ReID time: {total_reid_time:.2f} ms")
        else:
            print("  No ReID timing data (ReID may not have been triggered)")

        # Also show main timings for context
        print("\nMain timings:")
        main_timings = {k: v for k, v in tracker.timings.items() if not k.startswith("reid_")}
        for key, value in main_timings.items():
            print(f"  {key}: {value}")

    return True


if __name__ == "__main__":
    success = test_reid_direct()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
