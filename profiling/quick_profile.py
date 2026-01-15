#!/usr/bin/env python3
"""
Quick SwarmSort Profiler

A lightweight profiling script for quick performance analysis.
Focuses on the most critical functions and timing bottlenecks.

Usage:
    python quick_profile.py
"""

import time
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path to import swarmsort
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from swarmsort import SwarmSort, SwarmSortConfig, Detection

class QuickProfiler:
    """Lightweight profiler for SwarmSort operations"""
    
    def __init__(self):
        self.timings = {}
        self.call_counts = {}
        self.active_timers = {}
    
    def start(self, name):
        """Start timing an operation"""
        self.active_timers[name] = time.perf_counter()
    
    def end(self, name):
        """End timing an operation"""
        if name in self.active_timers:
            duration = time.perf_counter() - self.active_timers[name]
            if name not in self.timings:
                self.timings[name] = []
                self.call_counts[name] = 0
            self.timings[name].append(duration)
            self.call_counts[name] += 1
            del self.active_timers[name]
    
    def report(self):
        """Generate performance report"""
        print("\n" + "=" * 60)
        print("QUICK PERFORMANCE PROFILE REPORT")
        print("=" * 60)
        
        # Calculate statistics
        stats = []
        for name, times in self.timings.items():
            total_time = sum(times)
            avg_time = total_time / len(times)
            max_time = max(times)
            min_time = min(times)
            calls = self.call_counts[name]
            
            stats.append({
                'name': name,
                'total': total_time * 1000,  # Convert to ms
                'avg': avg_time * 1000,
                'max': max_time * 1000,
                'min': min_time * 1000,
                'calls': calls
            })
        
        # Sort by total time
        stats.sort(key=lambda x: x['total'], reverse=True)
        
        # Print header
        print(f"{'Operation':<25} {'Total(ms)':<10} {'Avg(ms)':<10} {'Max(ms)':<10} {'Calls':<8}")
        print("-" * 70)
        
        # Print stats
        for stat in stats:
            print(f"{stat['name']:<25} {stat['total']:<10.2f} {stat['avg']:<10.2f} {stat['max']:<10.2f} {stat['calls']:<8}")
        
        print("-" * 70)
        total_profiled_time = sum(stat['total'] for stat in stats)
        print(f"{'TOTAL PROFILED TIME':<25} {total_profiled_time:<10.2f}")
        print()

def create_test_scenario(num_objects=110, num_frames=50, use_embeddings=True):
    """
    Create test scenario with realistic tracking patterns.
    
    Args:
        num_objects: Base number of objects
        num_frames: Number of frames to simulate
        use_embeddings: Whether to include fake embeddings
    """
    scenarios = []
    np.random.seed(42)
    
    # Track persistent objects with motion
    track_states = {}
    for i in range(num_objects):
        x = (i % 12) * 80 + np.random.randn() * 15
        y = (i // 12) * 80 + np.random.randn() * 15
        track_states[i] = {
            'pos': np.array([x, y]),
            'vel': np.random.randn(2) * 2,
            'embedding_base': np.random.randn(128).astype(np.float32) if use_embeddings else None
        }
    
    for frame_id in range(num_frames):
        detections = []
        
        # Update and add existing tracks (with some dropouts)
        visible_tracks = np.random.choice(
            list(track_states.keys()), 
            size=max(1, int(len(track_states) * (0.85 + np.random.rand() * 0.15))),
            replace=False
        )
        
        for track_id in visible_tracks:
            track = track_states[track_id]
            
            # Update position with motion model
            track['pos'] += track['vel'] + np.random.randn(2) * 1
            track['pos'] = np.clip(track['pos'], 0, 1000)  # Keep in bounds
            
            # Occasionally change direction
            if np.random.rand() < 0.1:
                track['vel'] = np.random.randn(2) * 2
            
            x, y = track['pos']
            
            # Generate embedding with consistency
            embedding = None
            if use_embeddings and track['embedding_base'] is not None:
                # Base embedding + small noise for realism
                noise = np.random.randn(128).astype(np.float32) * 0.05
                embedding = track['embedding_base'] + noise
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            det = Detection(
                position=np.array([x, y], dtype=np.float32),
                confidence=0.7 + np.random.rand() * 0.3,
                embedding=embedding,
                bbox=np.array([x-20, y-20, x+20, y+20], dtype=np.float32),
                id=f"det_{frame_id}_{track_id}"
            )
            detections.append(det)
        
        scenarios.append(detections)
    
    return scenarios

def profile_swarmsort_update():
    """Profile the main SwarmSort update function over multiple frames"""
    
    profiler = QuickProfiler()
    
    # Create tracker with realistic configuration
    config = SwarmSortConfig(
        max_distance=120.0,
        reid_max_distance=150.0,
        do_embeddings=True,
        embedding_weight=1.0,
        uncertainty_weight=0.0,
        min_consecutive_detections=6,
        assignment_strategy="hybrid",
        collision_freeze_embeddings=True,
        debug_timings=False,  # We'll do our own timing
    )
    
    tracker = SwarmSort(config)
    
    print("Running SwarmSort profiling...")
    print("Testing with 110 objects over 50 frames with embeddings...")
    
    # Create realistic test scenario
    scenarios = create_test_scenario(110, 50, use_embeddings=True)
    
    total_start = time.perf_counter()
    frame_times = []
    
    # Run test scenario
    for frame_idx, detections in enumerate(scenarios):
        profiler.start("total_update")
        
        frame_start = time.perf_counter()
        tracks = tracker.update(detections)
        frame_duration = time.perf_counter() - frame_start
        frame_times.append(frame_duration)
        
        profiler.end("total_update")
        
        # Log frame performance
        if frame_duration > 0.020:  # Log slow frames (>20ms)
            print(f"Frame {frame_idx}: {len(detections)} dets -> {len(tracks)} tracks | {frame_duration*1000:.1f}ms [SLOW]")
        elif frame_idx % 10 == 0:  # Log every 10th frame
            print(f"Frame {frame_idx}: {len(detections)} dets -> {len(tracks)} tracks | {frame_duration*1000:.1f}ms")
    
    total_duration = time.perf_counter() - total_start
    
    print(f"\nPerformance Summary:")
    print(f"  Total test duration: {total_duration:.2f}s")
    print(f"  Average frame time: {np.mean(frame_times)*1000:.1f}ms")
    print(f"  Median frame time: {np.median(frame_times)*1000:.1f}ms")
    print(f"  Max frame time: {np.max(frame_times)*1000:.1f}ms")
    print(f"  95th percentile: {np.percentile(frame_times, 95)*1000:.1f}ms")
    
    # Generate report
    profiler.report()

def profile_component_breakdown():
    """Profile individual components in isolation"""

    print("\n" + "=" * 60)
    print("COMPONENT BREAKDOWN PROFILING")
    print("=" * 60)

    profiler = QuickProfiler()

    # Test data
    num_dets = 110
    num_tracks = 50

    # Generate test arrays
    det_positions = np.random.rand(num_dets, 2).astype(np.float32) * 1000
    track_predicted_positions = np.random.rand(num_tracks, 2).astype(np.float32) * 1000
    track_last_positions = track_predicted_positions + np.random.randn(num_tracks, 2).astype(np.float32) * 5
    track_misses = np.zeros(num_tracks, dtype=np.int32)
    det_embeddings = np.random.randn(num_dets, 128).astype(np.float32)
    track_embeddings = np.random.randn(num_tracks, 128).astype(np.float32)

    # Normalize embeddings
    det_embeddings = det_embeddings / (np.linalg.norm(det_embeddings, axis=1, keepdims=True) + 1e-8)
    track_embeddings = track_embeddings / (np.linalg.norm(track_embeddings, axis=1, keepdims=True) + 1e-8)

    # Import the cost computation functions
    from swarmsort.cost_computation import (
        cosine_similarity_normalized,
        compute_cost_matrix_vectorized
    )

    print(f"Testing with {num_dets} detections and {num_tracks} tracks...")

    # Test embedding similarity computation
    print("\nTesting embedding similarity (single pairs)...")
    profiler.start("embedding_similarity_single")
    for i in range(min(10, num_dets)):  # Test first 10
        for j in range(min(10, num_tracks)):
            _ = cosine_similarity_normalized(det_embeddings[i], track_embeddings[j])
    profiler.end("embedding_similarity_single")

    # Test batch embedding distance computation using matrix multiplication
    # This is the core operation: cosine distance = 1 - dot(a, b) for normalized vectors
    print("Testing embedding distances (batch matmul)...")
    profiler.start("embedding_distances_batch")
    # Cosine similarity via matmul: det_emb @ track_emb.T
    cos_sim = det_embeddings @ track_embeddings.T  # [num_dets, num_tracks]
    # Convert to distance in [0, 1]
    scaled_embedding_distances = (1.0 - cos_sim) * 0.5
    profiler.end("embedding_distances_batch")

    # Test cost matrix computation
    print("Testing cost matrix computation...")
    profiler.start("cost_matrix")
    _ = compute_cost_matrix_vectorized(
        det_positions=det_positions,
        track_predicted_positions=track_predicted_positions,
        track_last_positions=track_last_positions,
        track_misses=track_misses,
        scaled_embedding_distances=scaled_embedding_distances,
        embedding_weight=1.0,
        max_distance=120.0,
        do_embeddings=True
    )
    profiler.end("cost_matrix")

    # Test cost matrix without embeddings
    print("Testing cost matrix (no embeddings)...")
    profiler.start("cost_matrix_no_emb")
    empty_emb = np.zeros((num_dets, num_tracks), dtype=np.float32)
    _ = compute_cost_matrix_vectorized(
        det_positions=det_positions,
        track_predicted_positions=track_predicted_positions,
        track_last_positions=track_last_positions,
        track_misses=track_misses,
        scaled_embedding_distances=empty_emb,
        embedding_weight=0.0,
        max_distance=120.0,
        do_embeddings=False
    )
    profiler.end("cost_matrix_no_emb")

    # Report component performance
    profiler.report()

def main():
    """Main function"""
    print("SwarmSort Quick Profiler")
    print("This will run quick performance tests and identify bottlenecks.")
    
    # Profile main update function
    profile_swarmsort_update()
    
    # Profile individual components
    profile_component_breakdown()
    
    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)
    print("\nFor detailed profiling with call graphs, run:")
    print("  python profile_swarmsort.py")
    print("\nFor snakeviz visualization:")
    print("  pip install snakeviz")
    print("  python profile_swarmsort.py")

if __name__ == "__main__":
    main()