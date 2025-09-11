#!/usr/bin/env python3
"""
SwarmSort Performance Profiler

This script profiles SwarmSort performance using cProfile and generates
visualizations with snakeviz for detailed performance analysis.

Usage:
    python profile_swarmsort.py
    
The script will:
1. Run SwarmSort with a realistic test scenario
2. Generate profile data (.prof files)
3. Create snakeviz visualizations
4. Generate performance reports

Output:
- profile_data.prof: Raw profiling data
- profile_report.txt: Text-based performance report
- Opens browser with snakeviz visualization
"""

import cProfile
import pstats
import io
import os
import sys
import numpy as np
import time
from pathlib import Path

# Add parent directory to path to import swarmsort
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from swarmsort import SwarmSort, SwarmSortConfig, Detection

def create_realistic_scenario(num_objects=110, num_frames=100, use_fake_embeddings=True):
    """
    Create a realistic test scenario with varying object counts over many frames.
    
    Args:
        num_objects: Base number of objects to track
        num_frames: Number of frames to simulate (default 100 for better profiling)
        use_fake_embeddings: If True, generate random embeddings; if False, set to None
    """
    scenarios = []
    
    np.random.seed(42)  # For reproducible results
    
    # Simulate object tracks with consistent motion
    track_positions = {}
    track_velocities = {}
    track_ids = list(range(num_objects))
    
    for frame in range(num_frames):
        detections = []
        
        # Vary detection count slightly (simulate missed detections)
        current_objects = num_objects + np.random.randint(-10, 10)
        
        # Randomly select which tracks are visible this frame
        visible_tracks = np.random.choice(track_ids, size=min(current_objects, len(track_ids)), replace=False)
        
        for track_id in visible_tracks:
            # Initialize or update track position
            if track_id not in track_positions:
                # Initialize new track
                x = (track_id % 12) * 80 + np.random.randn() * 10
                y = (track_id // 12) * 80 + np.random.randn() * 10
                track_positions[track_id] = np.array([x, y])
                # Random velocity (simulating movement)
                track_velocities[track_id] = np.random.randn(2) * 2
            else:
                # Update position with velocity and noise
                track_positions[track_id] += track_velocities[track_id]
                track_positions[track_id] += np.random.randn(2) * 3  # Add noise
                
                # Occasionally change velocity (simulate direction changes)
                if np.random.rand() < 0.1:
                    track_velocities[track_id] = np.random.randn(2) * 2
                
                # Keep within bounds
                track_positions[track_id] = np.clip(track_positions[track_id], 0, 1000)
            
            x, y = track_positions[track_id]
            
            # Generate embedding
            if use_fake_embeddings:
                # Create semi-consistent embeddings for same track
                base_embedding = np.random.RandomState(track_id).randn(128)
                noise = np.random.randn(128) * 0.1  # Small variation per frame
                embedding = (base_embedding + noise).astype(np.float32)
                # Normalize embedding
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            else:
                embedding = None
            
            det = Detection(
                position=np.array([x, y], dtype=np.float32),
                confidence=0.6 + np.random.rand() * 0.4,
                embedding=embedding,
                bbox=np.array([x-25, y-25, x+25, y+25], dtype=np.float32),
                id=f"det_{frame}_{track_id}"
            )
            detections.append(det)
        
        # Add some new detections (false positives or new objects)
        if np.random.rand() < 0.1:
            for _ in range(np.random.randint(1, 5)):
                x = np.random.rand() * 1000
                y = np.random.rand() * 1000
                
                if use_fake_embeddings:
                    embedding = np.random.randn(128).astype(np.float32)
                    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                else:
                    embedding = None
                
                det = Detection(
                    position=np.array([x, y], dtype=np.float32),
                    confidence=0.5 + np.random.rand() * 0.3,
                    embedding=embedding,
                    bbox=np.array([x-25, y-25, x+25, y+25], dtype=np.float32),
                    id=f"det_{frame}_new_{_}"
                )
                detections.append(det)
        
        scenarios.append(detections)
    
    return scenarios

def profile_swarmsort_basic():
    """Basic SwarmSort profiling with standard configuration"""
    print("Running basic SwarmSort profiling...")
    print("Configuration: 110 objects, 100 frames, with embeddings")
    
    config = SwarmSortConfig(
        max_distance=120.0,
        reid_max_distance=150.0,
        pending_detection_distance=120.0,
        do_embeddings=True,
        embedding_weight=1.0,
        use_probabilistic_costs=False,
        uncertainty_weight=0.0,
        min_consecutive_detections=6,
        assignment_strategy="hybrid",
        greedy_threshold=40.0,
        collision_freeze_embeddings=True,
        embedding_freeze_density=2,
        debug_timings=False,  # Disable for cleaner profiling
    )
    
    tracker = SwarmSort(config)
    scenarios = create_realistic_scenario(110, 100, use_fake_embeddings=True)
    
    def run_tracking():
        frame_times = []
        for frame_idx, detections in enumerate(scenarios):
            start = time.perf_counter()
            tracks = tracker.update(detections)
            frame_time = time.perf_counter() - start
            frame_times.append(frame_time)
            
            # Print progress every 20 frames
            if (frame_idx + 1) % 20 == 0:
                avg_time = np.mean(frame_times[-20:]) * 1000
                print(f"  Frame {frame_idx+1}/100: {len(detections)} dets -> {len(tracks)} tracks | Avg: {avg_time:.1f}ms")
        
        # Print summary
        print(f"\n  Summary: Avg {np.mean(frame_times)*1000:.1f}ms, Max {np.max(frame_times)*1000:.1f}ms")
    
    return run_tracking

def profile_swarmsort_with_uncertainty():
    """Profile SwarmSort with uncertainty system enabled"""
    print("Running SwarmSort profiling with uncertainty...")
    print("Configuration: 110 objects, 100 frames, uncertainty=0.25, with embeddings")
    
    config = SwarmSortConfig(
        max_distance=120.0,
        reid_max_distance=150.0,
        pending_detection_distance=120.0,
        do_embeddings=True,
        embedding_weight=1.0,
        use_probabilistic_costs=False,
        uncertainty_weight=0.25,  # Enable uncertainty
        min_consecutive_detections=6,
        assignment_strategy="hybrid",
        greedy_threshold=40.0,
        collision_freeze_embeddings=True,
        embedding_freeze_density=2,
        debug_timings=False,
    )
    
    tracker = SwarmSort(config)
    scenarios = create_realistic_scenario(110, 100, use_fake_embeddings=True)
    
    def run_tracking():
        frame_times = []
        for frame_idx, detections in enumerate(scenarios):
            start = time.perf_counter()
            tracks = tracker.update(detections)
            frame_time = time.perf_counter() - start
            frame_times.append(frame_time)
            
            # Print progress every 20 frames
            if (frame_idx + 1) % 20 == 0:
                avg_time = np.mean(frame_times[-20:]) * 1000
                print(f"  Frame {frame_idx+1}/100: {len(detections)} dets -> {len(tracks)} tracks | Avg: {avg_time:.1f}ms")
        
        # Print summary
        print(f"\n  Summary: Avg {np.mean(frame_times)*1000:.1f}ms, Max {np.max(frame_times)*1000:.1f}ms")
    
    return run_tracking

def run_profiling_session(profile_func, output_name):
    """Run a profiling session and save results"""
    print(f"Profiling {output_name}...")
    
    # Create profiler
    profiler = cProfile.Profile()
    
    # Run profiling
    start_time = time.perf_counter()
    profiler.enable()
    profile_func()
    profiler.disable()
    end_time = time.perf_counter()
    
    # Save profile data
    profile_file = f"profiling/{output_name}.prof"
    profiler.dump_stats(profile_file)
    print(f"Profile data saved to: {profile_file}")
    
    # Generate text report
    report_file = f"profiling/{output_name}_report.txt"
    with open(report_file, 'w') as f:
        f.write(f"SwarmSort Profiling Report - {output_name}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total execution time: {end_time - start_time:.2f} seconds\n\n")
        
        # Get stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        
        # Sort by cumulative time and print top functions
        f.write("Top functions by cumulative time:\n")
        f.write("-" * 40 + "\n")
        ps.sort_stats('cumulative').print_stats(20)
        f.write(s.getvalue())
        
        # Reset string buffer
        s.seek(0)
        s.truncate(0)
        
        # Sort by internal time
        f.write("\n\nTop functions by internal time:\n")
        f.write("-" * 40 + "\n")
        ps.sort_stats('tottime').print_stats(20)
        f.write(s.getvalue())
        
        # Function call statistics
        s.seek(0)
        s.truncate(0)
        f.write("\n\nFunction call statistics:\n")
        f.write("-" * 40 + "\n")
        ps.sort_stats('calls').print_stats(20)
        f.write(s.getvalue())
    
    print(f"Text report saved to: {report_file}")
    return profile_file

def launch_snakeviz(profile_file):
    """Launch snakeviz for interactive visualization"""
    try:
        import snakeviz
        print(f"Launching snakeviz for {profile_file}...")
        os.system(f"snakeviz {profile_file}")
    except ImportError:
        print("snakeviz not installed. Install with: pip install snakeviz")
        print(f"You can manually run: snakeviz {profile_file}")

def generate_comparative_report():
    """Generate a comparative analysis report"""
    print("Generating comparative report...")
    
    try:
        # Load both profile files
        basic_stats = pstats.Stats("profiling/swarmsort_basic.prof")
        uncertainty_stats = pstats.Stats("profiling/swarmsort_uncertainty.prof")
        
        with open("profiling/comparative_report.txt", 'w') as f:
            f.write("SwarmSort Comparative Performance Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            # Get key metrics
            basic_total = basic_stats.total_tt
            uncertainty_total = uncertainty_stats.total_tt
            
            f.write(f"Basic configuration total time: {basic_total:.4f}s\n")
            f.write(f"Uncertainty configuration total time: {uncertainty_total:.4f}s\n")
            f.write(f"Performance ratio: {uncertainty_total/basic_total:.2f}x\n\n")
            
            # Identify bottlenecks
            f.write("Key Performance Bottlenecks:\n")
            f.write("-" * 30 + "\n")
            
            # Get top functions from basic config
            basic_stats.sort_stats('tottime')
            basic_funcs = basic_stats.get_stats_profile().func_profiles
            
            # Find SwarmSort-specific functions
            swarmsort_funcs = []
            for func_name, stats in basic_funcs.items():
                if 'swarmsort' in str(func_name).lower() or 'core.py' in str(func_name):
                    swarmsort_funcs.append((func_name, stats))
            
            # Sort by total time and take top 10
            swarmsort_funcs.sort(key=lambda x: x[1].tottime, reverse=True)
            
            for i, (func_name, stats) in enumerate(swarmsort_funcs[:10]):
                f.write(f"{i+1}. {func_name}: {stats.tottime:.4f}s ({stats.ncalls} calls)\n")
                
        print("Comparative report saved to: profiling/comparative_report.txt")
        
    except Exception as e:
        print(f"Could not generate comparative report: {e}")

def main():
    """Main profiling function"""
    print("SwarmSort Performance Profiler")
    print("=" * 40)
    
    # Ensure profiling directory exists
    os.makedirs("profiling", exist_ok=True)
    
    # Profile basic configuration
    basic_func = profile_swarmsort_basic()
    basic_profile = run_profiling_session(basic_func, "swarmsort_basic")
    
    # Profile with uncertainty
    uncertainty_func = profile_swarmsort_with_uncertainty()
    uncertainty_profile = run_profiling_session(uncertainty_func, "swarmsort_uncertainty")
    
    # Generate comparative report
    generate_comparative_report()
    
    print("\nProfiling complete!")
    print("Files generated:")
    print("- profiling/swarmsort_basic.prof")
    print("- profiling/swarmsort_basic_report.txt")
    print("- profiling/swarmsort_uncertainty.prof") 
    print("- profiling/swarmsort_uncertainty_report.txt")
    print("- profiling/comparative_report.txt")
    print("\nTo view interactive visualizations, run:")
    print(f"  snakeviz {basic_profile}")
    print(f"  snakeviz {uncertainty_profile}")
    
    # Optionally launch snakeviz
    launch_choice = input("\nLaunch snakeviz visualization now? (y/n): ").lower()
    if launch_choice == 'y':
        print("Launching basic configuration visualization...")
        launch_snakeviz(basic_profile)
        
        uncertainty_choice = input("Launch uncertainty configuration visualization? (y/n): ").lower()
        if uncertainty_choice == 'y':
            launch_snakeviz(uncertainty_profile)

if __name__ == "__main__":
    main()