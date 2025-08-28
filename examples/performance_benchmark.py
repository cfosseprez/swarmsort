"""
Simple SwarmSort Performance Benchmark

This script runs tracking performance tests with different object counts
and plots the timing results for 1000 frames of tracking.
"""
import numpy as np
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

# Add the parent directory to path for importing swarmsort
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarmsort import SwarmSortTracker, SwarmSortConfig
from swarmsort.data_classes import Detection
from swarmsort.simulator import ObjectMotionSimulator, SimulationConfig

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available - plots will not be generated")


def create_test_data(num_objects: int, num_frames: int = 1000) -> List[List[Detection]]:
    """Create test detection sequences for benchmarking."""
    # Simple simulation config
    sim_config = SimulationConfig(
        world_width=1920,
        world_height=1080,
        detection_probability=0.95,
        false_positive_rate=0.01,
        position_noise_std=1.0,
        use_embeddings=True,
        random_seed=42
    )
    
    sim = ObjectMotionSimulator(sim_config)
    
    # Add random walk objects with very small step sizes for realistic motion
    np.random.seed(42)
    for i in range(num_objects):
        start_x = np.random.uniform(100, 1820)
        start_y = np.random.uniform(100, 980)
        step_size = np.random.uniform(0.5, 2.0)  # Small but visible movement
        
        obj = sim.create_random_walk_object(
            object_id=i+1,
            start_pos=(start_x, start_y),
            step_size=step_size,
            class_id=i % 5,
            base_confidence=np.random.uniform(0.7, 0.95)
        )
        sim.add_object(obj)
    
    print(f"Generating {num_frames} frames with {num_objects} objects...")
    return sim.run_simulation(num_frames)


def benchmark_tracking(detections_sequence: List[List[Detection]], 
                      config: SwarmSortConfig) -> Tuple[float, float, int]:
    """Benchmark tracking performance and return timing results."""
    tracker = SwarmSortTracker(config)
    
    # Warm up
    for i in range(min(10, len(detections_sequence))):
        tracker.update(detections_sequence[i])
    
    tracker.reset()
    
    # Actual benchmark
    start_time = time.perf_counter()
    
    for frame_detections in detections_sequence:
        tracker.update(frame_detections)
    
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time_per_frame = (total_time / len(detections_sequence)) * 1000  # ms
    
    # Get final stats
    stats = tracker.get_statistics()
    total_tracks = stats.get('active_tracks', 0)
    
    return total_time, avg_time_per_frame, total_tracks


def run_performance_benchmark():
    """Run the main performance benchmark."""
    print("SwarmSort Simple Performance Benchmark")
    print("=" * 50)
    
    # Test parameters
    object_counts = [50, 100, 200, 300, 500]
    num_frames = 1000
    
    # Test configurations
    configs = {
        "Basic (No Embeddings)": SwarmSortConfig(
            max_distance=80.0,
            use_embeddings=False
        ),
        "With Embeddings": SwarmSortConfig(
            max_distance=80.0,
            use_embeddings=True,
            embedding_weight=0.5
        )
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\nTesting configuration: {config_name}")
        print("-" * 30)
        
        results[config_name] = {
            'object_counts': [],
            'total_times': [],
            'avg_times_per_frame': [],
            'total_tracks': []
        }
        
        for num_objects in object_counts:
            print(f"  Testing with {num_objects} objects...")
            
            # Generate test data
            detections_sequence = create_test_data(num_objects, num_frames)
            
            # Run benchmark
            total_time, avg_time_per_frame, total_tracks = benchmark_tracking(
                detections_sequence, config
            )
            
            # Store results
            results[config_name]['object_counts'].append(num_objects)
            results[config_name]['total_times'].append(total_time)
            results[config_name]['avg_times_per_frame'].append(avg_time_per_frame)
            results[config_name]['total_tracks'].append(total_tracks)
            
            # Print results
            fps = 1000.0 / avg_time_per_frame
            print(f"    Total time: {total_time:.2f}s")
            print(f"    Avg time per frame: {avg_time_per_frame:.2f}ms")
            print(f"    FPS: {fps:.1f}")
            print(f"    Total tracks created: {total_tracks}")
    
    return results


def create_performance_plots(results: Dict, output_dir: Path):
    """Create performance plots."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available - skipping plots")
        return
    
    print(f"\nCreating performance plots in {output_dir}...")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Set up the plot style
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SwarmSort Performance Benchmark (1000 frames)', fontsize=16)
    
    colors = ['#2E86C1', '#E74C3C']  # Blue and Red
    
    # Plot 1: Average Time per Frame
    for i, (config_name, data) in enumerate(results.items()):
        ax1.plot(data['object_counts'], data['avg_times_per_frame'], 
                marker='o', linewidth=2, markersize=6, 
                label=config_name, color=colors[i])
    
    ax1.set_xlabel('Number of Objects')
    ax1.set_ylabel('Average Time per Frame (ms)')
    ax1.set_title('Processing Time vs Object Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Frames Per Second (FPS)
    for i, (config_name, data) in enumerate(results.items()):
        fps_values = [1000.0 / t for t in data['avg_times_per_frame']]
        ax2.plot(data['object_counts'], fps_values, 
                marker='s', linewidth=2, markersize=6,
                label=config_name, color=colors[i])
    
    ax2.set_xlabel('Number of Objects')
    ax2.set_ylabel('Frames Per Second (FPS)')
    ax2.set_title('Tracking Speed vs Object Count')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Total Processing Time
    for i, (config_name, data) in enumerate(results.items()):
        ax3.plot(data['object_counts'], data['total_times'], 
                marker='^', linewidth=2, markersize=6,
                label=config_name, color=colors[i])
    
    ax3.set_xlabel('Number of Objects')
    ax3.set_ylabel('Total Time (seconds)')
    ax3.set_title('Total Processing Time (1000 frames)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance Ratio (Embeddings vs Basic)
    if len(results) == 2:
        config_names = list(results.keys())
        basic_times = results[config_names[0]]['avg_times_per_frame']
        embedding_times = results[config_names[1]]['avg_times_per_frame']
        
        ratios = [e/b for e, b in zip(embedding_times, basic_times)]
        object_counts = results[config_names[0]]['object_counts']
        
        ax4.plot(object_counts, ratios, 
                marker='d', linewidth=2, markersize=6,
                color='#8E44AD', label='Embedding Overhead')
        ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='No Overhead')
        
        ax4.set_xlabel('Number of Objects')
        ax4.set_ylabel('Time Ratio (Embeddings / Basic)')
        ax4.set_title('Embedding Processing Overhead')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "performance_benchmark.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Performance plot saved as {plot_path}")
    
    # Show plot
    plt.show()


def save_results_summary(results: Dict, output_dir: Path):
    """Save a text summary of the results."""
    output_dir.mkdir(exist_ok=True)
    summary_path = output_dir / "benchmark_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("SwarmSort Performance Benchmark Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Test: 1000 frames with varying object counts\n")
        f.write(f"Object counts tested: {results[list(results.keys())[0]]['object_counts']}\n\n")
        
        for config_name, data in results.items():
            f.write(f"\n{config_name}:\n")
            f.write("-" * len(config_name) + "-\n")
            
            for i, num_objects in enumerate(data['object_counts']):
                fps = 1000.0 / data['avg_times_per_frame'][i]
                f.write(f"  {num_objects:3d} objects: ")
                f.write(f"{data['avg_times_per_frame'][i]:6.2f}ms/frame, ")
                f.write(f"{fps:6.1f} FPS, ")
                f.write(f"{data['total_tracks'][i]:3d} tracks\n")
    
    print(f"Results summary saved as {summary_path}")


def main():
    """Run the performance benchmark."""
    try:
        # Create results directory
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Run benchmark
        results = run_performance_benchmark()
        
        # Create plots
        create_performance_plots(results, results_dir)
        
        # Save summary
        save_results_summary(results, results_dir)
        
        print("\n" + "=" * 50)
        print("Performance benchmark completed!")
        print("Check examples/results/ for plots and summary")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()