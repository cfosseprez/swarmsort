"""
SwarmSort Performance Benchmark Script

This script runs comprehensive performance benchmarks for SwarmSort,
testing scalability with increasing object counts over 300 frames and
generating detailed performance reports.
"""
import numpy as np
import sys
import time
from pathlib import Path
from typing import List, Dict

# Add the parent directory to path for importing swarmsort
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarmsort import SwarmSortTracker, SwarmSortConfig
from swarmsort.data_classes import Detection
from swarmsort.simulator import ObjectMotionSimulator, SimulationConfig, MotionType
from swarmsort.benchmarking import (
    TrackingBenchmark, ScalabilityBenchmark, BenchmarkResult, quick_benchmark
)

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def generate_scalability_detections(num_objects: int, num_frames: int) -> List[List[Detection]]:
    """Generate detection sequences for scalability testing."""
    # Create simulation config optimized for benchmarking
    sim_config = SimulationConfig(
        world_width=1000,
        world_height=800,
        detection_probability=0.95,
        false_positive_rate=0.01,
        position_noise_std=1.5,
        use_embeddings=True,
        random_seed=42  # For reproducible results
    )
    
    sim = ObjectMotionSimulator(sim_config)
    
    # Add objects with diverse motion patterns
    np.random.seed(42)  # Reproducible object placement
    
    for i in range(num_objects):
        # Randomize motion type and parameters
        motion_type = np.random.choice([
            MotionType.LINEAR,
            MotionType.CIRCULAR,
            MotionType.RANDOM_WALK,
            MotionType.FIGURE_EIGHT
        ])
        
        start_x = np.random.uniform(50, sim_config.world_width - 50)
        start_y = np.random.uniform(50, sim_config.world_height - 50)
        
        if motion_type == MotionType.LINEAR:
            velocity = np.random.uniform(-3, 3, 2)
            obj = sim.create_linear_motion_object(
                object_id=i+1,
                start_pos=(start_x, start_y),
                velocity=tuple(velocity),
                class_id=i % 5
            )
        
        elif motion_type == MotionType.CIRCULAR:
            radius = np.random.uniform(30, 80)
            angular_vel = np.random.uniform(-0.05, 0.05)
            obj = sim.create_circular_motion_object(
                object_id=i+1,
                center=(start_x, start_y),
                radius=radius,
                angular_velocity=angular_vel,
                class_id=i % 5
            )
        
        elif motion_type == MotionType.RANDOM_WALK:
            step_size = np.random.uniform(1, 4)
            obj = sim.create_random_walk_object(
                object_id=i+1,
                start_pos=(start_x, start_y),
                step_size=step_size,
                class_id=i % 5
            )
        
        elif motion_type == MotionType.FIGURE_EIGHT:
            width = np.random.uniform(30, 60)
            height = np.random.uniform(20, 40)
            period = np.random.uniform(80, 120)
            obj = sim.create_figure_eight_object(
                object_id=i+1,
                center=(start_x, start_y),
                width=width,
                height=height,
                period=period,
                class_id=i % 5
            )
        
        sim.add_object(obj)
    
    # Generate detection sequence
    return sim.run_simulation(num_frames)


def run_comprehensive_benchmark():
    """Run comprehensive performance benchmark."""
    print("SwarmSort Performance Benchmark")
    print("=" * 60)
    print(f"Test configuration: 300 frames with varying object counts")
    print("=" * 60)
    
    # Test configurations
    object_counts = [3, 5, 10, 15, 20, 25, 30]
    num_frames = 300
    
    # Create scalability benchmark
    scalability = ScalabilityBenchmark()
    
    # Test different tracker configurations
    configs = {
        "basic": SwarmSortConfig(
            max_distance=50.0,
            use_embeddings=False
        ),
        "embeddings": SwarmSortConfig(
            max_distance=50.0,
            use_embeddings=True,
            embedding_weight=0.3
        ),
        "high_performance": SwarmSortConfig(
            max_distance=60.0,
            use_embeddings=True,
            embedding_weight=0.4,
            reid_enabled=False  # Disable ReID for performance
        )
    }
    
    all_results = {}
    
    for config_name, config in configs.items():
        print(f"\nTesting configuration: {config_name}")
        print("-" * 40)
        
        results = scalability.run_object_count_scalability(
            detection_generator=generate_scalability_detections,
            object_counts=object_counts,
            num_frames=num_frames,
            config=config
        )
        
        all_results[config_name] = results
    
    # Generate reports
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    # Print summary table
    print(f"{'Config':<15} {'Objects':<8} {'Avg FPS':<10} {'Avg Time(ms)':<12} {'Peak Mem(MB)':<12}")
    print("-" * 65)
    
    for config_name, results in all_results.items():
        for test_name, result in results.items():
            objects = test_name.split('_')[0]
            print(f"{config_name:<15} {objects:<8} {result.average_fps:<10.1f} "
                  f"{result.avg_processing_time_ms:<12.2f} {result.peak_memory_mb:<12.1f}")
    
    # Save detailed results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    for config_name, results in all_results.items():
        for test_name, result in results.items():
            filename = f"benchmark_{config_name}_{test_name}_{timestamp}.json"
            benchmark = TrackingBenchmark()
            benchmark.save_results(result, filename)
    
    # Generate performance report
    scalability.results = {f"{config_name}_{test_name}": result 
                          for config_name, results in all_results.items() 
                          for test_name, result in results.items()}
    
    report_filename = f"performance_report_{timestamp}.json"
    scalability.generate_performance_report(report_filename)
    
    # Create performance plots if matplotlib is available
    if MATPLOTLIB_AVAILABLE:
        create_performance_plots(all_results, timestamp)
    
    return all_results


def create_performance_plots(results: Dict, timestamp: str):
    """Create performance visualization plots."""
    print(f"\nCreating performance plots...")
    
    # Extract data for plotting
    configs = list(results.keys())
    object_counts = []
    fps_data = {config: [] for config in configs}
    memory_data = {config: [] for config in configs}
    time_data = {config: [] for config in configs}
    
    # Get object counts from first config
    first_config = list(results.values())[0]
    object_counts = sorted([int(k.split('_')[0]) for k in first_config.keys()])
    
    # Extract performance data
    for config_name, config_results in results.items():
        for obj_count in object_counts:
            test_key = f"{obj_count}_objects"
            if test_key in config_results:
                result = config_results[test_key]
                fps_data[config_name].append(result.average_fps)
                memory_data[config_name].append(result.peak_memory_mb)
                time_data[config_name].append(result.avg_processing_time_ms)
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # FPS vs Object Count
    for config_name in configs:
        ax1.plot(object_counts, fps_data[config_name], marker='o', label=config_name)
    ax1.set_xlabel('Number of Objects')
    ax1.set_ylabel('Average FPS')
    ax1.set_title('Tracking Performance (FPS) vs Object Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Processing Time vs Object Count
    for config_name in configs:
        ax2.plot(object_counts, time_data[config_name], marker='s', label=config_name)
    ax2.set_xlabel('Number of Objects')
    ax2.set_ylabel('Average Processing Time (ms)')
    ax2.set_title('Processing Time vs Object Count')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Memory Usage vs Object Count
    for config_name in configs:
        ax3.plot(object_counts, memory_data[config_name], marker='^', label=config_name)
    ax3.set_xlabel('Number of Objects')
    ax3.set_ylabel('Peak Memory Usage (MB)')
    ax3.set_title('Memory Usage vs Object Count')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Efficiency Plot (FPS per MB)
    for config_name in configs:
        efficiency = [fps/mem if mem > 0 else 0 
                     for fps, mem in zip(fps_data[config_name], memory_data[config_name])]
        ax4.plot(object_counts, efficiency, marker='d', label=config_name)
    ax4.set_xlabel('Number of Objects')
    ax4.set_ylabel('Efficiency (FPS per MB)')
    ax4.set_title('Memory Efficiency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plots
    plot_filename = f"performance_plots_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Performance plots saved as {plot_filename}")
    
    # Show plots
    plt.show()


def run_single_config_benchmark():
    """Run a detailed benchmark for a single configuration."""
    print("\nRunning detailed single-configuration benchmark...")
    
    # Generate detection sequence
    num_objects = 15
    num_frames = 300
    
    print(f"Generating {num_frames} frames with {num_objects} objects...")
    detection_sequence = generate_scalability_detections(num_objects, num_frames)
    
    # Create optimized config
    config = SwarmSortConfig(
        max_distance=60.0,
        use_embeddings=True,
        embedding_weight=0.4,
        min_consecutive_detections=3
    )
    
    # Run detailed benchmark
    tracker = SwarmSortTracker(config)
    benchmark = TrackingBenchmark(tracker, enable_detailed_profiling=True)
    
    print("Running benchmark with detailed profiling...")
    
    def progress_callback(frame, total):
        if frame % 50 == 0:
            print(f"  Processing frame {frame}/{total}")
    
    result = benchmark.benchmark_sequence(detection_sequence, progress_callback)
    
    # Print detailed results
    print(f"\nDetailed Benchmark Results:")
    print(f"Total frames: {result.total_frames}")
    print(f"Total time: {result.total_time_ms/1000:.2f} seconds")
    print(f"Average FPS: {result.average_fps:.1f}")
    print(f"Processing time - Mean: {result.avg_processing_time_ms:.2f}ms, "
          f"Std: {result.std_processing_time_ms:.2f}ms")
    print(f"Processing time - Min: {result.min_processing_time_ms:.2f}ms, "
          f"Max: {result.max_processing_time_ms:.2f}ms")
    print(f"Memory - Average: {result.avg_memory_mb:.1f}MB, "
          f"Peak: {result.peak_memory_mb:.1f}MB")
    print(f"Tracks - Created: {result.total_tracks_created}, "
          f"Average active: {result.avg_active_tracks:.1f}")
    
    # Save detailed results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"detailed_benchmark_{timestamp}.json"
    benchmark.save_results(result, filename)
    
    return result


def run_memory_stress_test():
    """Run a memory stress test with many objects."""
    print("\nRunning memory stress test...")
    
    object_counts = [50, 75, 100]
    num_frames = 100  # Shorter sequence for stress test
    
    for num_objects in object_counts:
        print(f"\nStress testing with {num_objects} objects...")
        
        try:
            # Generate detections
            detection_sequence = generate_scalability_detections(num_objects, num_frames)
            
            # Create memory-optimized config
            config = SwarmSortConfig(
                max_distance=50.0,
                use_embeddings=False,  # Disable to save memory
                max_lost_time=30,  # Shorter retention
                reid_enabled=False
            )
            
            # Run benchmark
            result = quick_benchmark(detection_sequence, config, verbose=True)
            
            print(f"Result: {result.average_fps:.1f} FPS, "
                  f"{result.peak_memory_mb:.1f}MB peak memory")
        
        except MemoryError:
            print(f"Memory error with {num_objects} objects - reached system limits")
            break
        except Exception as e:
            print(f"Error with {num_objects} objects: {e}")


def main():
    """Run all benchmarks."""
    print("SwarmSort Performance Benchmark Suite")
    print("=" * 50)
    
    # Check what's available
    print("Available features:")
    print(f"- Matplotlib: {'Yes' if MATPLOTLIB_AVAILABLE else 'No'}")
    print()
    
    try:
        # Run comprehensive benchmark
        all_results = run_comprehensive_benchmark()
        
        # Run detailed single config benchmark
        detailed_result = run_single_config_benchmark()
        
        # Optional memory stress test
        response = input("\nRun memory stress test? (y/n): ").lower().strip()
        if response == 'y':
            run_memory_stress_test()
        
        print("\n" + "=" * 50)
        print("Benchmark suite completed!")
        print("Check the current directory for JSON reports and plots.")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()