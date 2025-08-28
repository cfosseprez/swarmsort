"""
SwarmSort Visualization Demo

This script demonstrates the visualization capabilities of SwarmSort using
a simple random walk simulation with a tunable number of objects.
"""
import numpy as np
import sys
import os
from pathlib import Path
import time

# Add the parent directory to path for importing swarmsort
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarmsort import SwarmSortTracker, SwarmSortConfig
from swarmsort.simulator import ObjectMotionSimulator, SimulationConfig
from swarmsort.drawing_utils import TrackingVisualizer, VisualizationConfig, quick_visualize
from swarmsort.benchmarking import quick_benchmark

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


def create_random_walk_simulation(num_objects: int = 10, 
                                 world_width: float = 1920, 
                                 world_height: float = 1080) -> ObjectMotionSimulator:
    """Create a simulation with random walk objects."""
    
    # Create simulation config
    sim_config = SimulationConfig(
        world_width=world_width,
        world_height=world_height,
        detection_probability=1,
        false_positive_rate=0,
        position_noise_std=2.0,
        use_embeddings=True,
        random_seed=42  # For reproducible results
    )
    
    sim = ObjectMotionSimulator(sim_config)
    
    # Add random walk objects
    np.random.seed(42)  # Reproducible placement
    
    for i in range(num_objects):
        # Random starting position
        start_x = np.random.uniform(50, world_width - 50)
        start_y = np.random.uniform(50, world_height - 50)
        
        # Very small step size - just a few pixels per frame
        step_size = np.random.uniform(0.01, 0.03)
        
        # Random confidence and class
        base_confidence = 1 #np.random.uniform(0.7, 0.95)
        class_id = i % 5  # 5 different classes
        
        obj = sim.create_random_walk_object(
            object_id=i+1,
            start_pos=(start_x, start_y),
            step_size=step_size,  # Explicitly pass the small step size
            class_id=class_id,
            base_confidence=base_confidence
        )
        
        sim.add_object(obj)
    
    return sim


def run_visualization_demo(num_objects: int = 50, realtime: bool = True):
    """Run visualization demo, optionally in realtime with OpenCV imshow."""
    print("SwarmSort Random Walk Visualization Demo")
    print("=" * 50)
    print(f"Configuration: {num_objects} objects")

    if not OPENCV_AVAILABLE:
        print("OpenCV not available, install with `pip install opencv-python`.")
        return

    # Create simulation
    print("Creating random walk simulation...")
    sim = create_random_walk_simulation(num_objects)

    # Create tracker with embeddings
    config = SwarmSortConfig(
        max_distance=100,
        use_embeddings=True,
        embedding_weight=1,
        min_consecutive_detections=3,
        debug_timings=True,
    )
    tracker = SwarmSortTracker(config)

    # Visualization config
    vis_config = VisualizationConfig(
        frame_width=1080,
        frame_height=1080,
        show_trails=True,
        trail_length=15,
        show_confidences=True,
        show_ids=True,
        show_velocities=False
    )
    visualizer = TrackingVisualizer(vis_config)

    print("Starting endless simulation. Press 'q' to quit.")

    frame = 0
    prev_time = time.time()
    fps = 0.0

    while True:


        # Step simulation
        detections = sim.step()

        start_time = time.time()
        tracks = tracker.update(detections)
        # Calculate FPS (tracking loop)
        end_time = time.time()
        fps = 1.0 / (end_time - start_time + 1e-8)

        # Render frame
        frame_img = visualizer.draw_frame_opencv(detections, tracks, frame_num=frame)

        # Print tracker timing in terminal
        print(f"[Frame {frame}] Tracker time: {(end_time - start_time)*1000:.0f} ms")
        
        # Overlay FPS on frame
        cv2.putText(frame_img, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Show in OpenCV
        cv2.imshow("SwarmSort Realtime Simulation", frame_img)

        # Press q to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        frame += 1

    cv2.destroyAllWindows()
    print("Simulation stopped by user.")

def main():
    """Main demo function with configurable parameters."""
    print("SwarmSort Random Walk Visualization Demo")
    print("=" * 50)
    
    # Default parameters
    default_objects = 150
    default_frames = 300
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        try:
            num_objects = int(sys.argv[1])
            if num_objects < 1 or num_objects > 50:
                raise ValueError("Number of objects must be between 1 and 50")
        except ValueError as e:
            print(f"Invalid number of objects: {e}")
            print(f"Using default: {default_objects}")
            num_objects = default_objects
    else:
        num_objects = default_objects
    
    if len(sys.argv) > 2:
        try:
            num_frames = int(sys.argv[2])
            if num_frames < 10 or num_frames > 1000:
                raise ValueError("Number of frames must be between 10 and 1000")
        except ValueError as e:
            print(f"Invalid number of frames: {e}")
            print(f"Using default: {default_frames}")
            num_frames = default_frames
    else:
        num_frames = default_frames
    
    # Run the demo
    try:
        detection_sequences, track_sequences = run_visualization_demo(num_objects, num_frames)
        
        print("\n" + "=" * 50)
        print("Visualization demo completed successfully!")
        print("Generated files in examples/results/:")
        print(f"- random_walk_demo_{num_objects}objects_{num_frames}frames.mp4 (if OpenCV available)")
        print(f"- random_walk_demo_{num_objects}objects.gif (if Matplotlib available)")
        
        print(f"\nUsage: python {sys.argv[0]} [num_objects] [num_frames]")
        print(f"Example: python {sys.argv[0]} 15 500")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()