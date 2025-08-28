"""
Basic usage examples for SwarmSort standalone package.

This script demonstrates how to use SwarmSort for multi-object tracking
in various scenarios.
"""
import numpy as np
from pathlib import Path
import sys

# Add the parent directory to path for importing swarmsort
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarmsort import (
    SwarmSort,
    SwarmSortConfig,
    Detection,
    create_tracker,
    is_within_swarmtracker,
    print_package_info,
)


def create_synthetic_detections(frame_num: int, num_objects: int = 3) -> list:
    """Create synthetic detections for testing."""
    detections = []

    # Object 1: Moving right
    pos1 = np.array([10.0 + frame_num * 2, 20.0 + np.sin(frame_num * 0.1) * 5])
    det1 = Detection(
        position=pos1,
        confidence=0.9,
        bbox=np.array([pos1[0] - 10, pos1[1] - 10, pos1[0] + 10, pos1[1] + 10]),
        embedding=np.random.randn(128) + np.array([1.0] * 128),  # Consistent embedding
        class_id=0,
        id=f"det1_{frame_num}",
    )
    detections.append(det1)

    # Object 2: Moving diagonally
    pos2 = np.array([50.0 + frame_num * 1.5, 50.0 + frame_num * 1.0])
    det2 = Detection(
        position=pos2,
        confidence=0.8,
        bbox=np.array([pos2[0] - 8, pos2[1] - 8, pos2[0] + 8, pos2[1] + 8]),
        embedding=np.random.randn(128) + np.array([-1.0] * 128),  # Different embedding
        class_id=1,
        id=f"det2_{frame_num}",
    )
    detections.append(det2)

    # Object 3: Circular motion (appears after frame 5)
    if frame_num > 5:
        center = np.array([80.0, 80.0])
        radius = 20.0
        angle = frame_num * 0.2
        pos3 = center + radius * np.array([np.cos(angle), np.sin(angle)])
        det3 = Detection(
            position=pos3,
            confidence=0.7,
            bbox=np.array([pos3[0] - 6, pos3[1] - 6, pos3[0] + 6, pos3[1] + 6]),
            embedding=np.random.randn(128),  # Random embedding
            class_id=2,
            id=f"det3_{frame_num}",
        )
        detections.append(det3)

    # Add some noise detections occasionally
    if frame_num % 7 == 0:
        noise_pos = np.random.rand(2) * 200
        noise_det = Detection(
            position=noise_pos, confidence=0.4, id=f"noise_{frame_num}"  # Low confidence
        )
        detections.append(noise_det)

    return detections


def example_basic_tracking():
    """Demonstrate basic tracking without embeddings."""
    print("=== Basic Tracking Example ===")

    # Create tracker with basic configuration
    config = SwarmSortConfig(
        max_distance=50.0,
        high_score_threshold=0.6,
        use_embeddings=False,  # Disable embeddings for this example
        min_consecutive_detections=2,
        detection_conf_threshold=0.5,
    )

    tracker = SwarmSort(config)

    print(f"Environment: {'SwarmTracker pipeline' if is_within_swarmtracker() else 'Standalone'}")
    print(f"Tracker config: max_distance={config.max_distance}, embeddings={config.use_embeddings}")

    # Process multiple frames
    all_tracks = []
    for frame_num in range(20):
        detections = create_synthetic_detections(
            frame_num, num_objects=2
        )  # Only 2 objects for simplicity

        tracked_objects = tracker.update(detections)

        print(
            f"Frame {frame_num:2d}: {len(detections)} detections -> {len(tracked_objects)} tracks"
        )

        # Store for visualization
        frame_tracks = []
        for track in tracked_objects:
            frame_tracks.append(
                {
                    "frame": frame_num,
                    "id": track.id,
                    "position": track.position.copy(),
                    "confidence": track.confidence,
                    "age": track.age,
                }
            )
        all_tracks.extend(frame_tracks)

    # Print final statistics
    stats = tracker.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Total frames: {stats['frame_count']}")
    print(f"  Active tracks: {stats['active_tracks']}")
    print(f"  Lost tracks: {stats['lost_tracks']}")
    print(f"  Next ID: {stats['next_id']}")

    return all_tracks


def example_embedding_tracking():
    """Demonstrate tracking with embeddings."""
    print("\n=== Embedding-based Tracking Example ===")

    # Create tracker with embeddings enabled
    config = SwarmSortConfig(
        max_distance=60.0,
        use_embeddings=True,
        embedding_weight=0.4,
        embedding_matching_method="best_match",
        max_embeddings_per_track=5,
        min_consecutive_detections=2,
    )

    tracker = SwarmSort(config)

    print(
        f"Tracker config: embedding_weight={config.embedding_weight}, "
        f"method={config.embedding_matching_method}"
    )

    # Process frames with embeddings
    for frame_num in range(15):
        detections = create_synthetic_detections(frame_num)
        tracked_objects = tracker.update(detections)

        print(
            f"Frame {frame_num:2d}: {len(detections)} detections -> {len(tracked_objects)} tracks"
        )

        # Show embedding scaler statistics periodically
        if frame_num % 5 == 0 and frame_num > 0:
            emb_stats = tracker.get_statistics()["embedding_scaler_stats"]
            if emb_stats["ready"]:
                print(
                    f"  Embedding scaler: method={emb_stats['method']}, "
                    f"samples={emb_stats['sample_count']}, ready={emb_stats['ready']}"
                )

    # Final statistics
    stats = tracker.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Active tracks: {stats['active_tracks']}")
    print(f"  Embedding scaler ready: {stats['embedding_scaler_stats']['ready']}")


def example_configuration_options():
    """Demonstrate different configuration options."""
    print("\n=== Configuration Examples ===")

    # Example 1: High-precision tracking
    print("1. High-precision config:")
    config1 = SwarmSortConfig(
        max_distance=30.0,  # Smaller distance threshold
        high_score_threshold=0.9,  # Higher confidence requirement
        min_consecutive_detections=4,  # More frames for initialization
        detection_conf_threshold=0.7,  # Higher detection confidence
    )
    tracker1 = create_tracker(config1)
    print(f"   Max distance: {config1.max_distance}")
    print(f"   Detection threshold: {config1.detection_conf_threshold}")

    # Example 2: Fast/permissive tracking
    print("2. Fast/permissive config:")
    config2 = SwarmSortConfig(
        max_distance=100.0,  # Larger distance threshold
        high_score_threshold=0.5,  # Lower confidence requirement
        min_consecutive_detections=1,  # Immediate initialization
        detection_conf_threshold=0.3,  # Lower detection confidence
    )
    tracker2 = create_tracker(config2)
    print(f"   Max distance: {config2.max_distance}")
    print(f"   Min consecutive: {config2.min_consecutive_detections}")

    # Example 3: Re-identification enabled
    print("3. ReID-enabled config:")
    config3 = SwarmSortConfig(
        reid_enabled=True, reid_max_distance=150.0, reid_embedding_threshold=0.3, reid_max_frames=15
    )
    tracker3 = create_tracker(config3)
    print(f"   ReID enabled: {config3.reid_enabled}")
    print(f"   ReID max distance: {config3.reid_max_distance}")


def example_factory_usage():
    """Demonstrate factory function usage."""
    print("\n=== Factory Function Examples ===")

    # Default tracker
    tracker1 = create_tracker()
    print("1. Default tracker created")

    # Tracker with dict config
    tracker2 = create_tracker({"max_distance": 75.0, "use_embeddings": True})
    print(f"2. Dict config tracker: max_distance={tracker2.config.max_distance}")

    # Force standalone mode
    tracker3 = create_tracker(force_standalone=True)
    print("3. Forced standalone tracker created")

    # Using SwarmSortConfig
    config = SwarmSortConfig(embedding_weight=0.6)
    tracker4 = create_tracker(config)
    print(f"4. Config object tracker: embedding_weight={tracker4.config.embedding_weight}")


def plot_tracking_results(all_tracks):
    """Plot tracking results if matplotlib is available."""
    if not all_tracks:
        return

    try:
        import matplotlib.pyplot as plt

        # Group tracks by ID
        track_data = {}
        for track in all_tracks:
            track_id = track["id"]
            if track_id not in track_data:
                track_data[track_id] = {"frames": [], "positions": [], "ages": []}

            track_data[track_id]["frames"].append(track["frame"])
            track_data[track_id]["positions"].append(track["position"])
            track_data[track_id]["ages"].append(track["age"])

        # Create plot
        plt.figure(figsize=(12, 8))

        # Plot trajectories
        plt.subplot(2, 2, 1)
        colors = plt.cm.tab10(np.linspace(0, 1, len(track_data)))
        for i, (track_id, data) in enumerate(track_data.items()):
            positions = np.array(data["positions"])
            plt.plot(
                positions[:, 0],
                positions[:, 1],
                "o-",
                color=colors[i],
                label=f"Track {track_id}",
                linewidth=2,
                markersize=4,
            )

        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Object Trajectories")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot track ages over time
        plt.subplot(2, 2, 2)
        for i, (track_id, data) in enumerate(track_data.items()):
            plt.plot(data["frames"], data["ages"], "o-", color=colors[i], label=f"Track {track_id}")

        plt.xlabel("Frame")
        plt.ylabel("Track Age")
        plt.title("Track Ages Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot number of active tracks per frame
        plt.subplot(2, 2, 3)
        frame_counts = {}
        for track in all_tracks:
            frame = track["frame"]
            frame_counts[frame] = frame_counts.get(frame, 0) + 1

        frames = sorted(frame_counts.keys())
        counts = [frame_counts[f] for f in frames]
        plt.plot(frames, counts, "o-", linewidth=2, markersize=6)
        plt.xlabel("Frame")
        plt.ylabel("Active Tracks")
        plt.title("Active Tracks Per Frame")
        plt.grid(True, alpha=0.3)

        # Plot track lifetime distribution
        plt.subplot(2, 2, 4)
        lifetimes = []
        for track_id, data in track_data.items():
            lifetime = max(data["ages"])
            lifetimes.append(lifetime)

        plt.hist(lifetimes, bins=max(1, len(lifetimes) // 2), alpha=0.7, edgecolor="black")
        plt.xlabel("Track Lifetime (frames)")
        plt.ylabel("Count")
        plt.title("Track Lifetime Distribution")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("swarmsort_tracking_results.png", dpi=150, bbox_inches="tight")
        plt.show()

        print(f"\nPlot saved as 'swarmsort_tracking_results.png'")

    except ImportError:
        print("Matplotlib not available - skipping visualization")
    except Exception as e:
        print(f"Error creating plot: {e}")


def example_visualization_capabilities():
    """Demonstrate visualization and simulation capabilities."""
    print("\n=== Visualization Capabilities ===")

    try:
        from swarmsort import (
            TrackingVisualizer,
            ObjectMotionSimulator,
            create_demo_scenario,
            quick_visualize,
            VISUALIZATION_AVAILABLE,
        )

        if not VISUALIZATION_AVAILABLE:
            print("Visualization tools not available (missing matplotlib/opencv)")
            return

        print("Creating simple simulation for visualization demo...")

        # Create a simple crossing paths scenario
        sim = create_demo_scenario("crossing_paths")
        tracker = SwarmSort()

        # Run a few frames
        detection_sequences = []
        track_sequences = []

        for frame in range(20):
            detections = sim.step()
            tracks = tracker.update(detections)
            detection_sequences.append(detections)
            track_sequences.append(tracks)

        # Show final frame using basic matplotlib if available
        if detection_sequences and track_sequences:
            print("Basic visualization capabilities available!")
            print(
                "Final frame: {} detections, {} tracks".format(
                    len(detection_sequences[-1]), len(track_sequences[-1])
                )
            )

        print(
            "For advanced visualization examples, run: python examples/visualization_demo.py [num_objects] [num_frames]"
        )

    except ImportError as e:
        print(f"Advanced visualization not available: {e}")
        print("Install with: pip install matplotlib opencv-python")


def example_performance_measurement():
    """Demonstrate performance measurement capabilities."""
    print("\n=== Performance Measurement ===")

    try:
        from swarmsort import quick_benchmark, timing_context

        # Create some test data
        detections_sequence = []
        for frame in range(50):
            frame_detections = []
            for i in range(5):  # 5 objects
                pos = np.array([50 + i * 100 + frame * 2, 100 + i * 50])
                det = Detection(
                    position=pos,
                    confidence=0.9,
                    bbox=np.array([pos[0] - 10, pos[1] - 10, pos[0] + 10, pos[1] + 10]),
                )
                frame_detections.append(det)
            detections_sequence.append(frame_detections)

        # Run benchmark
        print("Running performance benchmark...")
        with timing_context("Benchmark"):
            result = quick_benchmark(detections_sequence, verbose=False)

        print(f"Performance Results:")
        print(f"  Average FPS: {result.average_fps:.1f}")
        print(f"  Processing time: {result.avg_processing_time_ms:.2f}ms per frame")
        print(f"  Peak memory: {result.peak_memory_mb:.1f}MB")

        print("For comprehensive benchmarks, run: python examples/performance_benchmark.py")

    except ImportError:
        print("Performance measurement tools not available")


def main():
    """Run all examples."""
    print("SwarmSort Standalone Package - Usage Examples")
    print("=" * 50)

    # Print package information
    print_package_info()
    print()

    # Basic tracking
    all_tracks = example_basic_tracking()

    # Embedding tracking
    example_embedding_tracking()

    # Configuration options
    example_configuration_options()

    # Factory usage
    example_factory_usage()

    # New capabilities
    example_visualization_capabilities()
    example_performance_measurement()

    # Basic visualization
    plot_tracking_results(all_tracks)

    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print("\nAdditional Examples Available:")
    print("- python examples/visualization_demo.py  (comprehensive visualization)")
    print("- python examples/performance_benchmark.py  (performance testing)")


if __name__ == "__main__":
    main()
