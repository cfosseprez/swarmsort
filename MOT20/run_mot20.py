"""
Run SwarmSort on the MOT20 Test Set with Multiple Modes.

This script provides two modes for running the SwarmSort tracker on the MOT20 dataset:

1. Benchmark Mode (Default):
   - Processes all sequences in the MOT20 test set.
   - Uses a motion-only configuration on public detections.
   - Efficiently saves results in the official MOTChallenge submission format.
   - Usage: `python MOT20/run_mot20.py`

2. Visualization Mode (--visualize):
   - Processes a single specified sequence.
   - Loads video frames and computes embeddings on-the-fly.
   - Shows a live visualization of the tracking process.
   - **Requires you to implement your own embedding model in `get_embedding()`**.
   - Usage: `python MOT20/run_mot20.py MOT20-04 --visualize`

Directory Structure:
--------------------
swarmsort/
├── MOT20/
│   ├── run_mot20.py  (this script)
│   └── test/
│       ├── MOT20-04/
│       │   ├── det/
│       │   └── ...
│       └── ... (other sequences)
└── swarmsort/
    └── ...
"""
import sys
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import configparser

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from swarmsort import SwarmSortTracker, SwarmSortConfig, Detection
from swarmsort.drawing_utils import TrackingVisualizer, VisualizationConfig

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


# ==============================================================================

def load_detections_by_frame(det_file: Path) -> dict:
    """Loads MOT detections and groups them by frame."""
    if not det_file.exists():
        return {}
    detections_raw = np.loadtxt(str(det_file), delimiter=",")
    detections_by_frame = {}
    for det in detections_raw:
        frame_id = int(det[0])
        if frame_id not in detections_by_frame:
            detections_by_frame[frame_id] = []
        detections_by_frame[frame_id].append(det)
    return detections_by_frame, int(np.max(detections_raw[:, 0]))

def run_benchmark_mode(data_root: Path, results_dir: Path, sequence_name: str = None):
    """Run sequences in motion-only mode for benchmark submission."""
    if sequence_name:
        print(f"Running in Benchmark Mode for single sequence: {sequence_name}")
        sequence_dirs = [data_root / sequence_name]
        if not sequence_dirs[0].exists():
            raise FileNotFoundError(f"Sequence directory not found: {sequence_dirs[0]}")
    else:
        print("Running in Benchmark Mode for all test sequences...")
        sequence_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    
    config = SwarmSortConfig(
        use_embeddings=True,
        use_probabilistic_costs=False,
        max_age=30,
        min_consecutive_detections=3,
        max_distance=150.0,
        reid_enabled=True,
    )

    for seq_dir in sequence_dirs:
        seq_name = seq_dir.name
        print(f"Processing sequence: {seq_name}")
        
        detections_by_frame, max_frame = load_detections_by_frame(seq_dir / "det" / "det.txt")
        if not detections_by_frame:
            print(f"  - No detections found for {seq_name}, skipping.")
            continue

        tracker = SwarmSortTracker(config)
        results = []
        for frame_id in tqdm(range(1, max_frame + 1), desc=f"  - Tracking {seq_name}"):
            frame_detections = []
            if frame_id in detections_by_frame:
                for det in detections_by_frame[frame_id]:
                    x1, y1, w, h, conf = det[2], det[3], det[4], det[5], det[6]
                    frame_detections.append(
                        Detection(position=np.array([x1 + w / 2, y1 + h / 2]), bbox=np.array([x1, y1, x1 + w, y1 + h]), confidence=conf)
                    )
            
            tracked_objects = tracker.update(frame_detections)
            for track in tracked_objects:
                x1, y1, x2, y2 = track.bbox
                results.append([frame_id, track.id, x1, y1, x2 - x1, y2 - y1, 1, -1, -1, -1])

        output_file = results_dir / f"{seq_name}.txt"
        np.savetxt(str(output_file), np.array(results), fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%.1f,%d,%d,%d")
        print(f"  - Results saved to {output_file}")

def run_visual_mode(seq_name: str, data_root: Path, results_dir: Path):
    """Run a single sequence with live embedding extraction and visualization."""
    if not OPENCV_AVAILABLE:
        raise ImportError("OpenCV is required for visualization mode. Please run `pip install opencv-python`.")
    
    print(f"Running in Visualization Mode for sequence: {seq_name}")
    seq_path = data_root / seq_name
    if not seq_path.exists():
        raise FileNotFoundError(f"Sequence directory not found: {seq_path}")

    config = SwarmSortConfig(
        use_embeddings=True, use_probabilistic_costs=True, embedding_weight=1,
        max_age=30, min_consecutive_detections=3, max_distance=150.0,
        reid_enabled=True, reid_max_distance=200.0,
    )

    seqinfo = configparser.ConfigParser()
    seqinfo.read(seq_path / "seqinfo.ini")
    img_dir = seq_path / seqinfo.get("Sequence", "imDir")
    img_ext = seqinfo.get("Sequence", "imExt")

    detections_by_frame, max_frame = load_detections_by_frame(seq_path / "det" / "det.txt")
    
    tracker = SwarmSortTracker(config)
    visualizer = TrackingVisualizer(VisualizationConfig(show_trails=True, trail_length=20))
    results = []

    for frame_id in tqdm(range(1, max_frame + 1), desc=f"  - Tracking {seq_name}"):
        img_path = img_dir / f"{frame_id:06d}{img_ext}"
        if not img_path.exists(): continue
        frame_img = cv2.imread(str(img_path))

        frame_detections = []
        if frame_id in detections_by_frame:
            for det in detections_by_frame[frame_id]:
                x1, y1, w, h, conf = det[2], det[3], det[4], det[5], det[6]
                patch = frame_img[int(y1):int(y1 + h), int(x1):int(x1 + w)]
                if patch.size == 0: continue

                frame_detections.append(
                    Detection(position=np.array([x1 + w / 2, y1 + h / 2]), bbox=np.array([x1, y1, x1 + w, y1 + h]), confidence=conf)
                )

        tracked_objects = tracker.update(frame_detections)
        
        output_frame = visualizer.draw_frame_opencv(frame_detections, tracked_objects, frame_num=frame_id, frame_img=frame_img)
        cv2.imshow(f"SwarmSort - {seq_name}", output_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        for track in tracked_objects:
            x1, y1, x2, y2 = track.bbox
            results.append([frame_id, track.id, x1, y1, x2 - x1, y2 - y1, 1, -1, -1, -1])

    cv2.destroyAllWindows()
    output_file = results_dir / f"{seq_name}.txt"
    np.savetxt(str(output_file), np.array(results), fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%.1f,%d,%d,%d")
    print(f"  - Results for {seq_name} saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Run SwarmSort on the MOT20 dataset.")
    parser.add_argument(
        "sequence", nargs="?", default=None,
        help="The name of the sequence to process (e.g., MOT20-04). If not provided, runs all sequences in benchmark mode."
    )
    parser.add_argument(
        "--visualize", action="store_true", default=True,
        help="Enable visualization mode for a single sequence. Requires a sequence name."
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    # The MOT20 dataset typically has 'train' and 'test' folders at the same level as this script.
    # We will target the 'test' folder as per the benchmark guidelines.
    data_root = base_dir / "test"
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

    if not data_root.exists():
        print(f"Error: Data directory not found at {data_root}")
        print("Please download the MOT20 dataset and ensure the 'test' folder is in the 'MOT20' directory.")
        return

    if args.visualize:
        if not args.sequence:
            print("Error: You must specify a sequence name for visualization mode.")
            print("Example: python MOT20/run_mot20.py MOT20-04 --visualize")
            return
        run_visual_mode(args.sequence, data_root, results_dir)
    else:
        run_benchmark_mode(data_root, results_dir, args.sequence)

    print("\nProcessing complete.")

if __name__ == "__main__":
    main()