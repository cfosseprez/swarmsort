"""
Run SwarmSort on the MOT20 Test Set with Multiple Modes.

This script provides two modes for running the SwarmSort tracker on the MOT20 dataset:

1. Benchmark Mode (Default):
   - Processes all sequences in the MOT20 test set.
   - Uses motion tracking enhanced with CuPy visual embeddings from image patches.
   - Includes ReID capabilities for recovering lost tracks in crowded scenes.
   - Efficiently saves results in the official MOTChallenge submission format.
   - Usage: `python MOT20/run_mot20.py`
   - For submission: `python MOT20/run_mot20.py --create-zip`

2. Visualization Mode (--visualize):
   - Single sequence: `python MOT20/run_mot20.py MOT20-04 --visualize`
   - All sequences: `python MOT20/run_mot20.py --visualize`
   - Shows real-time tracking visualization with trails and IDs.
   - Uses ReID for better tracking performance.
   - Controls: 'q' to skip sequence, 'ESC' to quit all.
   - For submission with visualization: `python MOT20/run_mot20.py --visualize --create-zip`

MOTChallenge Submission:
-----------------------
1. Run: `python MOT20/run_mot20.py --create-zip`
2. Upload the generated ZIP file to https://motchallenge.net/
3. The ZIP contains results for all 8 MOT20 test sequences in the required format

Directory Structure:
--------------------
swarmsort/
├── MOT20/
│   ├── run_mot20.py  (this script)
│   ├── results/      (generated results)
│   └── test/
│       ├── MOT20-01/ to MOT20-08/
│       │   ├── det/det.txt
│       │   ├── img1/
│       │   └── seqinfo.ini
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

from swarmsort import SwarmSortTracker, SwarmSortConfig, Detection, get_embedding_extractor
from swarmsort.drawing_utils import TrackingVisualizer, VisualizationConfig

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


# ==============================================================================
# TRACKER CONFIGURATION
# ==============================================================================
"""
Centralized tracker configurations for MOT20 evaluation.

Key Parameters to Tune:
- embedding_weight: Balance between motion (0.0) and appearance (1.0) cues
- max_distance: Maximum pixel distance for track-detection association
- reid_max_distance: Maximum distance for re-identifying lost tracks  
- reid_embedding_threshold: Embedding similarity threshold (0-1, lower=more permissive)
- min_consecutive_detections: Frames required before confirming new track
- max_age: Frames to keep track alive without detections
"""

# MOT20 Tracker Configuration - Tune these parameters as needed
MOT20_CONFIG = SwarmSortConfig(
    # Core tracking parameters
    do_embeddings=True,              # Enable visual embeddings for better association
    use_probabilistic_costs=False,    # Use deterministic assignment for benchmark
    embedding_weight=1,             # Balance between motion and appearance (0=motion only, 1=equal)
    
    # Track lifecycle parameters  
    max_track_age=40,                       # Keep tracks alive for N frames without detections
    min_consecutive_detections=3,     # Require 3 consecutive detections before confirming track (default)

    # Distance thresholds
    max_distance=50.0,               # Maximum distance for track-detection association
    reid_enabled=True,                # Enable re-identification for lost tracks
    reid_max_distance=100.0,          # Maximum distance for ReID associations
    reid_embedding_threshold=0.4,     # Minimum embedding similarity for ReID (0-1, lower=more permissive)
)


# ==============================================================================


def load_detections_by_frame(det_file: Path) -> tuple:
    """Loads MOT detections and groups them by frame.
    
    MOT20 format: frame,id,left,top,width,height,conf,x,y,z
    Note: id=-1 for detections, actual tracking IDs are assigned by tracker
    """
    if not det_file.exists():
        return {}, 0
    detections_raw = np.loadtxt(str(det_file), delimiter=",")
    if len(detections_raw.shape) == 1:
        detections_raw = detections_raw.reshape(1, -1)
    
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
    
    # Use centralized MOT20 configuration
    config = MOT20_CONFIG

    # Initialize embedding extractor for benchmark mode
    embedding_extractor = get_embedding_extractor("cupytexture_color")
    
    for seq_dir in sequence_dirs:
        seq_name = seq_dir.name
        print(f"Processing sequence: {seq_name}")
        
        # Get sequence info for image loading
        seqinfo = configparser.ConfigParser()
        seqinfo_file = seq_dir / "seqinfo.ini"
        img_dir = seq_dir / "img1"  # default
        img_ext = ".jpg"  # default
        if seqinfo_file.exists():
            seqinfo.read(seqinfo_file)
            try:
                img_dir = seq_dir / seqinfo.get("Sequence", "imDir")
                img_ext = seqinfo.get("Sequence", "imExt")
            except:
                pass  # use defaults
        
        detections_by_frame, max_frame = load_detections_by_frame(seq_dir / "det" / "det.txt")
        if not detections_by_frame:
            print(f"  - No detections found for {seq_name}, skipping.")
            continue

        tracker = SwarmSortTracker(config)
        results = []
        for frame_id in tqdm(range(1, max_frame + 1), desc=f"  - Tracking {seq_name}"):
            # Load frame image for embedding extraction
            img_path = img_dir / f"{frame_id:06d}{img_ext}"
            frame_img = None
            if img_path.exists():
                frame_img = cv2.imread(str(img_path))
            
            frame_detections = []
            if frame_id in detections_by_frame:
                for det in detections_by_frame[frame_id]:
                    # MOT20 format: frame,id,left,top,width,height,conf,x,y,z
                    x1, y1, w, h, conf = det[2], det[3], det[4], det[5], det[6]
                    # In MOT20, confidence is binary (0 or 1), accept all valid detections
                    if conf < 0.0:  # Only skip negative confidence (invalid detections)
                        continue
                    
                    bbox = np.array([x1, y1, x1 + w, y1 + h])
                    
                    # Extract visual embedding from image patch if available
                    embedding = None
                    if frame_img is not None:
                        patch = frame_img[int(y1):int(y1 + h), int(x1):int(x1 + w)]
                        if patch.size > 0:
                            try:
                                embedding = embedding_extractor.extract(patch)
                            except:
                                embedding = None  # fallback to no embedding
                    
                    # MOT20 has binary confidence (0 or 1), but 0 doesn't mean bad detection
                    # Set confidence to 1.0 for all valid detections since they passed quality filtering
                    detection_conf = 1.0 if conf >= 0.0 else 0.5
                    
                    frame_detections.append(
                        Detection(
                            position=np.array([x1 + w / 2, y1 + h / 2]), 
                            bbox=bbox, 
                            confidence=detection_conf,
                            embedding=embedding
                        )
                    )
            
            tracked_objects = tracker.update(frame_detections)
            for track in tracked_objects:
                x1, y1, x2, y2 = track.bbox
                results.append([frame_id, track.id, x1, y1, x2 - x1, y2 - y1, 1, -1, -1, -1])

        # Save results in MOT20 format
        output_file = results_dir / f"{seq_name}.txt"
        if results:
            np.savetxt(str(output_file), np.array(results), fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d", delimiter=',')
            print(f"  - Results saved to {output_file} ({len(results)} tracks)")
        else:
            # Create empty file if no tracks
            with open(output_file, 'w') as f:
                pass
            print(f"  - No tracks found, empty file created: {output_file}")

def run_visual_mode(seq_name: str, data_root: Path, results_dir: Path):
    """Run a single sequence with live embedding extraction and visualization."""
    if not OPENCV_AVAILABLE:
        raise ImportError("OpenCV is required for visualization mode. Please run `pip install opencv-python`.")
    
    print(f"Running in Visualization Mode for sequence: {seq_name}")
    seq_path = data_root / seq_name
    if not seq_path.exists():
        raise FileNotFoundError(f"Sequence directory not found: {seq_path}")

    # Use centralized visualization configuration
    config = MOT20_CONFIG

    seqinfo = configparser.ConfigParser()
    seqinfo.read(seq_path / "seqinfo.ini")
    img_dir = seq_path / seqinfo.get("Sequence", "imDir")
    img_ext = seqinfo.get("Sequence", "imExt")

    detections_by_frame, max_frame = load_detections_by_frame(seq_path / "det" / "det.txt")
    
    # Initialize embedding extractor for visualization mode  
    embedding_extractor = get_embedding_extractor("cupytexture")
    
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
                # MOT20 format: frame,id,left,top,width,height,conf,x,y,z
                x1, y1, w, h, conf = det[2], det[3], det[4], det[5], det[6]
                # In MOT20, confidence is binary (0 or 1), accept all valid detections
                if conf < 0.0:  # Only skip negative confidence (invalid detections)
                    continue
                    
                patch = frame_img[int(y1):int(y1 + h), int(x1):int(x1 + w)]
                if patch.size == 0: continue
                
                # Extract visual embedding from patch
                embedding = None
                try:
                    embedding = embedding_extractor.extract(patch)
                except:
                    embedding = None  # fallback to no embedding

                # MOT20 has binary confidence (0 or 1), but 0 doesn't mean bad detection
                # Set confidence to 1.0 for all valid detections since they passed quality filtering
                detection_conf = 1.0 if conf >= 0.0 else 0.5
                
                frame_detections.append(
                    Detection(
                        position=np.array([x1 + w / 2, y1 + h / 2]), 
                        bbox=np.array([x1, y1, x1 + w, y1 + h]), 
                        confidence=detection_conf,
                        embedding=embedding
                    )
                )

        tracked_objects = tracker.update(frame_detections)
        
        output_frame = visualizer.draw_frame_opencv(frame_detections, tracked_objects, frame_num=frame_id, frame=frame_img)
        cv2.imshow(f"SwarmSort - {seq_name}", output_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        for track in tracked_objects:
            x1, y1, x2, y2 = track.bbox
            results.append([frame_id, track.id, x1, y1, x2 - x1, y2 - y1, 1, -1, -1, -1])

    cv2.destroyAllWindows()
    
    # Save results in MOT20 format
    output_file = results_dir / f"{seq_name}.txt"
    if results:
        np.savetxt(str(output_file), np.array(results), fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d", delimiter=',')
        print(f"  - Results for {seq_name} saved to {output_file} ({len(results)} tracks)")
    else:
        # Create empty file if no tracks
        with open(output_file, 'w') as f:
            pass
        print(f"  - No tracks found, empty file created: {output_file}")

def run_visual_mode_all_sequences(data_root: Path, results_dir: Path):
    """Run all sequences in visualization mode with real-time display."""
    if not OPENCV_AVAILABLE:
        raise ImportError("OpenCV is required for visualization mode. Please run `pip install opencv-python`.")
    
    print("Running in Visualization Mode for all test sequences...")
    sequence_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    
    # Use centralized visualization configuration
    config = MOT20_CONFIG
    
    # Initialize embedding extractor
    embedding_extractor = get_embedding_extractor("cupytexture")
    
    for seq_dir in sequence_dirs:
        seq_name = seq_dir.name
        print(f"\nProcessing sequence: {seq_name}")
        print("Press 'q' to skip to next sequence, 'ESC' to quit all")
        
        # Get sequence info
        seqinfo = configparser.ConfigParser()
        seqinfo_file = seq_dir / "seqinfo.ini"
        img_dir = seq_dir / "img1"
        img_ext = ".jpg"
        if seqinfo_file.exists():
            seqinfo.read(seqinfo_file)
            try:
                img_dir = seq_dir / seqinfo.get("Sequence", "imDir")
                img_ext = seqinfo.get("Sequence", "imExt")
            except:
                pass
        
        detections_by_frame, max_frame = load_detections_by_frame(seq_dir / "det" / "det.txt")
        if not detections_by_frame:
            print(f"  - No detections found for {seq_name}, skipping.")
            continue
        
        tracker = SwarmSortTracker(config)
        visualizer = TrackingVisualizer(VisualizationConfig(show_trails=True, trail_length=20))
        results = []
        
        skip_sequence = False
        quit_all = False
        
        for frame_id in tqdm(range(1, max_frame + 1), desc=f"  - Tracking {seq_name}"):
            img_path = img_dir / f"{frame_id:06d}{img_ext}"
            if not img_path.exists(): 
                continue
            frame_img = cv2.imread(str(img_path))
            
            frame_detections = []
            if frame_id in detections_by_frame:
                for det in detections_by_frame[frame_id]:
                    x1, y1, w, h, conf = det[2], det[3], det[4], det[5], det[6]
                    # In MOT20, confidence is binary (0 or 1), accept all valid detections
                    if conf < 0.0:  # Only skip negative confidence (invalid detections)
                        continue
                    
                    patch = frame_img[int(y1):int(y1 + h), int(x1):int(x1 + w)]
                    if patch.size == 0: continue
                    
                    embedding = None
                    try:
                        embedding = embedding_extractor.extract(patch)
                    except:
                        embedding = None
                    
                    # MOT20 has binary confidence (0 or 1), but 0 doesn't mean bad detection
                    # Set confidence to 1.0 for all valid detections since they passed quality filtering
                    detection_conf = 1.0 if conf >= 0.0 else 0.5
                    
                    frame_detections.append(
                        Detection(
                            position=np.array([x1 + w / 2, y1 + h / 2]), 
                            bbox=np.array([x1, y1, x1 + w, y1 + h]), 
                            confidence=detection_conf,
                            embedding=embedding
                        )
                    )
            
            tracked_objects = tracker.update(frame_detections)
            
            # Render and display
            output_frame = visualizer.draw_frame_opencv(frame_detections, tracked_objects, frame_num=frame_id, frame=frame_img)
            cv2.imshow(f"SwarmSort - {seq_name}", output_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                skip_sequence = True
                break
            elif key == 27:  # ESC key
                quit_all = True
                break
            
            # Save results
            for track in tracked_objects:
                x1, y1, x2, y2 = track.bbox
                results.append([frame_id, track.id, x1, y1, x2 - x1, y2 - y1, 1, -1, -1, -1])
        
        cv2.destroyAllWindows()
        
        # Save results for this sequence
        output_file = results_dir / f"{seq_name}.txt"
        if results:
            np.savetxt(str(output_file), np.array(results), fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d", delimiter=',')
            print(f"  - Results for {seq_name} saved to {output_file} ({len(results)} tracks)")
        else:
            with open(output_file, 'w') as f:
                pass
            print(f"  - No tracks found, empty file created: {output_file}")
        
        if quit_all:
            print("\nVisualization stopped by user.")
            break
        elif skip_sequence:
            print(f"  - Skipped to next sequence")

def create_submission_zip(results_dir: Path, zip_name: str = "mot20_submission.zip"):
    """Create a ZIP file ready for MOTChallenge submission."""
    import zipfile
    
    zip_path = results_dir / zip_name
    
    # Expected MOT20 test sequences
    expected_sequences = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-04", 
                         "MOT20-05", "MOT20-06", "MOT20-07", "MOT20-08"]
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for seq_name in expected_sequences:
            result_file = results_dir / f"{seq_name}.txt"
            if result_file.exists():
                zipf.write(result_file, f"{seq_name}.txt")
                print(f"  - Added {seq_name}.txt to ZIP")
            else:
                print(f"  - WARNING: {seq_name}.txt not found, creating empty file")
                # Create empty file in ZIP for missing sequences
                zipf.writestr(f"{seq_name}.txt", "")
    
    print(f"\nSubmission ZIP created: {zip_path}")
    print(f"Upload this file to: https://motchallenge.net/")
    return zip_path

def main():
    parser = argparse.ArgumentParser(description="Run SwarmSort on the MOT20 dataset.")
    parser.add_argument(
        "sequence", nargs="?", default=None,
        help="The name of the sequence to process (e.g., MOT20-04). If not provided, runs all sequences in benchmark mode."
    )
    parser.add_argument(
        "--visualize", action="store_true", default=True,
        help="Enable visualization mode. Can process single sequence or all sequences with real-time display."
    )
    parser.add_argument(
        "--create-zip", action="store_true", default=True,
        help="Create a submission-ready ZIP file after processing all sequences."
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
        if args.sequence:
            # Single sequence visualization
            run_visual_mode(args.sequence, data_root, results_dir)
        else:
            # All sequences visualization
            run_visual_mode_all_sequences(data_root, results_dir)
            
        # Create submission ZIP if requested and processed all sequences
        if args.create_zip and not args.sequence:
            print("\nCreating submission ZIP...")
            create_submission_zip(results_dir)
    else:
        run_benchmark_mode(data_root, results_dir, args.sequence)
        
        # Create submission ZIP if requested and not processing single sequence
        if args.create_zip and not args.sequence:
            print("\nCreating submission ZIP...")
            create_submission_zip(results_dir)

    print("\nProcessing complete.")

if __name__ == "__main__":
    main()