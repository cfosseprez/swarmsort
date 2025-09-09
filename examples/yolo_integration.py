"""
Example of using SwarmSort with YOLO v8/v11 object detection.

This example shows how to use the new prepare_input utilities for
seamless integration with YOLO models.
"""

import cv2
import numpy as np
from pathlib import Path

# SwarmSort imports
from swarmsort import (
    SwarmSortTracker,
    SwarmSortConfig,
    yolo_to_detections,
    prepare_detections,
    verify_detections
)


def track_with_yolo_v8():
    """Example using YOLO v8 with SwarmSort."""
    
    # Try importing YOLO (optional dependency)
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Please install ultralytics: pip install ultralytics")
        return
    
    # Initialize YOLO model
    model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt, etc.
    
    # Initialize SwarmSort tracker
    config = SwarmSortConfig(
        max_distance=150,
        do_embeddings=False,  # YOLO doesn't provide embeddings by default
        uncertainty_weight=0.3,
        assignment_strategy='hybrid'
    )
    tracker = SwarmSortTracker(config)
    
    # Process video
    video_path = "path/to/your/video.mp4"  # Change this
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run YOLO detection
        results = model.predict(frame, conf=0.5, verbose=False)
        
        # Convert YOLO results to SwarmSort format
        detections = yolo_to_detections(
            results[0],
            confidence_threshold=0.5,
            class_filter=[0]  # Only track person class (class 0 in COCO)
        )
        
        # Optional: Verify detections
        verified_detections, warnings = verify_detections(
            detections,
            image_shape=frame.shape[:2],
            auto_fix=True
        )
        
        if warnings:
            print(f"Frame {frame_count}: {len(warnings)} detection issues (auto-fixed)")
        
        # Update tracker
        tracked_objects = tracker.update(verified_detections)
        
        # Draw results
        for obj in tracked_objects:
            if obj.bbox is not None:
                x1, y1, x2, y2 = obj.bbox.astype(int)
                
                # Draw bounding box
                color = (0, 255, 0) if obj.confirmed else (255, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw ID and info
                label = f"ID:{obj.id}"
                if obj.confidence > 0:
                    label += f" ({obj.confidence:.2f})"
                cv2.putText(frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display
        cv2.imshow('YOLO + SwarmSort Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Processed {frame_count} frames")
    print(f"Final track count: {len(tracker.get_active_tracks())}")


def track_with_yolo_batch():
    """Example of batch processing with YOLO."""
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Please install ultralytics: pip install ultralytics")
        return
    
    # Initialize
    model = YOLO('yolov8n.pt')
    tracker = SwarmSortTracker(SwarmSortConfig())
    
    # Process entire video at once (for analysis)
    video_path = "path/to/video.mp4"
    results = model.predict(video_path, stream=True, conf=0.5)
    
    all_tracks = []
    for frame_idx, result in enumerate(results):
        # Convert detections
        detections = yolo_to_detections(result, confidence_threshold=0.5)
        
        # Track
        tracked_objects = tracker.update(detections)
        all_tracks.append(tracked_objects)
        
        print(f"Frame {frame_idx}: {len(detections)} detections, {len(tracked_objects)} tracks")
    
    return all_tracks


def track_with_custom_detector():
    """Example using custom detector with numpy arrays."""
    
    from swarmsort import numpy_to_detections, prepare_detections
    
    # Initialize tracker
    tracker = SwarmSortTracker(SwarmSortConfig(
        do_embeddings=True,
        embedding_function='cupytexture'
    ))
    
    # Simulate custom detector output
    # In practice, this would come from your detector
    boxes = np.array([
        [100, 100, 200, 200],  # [x1, y1, x2, y2]
        [300, 150, 400, 250],
        [500, 200, 600, 300]
    ], dtype=np.float32)
    
    confidences = np.array([0.9, 0.85, 0.7], dtype=np.float32)
    
    # Optional: Add embeddings from your feature extractor
    embeddings = np.random.randn(3, 128).astype(np.float32)
    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Convert to Detection format
    detections = numpy_to_detections(
        boxes=boxes,
        confidences=confidences,
        embeddings=embeddings,
        format='xyxy'  # or 'xywh', 'cxcywh'
    )
    
    # Or use the universal prepare function
    detections = prepare_detections(
        boxes,  # Auto-detects numpy array
        source_format='numpy',
        confidences=confidences,
        embeddings=embeddings
    )
    
    # Update tracker
    tracked_objects = tracker.update(detections)
    
    print(f"Tracking {len(tracked_objects)} objects")
    for obj in tracked_objects:
        print(f"  Track {obj.id}: pos={obj.position}, conf={obj.confidence:.2f}")


def main():
    """Run examples based on available dependencies."""
    
    print("SwarmSort Input Preparation Examples")
    print("=" * 50)
    
    # Check if YOLO is available
    try:
        from ultralytics import YOLO
        print("\n1. Running YOLO v8 integration example...")
        # Uncomment to run with real video:
        # track_with_yolo_v8()
        print("   (Skipped - requires video file)")
    except ImportError:
        print("\n1. YOLO not installed. Install with: pip install ultralytics")
    
    print("\n2. Running custom detector example...")
    track_with_custom_detector()
    
    print("\n3. Testing input verification...")
    
    # Create some problematic detections
    from swarmsort import Detection
    
    bad_detections = [
        Detection(position=np.array([100, 100]), confidence=1.5),  # Bad confidence
        Detection(position=np.array([-10, 500]), confidence=0.9),   # Out of bounds
        Detection(position=np.array([np.nan, 100]), confidence=0.8), # NaN value
    ]
    
    # Verify and auto-fix
    fixed_detections, warnings = verify_detections(
        bad_detections,
        image_shape=(480, 640),
        auto_fix=True
    )
    
    print(f"   Found {len(warnings)} issues:")
    for warning in warnings:
        print(f"   - {warning}")
    print(f"   Fixed {len(fixed_detections)}/{len(bad_detections)} detections")


if __name__ == "__main__":
    main()