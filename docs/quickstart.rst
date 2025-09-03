Quick Start Guide
=================

This guide will get you tracking objects in under 5 minutes.

Basic Example
-------------

The simplest way to use SwarmSort:

.. code-block:: python

   from swarmsort import SwarmSortTracker, Detection
   import numpy as np

   # Create a tracker with default settings
   tracker = SwarmSortTracker()

   # Create detections for the current frame
   detections = [
       Detection(position=np.array([100, 200]), confidence=0.9),
       Detection(position=np.array([300, 400]), confidence=0.85),
   ]

   # Update tracker and get results
   tracked_objects = tracker.update(detections)

   # Use the results
   for obj in tracked_objects:
       print(f"Track {obj.id}: position {obj.position}")

With Video Processing
---------------------

Process a video file with object detection:

.. code-block:: python

   import cv2
   from swarmsort import SwarmSortTracker, Detection
   
   tracker = SwarmSortTracker()
   cap = cv2.VideoCapture('video.mp4')
   
   while cap.isOpened():
       ret, frame = cap.read()
       if not ret:
           break
       
       # Get detections from your detector (e.g., YOLO)
       detections = detect_objects(frame)  # Your detection function
       
       # Convert to SwarmSort format
       swarmsort_detections = []
       for det in detections:
           swarmsort_detections.append(Detection(
               position=det['center'],
               confidence=det['score'],
               bbox=det['bbox']
           ))
       
       # Track objects
       tracked_objects = tracker.update(swarmsort_detections)
       
       # Draw results
       for obj in tracked_objects:
           if obj.bbox is not None:
               x1, y1, x2, y2 = obj.bbox.astype(int)
               cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
               cv2.putText(frame, f"ID: {obj.id}", (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
       
       cv2.imshow('Tracking', frame)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

Common Configurations
---------------------

**For Crowded Scenes**

.. code-block:: python

   from swarmsort import SwarmSortConfig, SwarmSortTracker
   
   config = SwarmSortConfig(
       max_distance=100.0,                 # Shorter association distance
       uncertainty_weight=0.5,             # More conservative
       collision_freeze_embeddings=True,   # Prevent ID switches
       min_consecutive_detections=3        # Faster initialization
   )
   tracker = SwarmSortTracker(config)

**For Fast-Moving Objects**

.. code-block:: python

   config = SwarmSortConfig(
       max_distance=200.0,                 # Longer association distance
       kalman_type='oc',                   # Better motion model
       assignment_strategy='hungarian',    # Optimal assignment
       max_track_age=15                    # Quick cleanup
   )
   tracker = SwarmSortTracker(config)

**For High Accuracy**

.. code-block:: python

   config = SwarmSortConfig(
       do_embeddings=True,                 # Use visual features
       embedding_weight=1.5,               # Trust appearance
       reid_enabled=True,                  # Re-identify lost tracks
       min_consecutive_detections=8        # Careful initialization
   )
   tracker = SwarmSortTracker(config)

Next Steps
----------

- Learn about :doc:`configuration` options
- Explore :doc:`advanced_features`
- See :doc:`examples/visualization` for drawing tracks
- Check :doc:`performance_tuning` for optimization