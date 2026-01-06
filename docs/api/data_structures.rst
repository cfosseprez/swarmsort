Data Structures Module
=======================

.. module:: swarmsort.data_classes
   :no-index:

This module contains the core data structures used throughout SwarmSort for representing detections, tracked objects, and internal state.

Input Data Classes
------------------

Detection
~~~~~~~~~

.. autoclass:: swarmsort.data_classes.Detection
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

   Represents a single object detection in a frame.

   **Required Fields:**
   
   - ``position``: numpy array [x, y] representing object center
   - ``confidence``: float between 0 and 1
   
   **Optional Fields:**
   
   - ``bbox``: numpy array [x1, y1, x2, y2] for bounding box
   - ``embedding``: numpy array of visual features
   - ``class_id``: integer class identifier
   - ``id``: string identifier from detector

   **Example Usage:**

   .. code-block:: python

      import numpy as np
      from swarmsort import Detection

      # Minimal detection
      det = Detection(
          position=np.array([100.0, 200.0]),
          confidence=0.9
      )

      # Full detection with all fields
      det = Detection(
          position=np.array([100.0, 200.0]),
          confidence=0.95,
          bbox=np.array([80, 180, 120, 220]),
          embedding=np.random.randn(128),
          class_id=0,  # 0 = person
          id="yolo_det_42"
      )

      # From YOLO output
      def yolo_to_detection(yolo_box):
          x1, y1, x2, y2 = yolo_box.xyxy[0].numpy()
          return Detection(
              position=np.array([(x1+x2)/2, (y1+y2)/2]),
              confidence=yolo_box.conf[0].item(),
              bbox=np.array([x1, y1, x2, y2]),
              class_id=int(yolo_box.cls[0])
          )

Output Data Classes
-------------------

TrackedObject
~~~~~~~~~~~~~

.. autoclass:: swarmsort.data_classes.TrackedObject
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

   Represents a tracked object returned by the tracker.

   **Attributes:**
   
   - ``id``: Unique track identifier (persists across frames)
   - ``position``: Current position [x, y]
   - ``velocity``: Current velocity [vx, vy]
   - ``confidence``: Track confidence score
   - ``age``: Number of frames since track creation
   - ``hits``: Number of successful detections
   - ``time_since_update``: Frames since last detection (0 = currently visible)
   - ``state``: Track state (1 = active, 2 = recently lost)
   - ``bbox``: Bounding box if available
   - ``class_id``: Object class if available
   - ``predicted_position``: Kalman filter predicted position

   **Example Usage:**

   .. code-block:: python

      # Get tracking results
      tracked_objects = tracker.update(detections)

      for obj in tracked_objects:
          # Identity
          print(f"Track ID: {obj.id}")
          
          # Position and motion
          print(f"Position: {obj.position}")
          print(f"Velocity: {obj.velocity}")
          
          # Track quality
          print(f"Confidence: {obj.confidence:.2%}")
          print(f"Hit rate: {obj.hits}/{obj.age}")
          
          # Track status
          if obj.time_since_update == 0:
              print("Currently visible")
          else:
              print(f"Lost for {obj.time_since_update} frames")
          
          # Bounding box
          if obj.bbox is not None:
              x1, y1, x2, y2 = obj.bbox
              width, height = x2 - x1, y2 - y1
              print(f"Size: {width}x{height}")

Internal Data Classes
---------------------

PendingDetection
~~~~~~~~~~~~~~~~

.. autoclass:: swarmsort.data_classes.PendingDetection
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

   Internal class for detections waiting to be confirmed as tracks.

   Tracks are created only after ``min_consecutive_detections`` are received within ``max_detection_gap`` frames.

FastTrackState
~~~~~~~~~~~~~~

.. autoclass:: swarmsort.core.FastTrackState
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

   Internal representation of a track with full state information.

   **Key Attributes:**
   
   - ``id``: Track identifier
   - ``position``: Current position
   - ``velocity``: Current velocity
   - ``kalman``: Kalman filter state
   - ``embedding_history``: Stored embeddings
   - ``observation_history``: Past positions
   - ``age``: Total frames alive
   - ``hits``: Successful matches
   - ``misses``: Consecutive missed frames

Type Conversion Utilities
-------------------------

.. code-block:: python

   # Convert between formats
   from swarmsort import Detection
   import numpy as np

   # From dictionary
   det_dict = {'position': [100, 200], 'confidence': 0.9}
   det = Detection(**det_dict)

   # To numpy arrays for batch processing
   detections = [det1, det2, det3]
   positions = np.array([d.position for d in detections])
   confidences = np.array([d.confidence for d in detections])

   # From batch predictions
   def batch_to_detections(boxes, scores, embeddings=None):
       detections = []
       for i, (box, score) in enumerate(zip(boxes, scores)):
           det = Detection(
               position=box[:2],  # Use center or compute from box
               confidence=score,
               bbox=box,
               embedding=embeddings[i] if embeddings else None
           )
           detections.append(det)
       return detections