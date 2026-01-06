Core Tracker Module
====================

.. module:: swarmsort.core
   :no-index:

The core tracking module contains the main SwarmSortTracker class and associated tracking logic.

SwarmSortTracker
----------------

.. autoclass:: swarmsort.core.SwarmSortTracker
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

   .. automethod:: __init__
   .. automethod:: update
   .. automethod:: reset
   .. automethod:: get_statistics
   .. automethod:: get_recently_lost_tracks
   .. automethod:: get_all_active_tracks

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

   from swarmsort import SwarmSortTracker, SwarmSortConfig, Detection
   import numpy as np

   # Configure tracker
   config = SwarmSortConfig(
       max_distance=150.0,
       uncertainty_weight=0.33,
       do_embeddings=True
   )
   tracker = SwarmSortTracker(config)

   # Process detections
   detections = [
       Detection(position=np.array([100, 200]), confidence=0.9)
   ]
   tracked_objects = tracker.update(detections)

   # Get different track states
   alive = tracker.update(detections)  # Currently visible
   recently_lost = tracker.get_recently_lost_tracks(max_frames_lost=5)
   all_active = tracker.get_all_active_tracks()

Track State Management
----------------------

FastTrackState
~~~~~~~~~~~~~~

.. autoclass:: swarmsort.core.FastTrackState
   :members:
   :undoc-members:
   :show-inheritance:

   Internal representation of a track with Kalman filtering and embedding history.

PendingDetection
~~~~~~~~~~~~~~~~

.. autoclass:: swarmsort.data_classes.PendingDetection
   :members:
   :undoc-members:
   :show-inheritance:

   Represents a detection that hasn't been confirmed as a track yet.

Cost Computation Functions
--------------------------

The core module includes several Numba-compiled functions for efficient cost computation:

.. autofunction:: swarmsort.core.compute_track_uncertainties
   :no-index:
.. autofunction:: swarmsort.core.compute_local_density
   :no-index:
.. autofunction:: swarmsort.core.compute_cost_matrix_vectorized
   :no-index:

Assignment Strategies
---------------------

.. autofunction:: swarmsort.core.numba_greedy_assignment
   :no-index:
.. autofunction:: swarmsort.core.hungarian_assignment_wrapper
   :no-index:

These functions implement the different assignment strategies available in SwarmSort:

- **Greedy**: Fast assignment for obvious matches
- **Hungarian**: Optimal assignment using the Hungarian algorithm
- **Hybrid**: Combination of greedy and Hungarian for best performance