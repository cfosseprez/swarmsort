SwarmSort Documentation
=======================

.. image:: https://github.com/user-attachments/assets/206946bb-9553-405d-812f-4056afeecc28
   :alt: SwarmSort Demo
   :align: center

*High-performance multi-object tracking optimized for microscopy and scenarios with hundreds of objects*

SwarmSort is a standalone Python library for real-time multi-object tracking (MOT) that maintains consistent object identities across video frames. It implements state-of-the-art algorithms including uncertainty-based cost systems, smart embedding freezing for collision scenarios, and hybrid assignment strategies to achieve robust tracking in challenging conditions.

Quick Start
-----------

.. code-block:: python

   from swarmsort import SwarmSortTracker, Detection, SwarmSortConfig
   import numpy as np

   # Configure tracker for your use case
   config = SwarmSortConfig(
       max_distance=80.0,               # Maximum association distance
       uncertainty_weight=0.33,         # Enable uncertainty-aware tracking
       assignment_strategy='hybrid',    # Smart assignment strategy
       do_embeddings=True               # Use visual features
   )
   tracker = SwarmSortTracker(config)

   # Process detections from your detector
   detections = [
       Detection(position=np.array([100, 200]), confidence=0.9),
       Detection(position=np.array([300, 400]), confidence=0.85)
   ]

   # Get tracked objects with persistent IDs
   tracked_objects = tracker.update(detections)

   for obj in tracked_objects:
       print(f"Track {obj.id}: position {obj.position}")

Key Features
------------

**Robust Identity Management**
   Maintains consistent IDs through occlusions, crossings, and disappearances using motion prediction and visual re-identification.

**Uncertainty-Aware Association**
   Adaptive cost computation based on track age, local density, and detection reliability for robust tracking.

**Smart Collision Handling**
   Prevents ID switches in dense scenarios through intelligent embedding freezing and density-based adaptation.

**Real-Time Performance**
   30-120 FPS through Numba JIT compilation, vectorized operations, and optional GPU acceleration.

**Production Ready**
   200+ unit tests, cross-platform support, and comprehensive error handling.

How It Works
------------

SwarmSort uses a position-centric additive cost model for detection-track association:

.. code-block:: text

   cost = position_distance + embedding_weight × embedding_distance × max_distance

**Tracking Pipeline:**

1. **Motion Prediction** - Kalman filter predicts track positions
2. **Cost Computation** - Calculate association costs (spatial + appearance)
3. **Assignment** - Match detections to tracks (Hungarian/greedy/hybrid)
4. **Track Management** - Confirm new tracks, delete lost tracks
5. **Re-identification** - Recover lost tracks using visual features

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Reference

   configuration
   api_reference

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
