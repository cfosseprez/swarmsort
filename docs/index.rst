SwarmSort Documentation
=======================

.. image:: _static/swarmsort_logo.png
   :alt: SwarmSort Logo
   :align: center

*A fast, flexible, and GPU-accelerated Python library for real-time multi-object tracking.*

SwarmSort provides advanced tracking algorithms using embeddings, Kalman filtering, and collision-free assignment. It is highly configurable and easy to use.

Quick Start
-----------

.. code-block:: python

   from swarmsort import SwarmSortTracker, SwarmSortConfig, Detection

   # Create configuration
   config = SwarmSortConfig(max_distance=120.0)
   tracker = SwarmSortTracker(config=config)

   # Feed detections frame by frame
   detections = [Detection(id=1, position=(10, 20))]
   tracker.update(detections)

Key Features
------------

- **GPU-accelerated embeddings** with CuPy support.
- **Flexible assignment strategies**: hybrid, greedy, Hungarian fallback.
- **Collision-free tracking** with optional embedding freeze.
- **Extensive configuration** through `SwarmSortConfig`.

API Reference
=============

.. toctree::
   :maxdepth: 2

   generated/swarmsort
