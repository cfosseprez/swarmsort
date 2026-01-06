Configuration Module
====================

.. module:: swarmsort.config
   :no-index:

The configuration module provides the SwarmSortConfig class and utilities for managing tracker parameters.

SwarmSortConfig
---------------

.. autoclass:: swarmsort.config.SwarmSortConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

   Main configuration class for SwarmSort tracker parameters.

   .. important::
      
      The ``max_distance`` parameter should be set **1.5-2x higher** than the expected maximum movement between frames, as the actual matching combines spatial distance, embedding similarity, and uncertainty penalties.

   **Key Parameter Relationships:**
   
   - ``local_density_radius`` defaults to ``max_distance``
   - ``greedy_threshold`` defaults to ``max_distance / 5``
   - ``reid_max_distance`` defaults to ``max_distance``

   .. automethod:: __init__
   .. automethod:: validate
   .. automethod:: to_dict
   .. automethod:: from_dict
   .. automethod:: from_yaml
   .. automethod:: save_yaml

Configuration Examples
----------------------

Basic Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from swarmsort import SwarmSortConfig

   # Simple configuration
   config = SwarmSortConfig(
       max_distance=150.0,
       do_embeddings=True,
       min_consecutive_detections=6
   )

Domain-Specific Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Microscopy tracking (many slow-moving objects)
   microscopy_config = SwarmSortConfig(
       max_distance=50.0,               # Small movements
       min_consecutive_detections=3,    # Quick initialization
       assignment_strategy='hybrid',    # Handle many objects efficiently
       uncertainty_weight=0.2           # Low uncertainty for predictable motion
   )

   # Surveillance (people tracking with occlusions)
   surveillance_config = SwarmSortConfig(
       max_distance=150.0,
       do_embeddings=True,              # Use appearance
       embedding_weight=1.5,            # Trust appearance more
       reid_enabled=True,               # Re-identify after occlusions
       max_track_age=60                 # Keep tracks longer (2s at 30fps)
   )

   # Vehicle tracking (fast, predictable motion)
   vehicle_config = SwarmSortConfig(
       max_distance=200.0,              # Fast movement
       kalman_type='oc',                # Better acceleration model
       assignment_strategy='hungarian', # Optimal assignment
       detection_conf_threshold=0.5     # Only high-quality detections
   )

Loading from YAML
~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # config.yaml
   max_distance: 150.0
   do_embeddings: true
   embedding_weight: 1.0
   uncertainty_weight: 0.33
   assignment_strategy: hybrid
   min_consecutive_detections: 4

.. code-block:: python

   # Load configuration
   config = SwarmSortConfig.from_yaml('config.yaml')
   
   # Modify and save
   config.max_distance = 200.0
   config.save_yaml('modified_config.yaml')

BaseConfig
----------

.. autoclass:: swarmsort.config.BaseConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

   Base configuration class with YAML loading capabilities.

Configuration Utilities
-----------------------

.. autofunction:: swarmsort.config.merge_config_with_priority
   :no-index:

   Merges configuration from multiple sources with priority:
   runtime_config > yaml_config > default_config

.. autofunction:: swarmsort.config.load_config
   :no-index:

   Convenience function to load SwarmSort configuration.

Parameter Validation
--------------------

The configuration system includes automatic validation:

.. code-block:: python

   try:
       config = SwarmSortConfig(
           max_distance=-10  # Invalid!
       )
   except ValueError as e:
       print(f"Configuration error: {e}")

   # All parameters are validated
   config = SwarmSortConfig()
   config.validate()  # Checks all constraints