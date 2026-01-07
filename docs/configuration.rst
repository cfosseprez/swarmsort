Configuration Guide
===================

SwarmSort is highly configurable to adapt to different tracking scenarios. This guide covers all configuration options and their relationships.

.. important::
   
   **Understanding max_distance**: The ``max_distance`` parameter is the foundation of SwarmSort's configuration. It should be set **higher than the maximum expected movement** between frames because:
   
   - The actual matching uses a **combination** of spatial distance, embedding similarity, and uncertainty
   - With embeddings enabled, the effective matching distance is reduced by visual similarity
   - Uncertainty penalties further modify the association costs
   - A good starting point is **1.5-2x the expected maximum pixel movement** between frames
   
   Example: If objects move up to 100 pixels between frames, set ``max_distance=150`` to account for matching uncertainties.

Configuration Overview Table
----------------------------

.. list-table:: SwarmSort Configuration Parameters
   :header-rows: 1
   :widths: 25 20 15 40
   
   * - Parameter
     - Default
     - Type
     - Description
   * - **Core Tracking**
     -
     -
     -
   * - max_distance
     - 80.0
     - float
     - Base distance for association (see note above)
   * - detection_conf_threshold
     - 0.0
     - float
     - Minimum confidence to process detections
   * - max_track_age
     - 30
     - int
     - Frames before track deletion (30 = 1s at 30fps)
   * - deduplication_distance
     - 10.0
     - float
     - Min distance between detections (NMS-style)
   * - **Motion Modeling**
     - 
     - 
     - 
   * - kalman_type
     - 'simple'
     - str
     - 'simple' or 'oc' (OC-SORT style)
   * - **Uncertainty System**
     - 
     - 
     - 
   * - uncertainty_weight
     - 0.33
     - float
     - Weight for uncertainty penalties (0=disabled)
   * - local_density_radius
     - max_distance
     - float
     - Radius for density computation (defaults to max_distance)
   * - **Embeddings**
     - 
     - 
     - 
   * - do_embeddings
     - True
     - bool
     - Enable visual feature matching
   * - embedding_weight
     - 1.0
     - float
     - Weight of appearance vs motion
   * - max_embeddings_per_track
     - 15
     - int
     - Embedding history size per track
   * - embedding_matching_method
     - 'weighted_average'
     - str
     - 'average', 'weighted_average', 'best_match', or 'median'
   * - embedding_function
     - 'cupytexture'
     - str
     - Feature extractor: 'cupytexture', 'cupytexture_color', 'cupytexture_mega'
   * - embedding_threshold_adjustment
     - 1.0
     - float
     - Gating threshold expansion factor
   * - store_embedding_scores
     - False
     - bool
     - Store match scores in TrackedObject
   * - embedding_score_history_length
     - 5
     - int
     - Number of recent scores to keep per track
   * - **Embedding Scaling**
     -
     -
     -
   * - embedding_scaling_method
     - 'min_robustmax'
     - str
     - Distance normalization method
   * - embedding_scaling_update_rate
     - 0.05
     - float
     - Learning rate for scaling statistics
   * - embedding_scaling_min_samples
     - 200
     - int
     - Samples before scaling activation
   * - **Collision Handling**
     - 
     - 
     - 
   * - collision_freeze_embeddings
     - True
     - bool
     - Freeze embeddings when tracks are close
   * - embedding_freeze_density
     - 1
     - int
     - Number of nearby tracks to trigger freeze
   * - collision_safety_distance
     - 30.0
     - float
     - Distance threshold for collision detection
   * - **Assignment Strategy**
     - 
     - 
     - 
   * - assignment_strategy
     - 'hybrid'
     - str
     - 'hungarian', 'greedy', or 'hybrid'
   * - greedy_threshold
     - max_distance/5
     - float
     - Distance for greedy matching (default: 30.0 when max_distance=150)
   * - hungarian_fallback_threshold
     - 1.0
     - float
     - Multiplier of max_distance for Hungarian fallback
   * - sparse_computation_threshold
     - 300
     - int
     - Object count threshold for sparse mode
   * - use_probabilistic_costs
     - False
     - bool
     - Use Mahalanobis distance for costs
   * - greedy_confidence_boost
     - 1.0
     - float
     - Confidence boost for greedy matches
   * - **Track Initialization**
     - 
     - 
     - 
   * - min_consecutive_detections
     - 6
     - int
     - Detections needed to confirm track
   * - max_detection_gap
     - 2
     - int
     - Max frame gap during initialization
   * - pending_detection_distance
     - 80.0
     - float
     - Distance threshold for pending detections
   * - init_conf_threshold
     - 0.0
     - float
     - Min confidence to start tracking
   * - **Re-identification**
     - 
     - 
     - 
   * - reid_enabled
     - True
     - bool
     - Enable track re-identification
   * - reid_max_distance
     - 150.0
     - float
     - Maximum distance for ReID
   * - reid_embedding_threshold
     - 0.3
     - float
     - Embedding similarity for ReID
   * - reid_min_frames_lost
     - 2
     - int
     - Frames lost before ReID attempt
   * - **Probabilistic Mode**
     -
     -
     -
   * - base_position_variance
     - 15.0
     - float
     - Base uncertainty in pixels²
   * - velocity_variance_scale
     - 2.0
     - float
     - Velocity contribution to uncertainty
   * - velocity_isotropic_threshold
     - 0.1
     - float
     - Threshold for circular covariance
   * - time_covariance_inflation
     - 0.1
     - float
     - Uncertainty growth per missed frame
   * - mahalanobis_normalization
     - 5.0
     - float
     - Scale factor for Mahalanobis distance
   * - probabilistic_gating_multiplier
     - 1.5
     - float
     - Pre-filter threshold expansion
   * - singular_covariance_threshold
     - 1e-6
     - float
     - Fallback to Euclidean threshold
   * - **Debug Settings**
     -
     -
     -
   * - debug_embeddings
     - False
     - bool
     - Print embedding debug info
   * - plot_embeddings
     - False
     - bool
     - Generate embedding plots (requires matplotlib)
   * - debug_timings
     - True
     - bool
     - Print per-frame timing info

Parameter Relationships
-----------------------

Several parameters are **dynamically linked** to ``max_distance`` by default:

.. code-block:: python

   # Default relationships when not explicitly set:
   local_density_radius = max_distance          # Same as max_distance
   greedy_threshold = max_distance / 5          # 20% of max_distance
   hungarian_fallback_threshold = 1.0           # 1x max_distance
   reid_max_distance = max_distance            # Same as max_distance

This means changing ``max_distance`` automatically scales related parameters:

.. code-block:: python

   # Example: Adjusting for faster movement
   config = SwarmSortConfig(max_distance=200)
   # Automatically sets:
   # - local_density_radius = 200
   # - greedy_threshold = 40
   # - reid_max_distance = 200

Core Parameters Detailed
------------------------

Motion Modeling
---------------

**kalman_type** : {'simple', 'oc'}, default='simple'
   Type of Kalman filter for motion prediction.
   
   - 'simple': Constant velocity model (faster)
   - 'oc': OC-SORT style with acceleration (better for erratic motion)

Uncertainty System
------------------

**uncertainty_weight** : float, default=0.33
   Weight for uncertainty penalties in cost computation.
   
   - 0.0: Disable uncertainty system (aggressive matching)
   - 0.5: Balanced approach
   - 1.0: Very conservative matching

**local_density_radius** : float, default=max_distance
   Radius for computing local track density.

Embedding Features
------------------

**do_embeddings** : bool, default=True
   Whether to use visual embeddings for matching.

**embedding_weight** : float, default=1.0
   Weight of embedding similarity in the cost function.
   
   - 0: Only use motion
   - 1: Equal weight to motion and appearance
   - >1: Trust appearance more than motion

**max_embeddings_per_track** : int, default=15
   Maximum number of embeddings stored per track.

**embedding_matching_method** : {'average', 'weighted_average', 'best_match', 'median'}, default='weighted_average'
   Method for matching multiple embeddings.

   - 'average': Mean distance to all stored embeddings
   - 'weighted_average': Recent embeddings weighted more (recommended)
   - 'best_match': Minimum distance (most similar embedding wins)
   - 'median': Median distance (robust to outliers)

**embedding_function** : str, default='cupytexture'
   Feature extractor algorithm for visual embeddings.

   - 'cupytexture': Fast texture-based features (GPU-accelerated)
   - 'cupytexture_color': Includes color information
   - 'cupytexture_mega': Most detailed features (slower)

**embedding_threshold_adjustment** : float, default=1.0
   Expands the gating threshold for embedding-based matching.
   Effective max distance = max_distance × (1 + embedding_weight × embedding_threshold_adjustment).

**store_embedding_scores** : bool, default=False
   When True, stores embedding match scores in TrackedObject for visualization/debugging.

Embedding Scaling
-----------------

**embedding_scaling_method** : str, default='min_robustmax'
   Method for normalizing embedding distances to [0,1] range.

**embedding_scaling_update_rate** : float, default=0.05
   Learning rate for updating scaling statistics (0.0 to 1.0).

**embedding_scaling_min_samples** : int, default=200
   Minimum samples before embedding scaling is activated.

Collision Handling
------------------

**collision_freeze_embeddings** : bool, default=True
   Freeze embedding updates when tracks are close together.

**embedding_freeze_density** : int, default=1
   Number of nearby tracks to trigger embedding freeze.

**collision_safety_distance** : float, default=30.0
   Distance threshold below which objects are considered in collision.

Assignment Strategy
-------------------

**assignment_strategy** : {'hungarian', 'greedy', 'hybrid'}, default='hybrid'
   Algorithm for detection-track assignment.
   
   - 'hungarian': Optimal but slower
   - 'greedy': Fast but suboptimal
   - 'hybrid': Best of both worlds

**greedy_threshold** : float, default=30.0
   Distance threshold for greedy assignment in hybrid mode.

**sparse_computation_threshold** : int, default=300
   Object count threshold for switching to sparse computation mode.

**use_probabilistic_costs** : bool, default=False
   When True, use Mahalanobis distance instead of Euclidean for cost computation.

**greedy_confidence_boost** : float, default=1.0
   Confidence multiplier for greedy matches in hybrid mode.

Track Initialization
--------------------

**min_consecutive_detections** : int, default=6
   Minimum detections required to confirm a track.
   
   - Lower: Faster response, more false positives
   - Higher: Slower response, fewer false positives

**max_detection_gap** : int, default=2
   Maximum frame gap allowed during track initialization.

**pending_detection_distance** : float, default=80.0
   Distance threshold for matching pending detections.

**init_conf_threshold** : float, default=0.0
   Minimum confidence score to start tracking an object (0.0 to 1.0).

Re-identification
-----------------

**reid_enabled** : bool, default=True
   Enable re-identification of lost tracks.

**reid_max_distance** : float, default=150.0
   Maximum distance for re-identification.

**reid_embedding_threshold** : float, default=0.3
   Embedding similarity threshold for re-identification.

**reid_min_frames_lost** : int, default=2
   Minimum frames a track must be lost before attempting ReID.

Probabilistic Mode
------------------

Enable probabilistic mode with ``use_probabilistic_costs=True`` for Mahalanobis distance-based matching.

**base_position_variance** : float, default=15.0
   Base uncertainty in position (pixels²).

**velocity_variance_scale** : float, default=2.0
   Velocity contribution to uncertainty computation.

**velocity_isotropic_threshold** : float, default=0.1
   Speed threshold below which covariance becomes circular (isotropic).

**time_covariance_inflation** : float, default=0.1
   Uncertainty growth rate per missed frame.

**mahalanobis_normalization** : float, default=5.0
   Scale factor for Mahalanobis distance normalization.

**probabilistic_gating_multiplier** : float, default=1.5
   Pre-filter threshold expansion for probabilistic mode.

**singular_covariance_threshold** : float, default=1e-6
   Threshold for detecting singular covariance matrices; falls back to Euclidean.

Debug Settings
--------------

**debug_embeddings** : bool, default=False
   Print detailed embedding information for debugging.

**plot_embeddings** : bool, default=False
   Generate embedding visualization plots (requires matplotlib).

**debug_timings** : bool, default=True
   Print per-frame timing information for performance analysis.

Configuration Examples
----------------------

**Microscopy Tracking**

.. code-block:: python

   config = SwarmSortConfig(
       max_distance=50.0,                  # Small movements
       kalman_type='simple',               # Predictable motion
       min_consecutive_detections=3,       # Quick initialization
       assignment_strategy='hybrid',       # Handle many objects
       uncertainty_weight=0.2              # Low uncertainty
   )

**Surveillance Camera**

.. code-block:: python

   config = SwarmSortConfig(
       do_embeddings=True,
       embedding_weight=1.5,               # Appearance important
       reid_enabled=True,                  # Re-identify people
       reid_max_distance=200.0,
       max_track_age=60                    # Keep tracks longer
   )

**Sports Analytics**

.. code-block:: python

   config = SwarmSortConfig(
       max_distance=150.0,
       kalman_type='oc',                   # Handle acceleration
       detection_conf_threshold=0.5,       # Only clear detections
       assignment_strategy='hungarian',    # Accurate assignment
       min_consecutive_detections=2        # Fast player detection
   )

Loading from YAML
-----------------

You can also load configuration from a YAML file:

.. code-block:: yaml

   # swarmsort_config.yaml
   max_distance: 150.0
   detection_conf_threshold: 0.3
   do_embeddings: true
   embedding_weight: 1.0
   assignment_strategy: hybrid
   min_consecutive_detections: 4

.. code-block:: python

   from swarmsort import SwarmSortConfig, SwarmSortTracker
   
   config = SwarmSortConfig.from_yaml('swarmsort_config.yaml')
   tracker = SwarmSortTracker(config)