"""
SwarmSort Standalone - Multi-Object Tracker with GPU-Accelerated Embeddings

SwarmSort is a high-performance, standalone multi-object tracking library that combines
advanced computer vision techniques with deep learning embeddings for robust real-time
object tracking applications.

Key Features:
    - Real-time multi-object tracking with motion prediction via Kalman filtering
    - GPU-accelerated embedding extraction using CuPy (optional)
    - Advanced distance scaling with 11 different normalization methods  
    - Hungarian algorithm for optimal detection-to-track assignment
    - Re-identification (ReID) capabilities for recovering lost tracks
    - Probabilistic and non-probabilistic cost computation methods
    - Comprehensive configuration system with sensible defaults
    - Extensive test suite with 200+ tests for reliability

Installation:
    # Clone from repository
    git clone https://github.com/cfosseprez/swarmsort.git
    cd swarmsort
    poetry install
    
    # Development installation with testing tools
    poetry install --with dev

Basic Usage:
    import numpy as np
    from swarmsort import SwarmSortTracker, Detection
    
    # Initialize tracker with default settings
    tracker = SwarmSortTracker()
    
    # Create detections for current frame
    detections = [
        Detection(
            position=np.array([100.0, 150.0], dtype=np.float32),
            confidence=0.9
        ),
        Detection(
            position=np.array([300.0, 200.0], dtype=np.float32), 
            confidence=0.8
        )
    ]
    
    # Update tracker and get current tracked objects
    tracked_objects = tracker.update(detections)
    
    # Process results
    for obj in tracked_objects:
        print(f"Track ID: {obj.id}, Position: {obj.position}")
        print(f"  Velocity: {obj.velocity}, Age: {obj.age}")

Advanced Configuration:
    from swarmsort import SwarmSortTracker, SwarmSortConfig
    
    # Configure tracker for embedding-based tracking
    config = SwarmSortConfig(
        use_embeddings=True,
        embedding_weight=0.4,
        reid_enabled=True,
        max_distance=100.0
    )
    tracker = SwarmSortTracker(config)

GPU Acceleration:
    from swarmsort import is_gpu_available, SwarmSortTracker
    
    if is_gpu_available():
        print("GPU acceleration available for embeddings")
        # GPU will be used automatically for embedding operations
        tracker = SwarmSortTracker()
    else:
        print("Using CPU mode")
        tracker = SwarmSortTracker()
"""

# Core tracking functionality
from .core import SwarmSortTracker
from .data_classes import Detection, TrackedObject
from .config import SwarmSortConfig
from .embedding_scaler import EmbeddingDistanceScaler

# Embedding functionality
from .embeddings import (
    CupyTextureEmbedding,
    CupyTextureColorEmbedding,
    MegaCupyTextureEmbedding,
    get_embedding_extractor,
    list_available_embeddings,
    compute_embedding_distance,
    compute_embedding_distances_batch,
    is_gpu_available,
    CUPY_AVAILABLE,
)

# Integration utilities
try:
    from .integration import (
        AdaptiveSwarmSortTracker,
        SwarmSort,
        StandaloneSwarmSort,
        create_tracker,
        is_within_swarmtracker,
    )

    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False

# SwarmTracker Pipeline Integration
from .swarmtracker_adapter import (
    RawTrackerSwarmSORT,
    create_swarmsort_tracker,
    TrackingResult,
    create_tracked_object_fast,
    FastMultiHypothesisTracker,
)

# Visualization and simulation tools (optional)
try:
    from .drawing_utils import (
        TrackingVisualizer,
        VisualizationConfig,
        ColorManager,
        quick_visualize,
    )
    from .simulator import (
        ObjectMotionSimulator,
        SimulationConfig,
        SimulatedObject,
        MotionType,
        create_demo_scenario,
    )
    from .benchmarking import (
        TrackingBenchmark,
        ScalabilityBenchmark,
        BenchmarkResult,
        FrameTimingResult,
        quick_benchmark,
        timing_context,
    )

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

__version__ = "0.1.0"
__author__ = "Charles Fosseprez"
__email__ = "charles.fosseprez.pro@gmail.com"
__license__ = "MIT"

__all__ = [
    # Core classes
    "SwarmSortTracker",
    "Detection",
    "TrackedObject",
    "SwarmSortConfig",
    "EmbeddingDistanceScaler",
    # Embedding classes
    "CupyTextureEmbedding",
    "CupyTextureColorEmbedding",
    "MegaCupyTextureEmbedding",
    "get_embedding_extractor",
    "list_available_embeddings",
    "compute_embedding_distance",
    "compute_embedding_distances_batch",
    "is_gpu_available",
    "CUPY_AVAILABLE",
    # Integration (if available)
    *(
        [
            "AdaptiveSwarmSortTracker",
            "SwarmSort",
            "StandaloneSwarmSort",
            "create_tracker",
            "is_within_swarmtracker",
        ]
        if INTEGRATION_AVAILABLE
        else []
    ),
    # Visualization and simulation (if available)
    *(
        [
            "TrackingVisualizer",
            "VisualizationConfig",
            "ColorManager",
            "quick_visualize",
            "ObjectMotionSimulator",
            "SimulationConfig",
            "SimulatedObject",
            "MotionType",
            "create_demo_scenario",
            "TrackingBenchmark",
            "ScalabilityBenchmark",
            "BenchmarkResult",
            "FrameTimingResult",
            "quick_benchmark",
            "timing_context",
        ]
        if VISUALIZATION_AVAILABLE
        else []
    ),
    # SwarmTracker adapter exports
    "RawTrackerSwarmSORT",
    "create_swarmsort_tracker",
    "TrackingResult",
    "create_tracked_object_fast",
    "FastMultiHypothesisTracker",
    # Constants
    "INTEGRATION_AVAILABLE",
    "VISUALIZATION_AVAILABLE",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]


# Package metadata
def get_package_info():
    """Get package information."""
    gpu_status = "Available" if is_gpu_available() else "Not Available"
    integration_status = "Available" if INTEGRATION_AVAILABLE else "Not Available"
    visualization_status = "Available" if VISUALIZATION_AVAILABLE else "Not Available"

    return {
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "gpu_support": gpu_status,
        "swarmtracker_integration": integration_status,
        "visualization_tools": visualization_status,
        "available_embeddings": list_available_embeddings(),
    }


def print_package_info():
    """Print package information."""
    info = get_package_info()

    print("=" * 50)
    print("SwarmSort Standalone Package Information")
    print("=" * 50)
    print(f"Version: {info['version']}")
    print(f"Author: {info['author']}")
    print(f"License: {info['license']}")
    print(f"GPU Support: {info['gpu_support']}")
    print(f"SwarmTracker Integration: {info['swarmtracker_integration']}")
    print(f"Visualization Tools: {info['visualization_tools']}")
    print(f"Available Embeddings: {', '.join(info['available_embeddings'])}")
    print("=" * 50)

