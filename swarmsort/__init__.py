"""
SwarmSort Standalone - Multi-Object Tracker with GPU-Accelerated Embeddings

A standalone implementation of the SwarmSort algorithm for multi-object tracking
with optional GPU acceleration using CuPy for embedding extraction.

Features:
- Real-time multi-object tracking
- GPU-accelerated embedding extraction (optional)
- Advanced distance scaling with 11 different methods
- Kalman filtering for motion prediction
- Hungarian algorithm for optimal assignment
- Re-identification for lost tracks

Installation:
    # Basic installation (CPU only)
    pip install swarmsort
    
    # With GPU support
    pip install swarmsort[gpu]
    
    # Development installation
    pip install swarmsort[dev]

Basic Usage:
    import numpy as np
    from swarmsort import SwarmSortTracker, Detection
    
    # Initialize tracker
    tracker = SwarmSortTracker()
    
    # Create detections (x1, y1, x2, y2, confidence, class_id)
    detections = [
        Detection([100, 100, 200, 200, 0.9, 0]),
        Detection([300, 150, 400, 250, 0.8, 1])
    ]
    
    # Track objects
    tracked_objects = tracker.track(detections, frame_data)
    
    for obj in tracked_objects:
        print(f"Track ID: {obj.id}, Position: {obj.position}")

GPU Usage:
    # With GPU-accelerated embeddings
    from swarmsort import SwarmSortTracker
    from swarmsort.embeddings import is_gpu_available
    
    if is_gpu_available():
        tracker = SwarmSortTracker(
            embedding_type='cupytexture',  # or 'mega_cupytexture'
            use_gpu=True
        )
    else:
        tracker = SwarmSortTracker(embedding_type='histogram')
"""

# Core tracking functionality
from .core import SwarmSortTracker
from .data_classes import Detection, TrackedObject
from .config import SwarmSortConfig
from .embedding_scaler import EmbeddingDistanceScaler

# Embedding functionality
from .embeddings import (
    CupyTextureEmbedding,
    MegaCupyTextureEmbedding,
    get_embedding_extractor,
    list_available_embeddings,
    compute_embedding_distance,
    compute_embedding_distances_batch,
    is_gpu_available,
    CUPY_AVAILABLE
)

# Integration utilities
try:
    from .integration import (
        AdaptiveSwarmSortTracker,
        SwarmSort,
        StandaloneSwarmSort,
        create_tracker,
        is_within_swarmtracker
    )
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False

# Visualization and simulation tools (optional)
try:
    from .drawing_utils import (
        TrackingVisualizer,
        VisualizationConfig,
        ColorManager,
        quick_visualize
    )
    from .simulator import (
        ObjectMotionSimulator,
        SimulationConfig,
        SimulatedObject,
        MotionType,
        create_demo_scenario
    )
    from .benchmarking import (
        TrackingBenchmark,
        ScalabilityBenchmark,
        BenchmarkResult,
        FrameTimingResult,
        quick_benchmark,
        timing_context
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
    'SwarmSortTracker',
    'Detection', 
    'TrackedObject',
    'SwarmSortConfig',
    'EmbeddingDistanceScaler',
    
    # Embedding classes
    'CupyTextureEmbedding',
    'MegaCupyTextureEmbedding',
    'get_embedding_extractor',
    'list_available_embeddings',
    'compute_embedding_distance',
    'compute_embedding_distances_batch',
    'is_gpu_available',
    'CUPY_AVAILABLE',
    
    # Integration (if available)
    *(['AdaptiveSwarmSortTracker', 'SwarmSort', 'StandaloneSwarmSort', 
       'create_tracker', 'is_within_swarmtracker'] 
      if INTEGRATION_AVAILABLE else []),
    
    # Visualization and simulation (if available)
    *(['TrackingVisualizer', 'VisualizationConfig', 'ColorManager', 'quick_visualize',
       'ObjectMotionSimulator', 'SimulationConfig', 'SimulatedObject', 'MotionType', 'create_demo_scenario',
       'TrackingBenchmark', 'ScalabilityBenchmark', 'BenchmarkResult', 'FrameTimingResult', 
       'quick_benchmark', 'timing_context']
      if VISUALIZATION_AVAILABLE else []),
    
    # Constants
    'INTEGRATION_AVAILABLE',
    'VISUALIZATION_AVAILABLE',
    '__version__',
    '__author__',
    '__email__',
    '__license__'
]

# Package metadata
def get_package_info():
    """Get package information."""
    gpu_status = "Available" if is_gpu_available() else "Not Available"
    integration_status = "Available" if INTEGRATION_AVAILABLE else "Not Available"
    visualization_status = "Available" if VISUALIZATION_AVAILABLE else "Not Available"
    
    return {
        'version': __version__,
        'author': __author__,
        'license': __license__,
        'gpu_support': gpu_status,
        'swarmtracker_integration': integration_status,
        'visualization_tools': visualization_status,
        'available_embeddings': list_available_embeddings()
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

# Convenient aliases for common usage
Tracker = SwarmSortTracker  # Short alias
GPUTracker = lambda **kwargs: SwarmSortTracker(embedding_type='cupytexture', use_gpu=True, **kwargs)
CPUTracker = lambda **kwargs: SwarmSortTracker(embedding_type='histogram', use_gpu=False, **kwargs)