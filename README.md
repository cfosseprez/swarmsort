# SwarmSort Standalone

A high-performance standalone multi-object tracking library with GPU-accelerated embeddings. SwarmSort combines advanced computer vision techniques with deep learning embeddings for real-time tracking applications.

## Features

- **Real-time multi-object tracking** with optimized algorithms
- **GPU-accelerated embedding extraction** using CuPy (optional)
- **Advanced distance scaling** with 11 different normalization methods
- **Kalman filtering** for motion prediction and smoothing
- **Hungarian algorithm** for optimal detection-track assignment
- **Re-identification capabilities** for lost track recovery
- **Standalone operation** with optional SwarmTracker integration
- **Comprehensive test suite** with 200+ tests

## Installation

### Standalone Installation (via Poetry)

```bash
cd swarmsort_standalone
poetry install
```

### Development Installation

```bash
cd swarmsort_standalone
poetry install --with dev
```

## Quick Start

### Basic Usage

```python
import numpy as np
from swarmsort import SwarmSort, Detection

# Create tracker
tracker = SwarmSort()

# Create detections for current frame
detections = [
    Detection(position=[10.0, 20.0], confidence=0.9),
    Detection(position=[50.0, 60.0], confidence=0.8),
]

# Update tracker
tracked_objects = tracker.update(detections)

# Print results
for obj in tracked_objects:
    print(f"Track {obj.id}: position {obj.position}, confidence {obj.confidence}")
```

### With Embeddings

```python
from swarmsort import SwarmSort, SwarmSortConfig, Detection

# Configure tracker for embeddings
config = SwarmSortConfig(
    use_embeddings=True,
    embedding_weight=0.4,
    embedding_matching_method='best_match'
)
tracker = SwarmSort(config)

# Create detection with embedding
embedding = np.random.randn(128).astype(np.float32)
detection = Detection(
    position=[10.0, 20.0],
    confidence=0.9,
    embedding=embedding,
    bbox=[5.0, 15.0, 15.0, 25.0]  # [x1, y1, x2, y2]
)

tracked_objects = tracker.update([detection])
```

### Configuration Options

```python
from swarmsort import SwarmSortConfig

config = SwarmSortConfig(
    max_distance=80.0,                    # Maximum association distance
    high_score_threshold=0.8,             # High confidence detection threshold
    use_embeddings=True,                  # Enable embedding matching
    embedding_weight=0.3,                 # Weight of embeddings in cost function
    max_embeddings_per_track=15,          # Maximum embeddings stored per track
    embedding_matching_method='best_match', # 'best_match', 'average', 'weighted_average'
    
    # Track initialization
    min_consecutive_detections=3,         # Minimum detections to create track
    max_detection_gap=2,                  # Maximum gap between detections
    
    # Re-identification
    reid_enabled=True,                    # Enable re-identification
    reid_max_distance=150.0,              # Maximum distance for ReID
    reid_embedding_threshold=0.4,         # Embedding threshold for ReID
    
    # Performance tuning
    embedding_scaling_method='min_robustmax', # Embedding distance scaling method
    duplicate_detection_threshold=25.0,   # Distance threshold for duplicate removal
)

tracker = SwarmSort(config)
```

## Advanced Usage

### Factory Function

```python
from swarmsort import create_tracker

# Default tracker
tracker = create_tracker()

# With configuration dict
tracker = create_tracker({'max_distance': 100.0, 'use_embeddings': True})

# Force standalone mode (disable SwarmTracker integration)
tracker = create_tracker(force_standalone=True)

# From YAML config file
tracker = create_tracker('config.yaml')
```

### Integration Detection

```python
from swarmsort import is_within_swarmtracker

if is_within_swarmtracker():
    print("Running within SwarmTracker pipeline")
else:
    print("Running in standalone mode")
```

### Different Tracker Types

```python
from swarmsort import SwarmSort, StandaloneSwarmSort, AdaptiveSwarmSortTracker

# Adaptive tracker (automatically detects environment)
tracker1 = SwarmSort()

# Explicit standalone tracker
tracker2 = StandaloneSwarmSort()

# Adaptive tracker (same as SwarmSort)
tracker3 = AdaptiveSwarmSortTracker()
```

## Data Classes

### Detection

```python
from swarmsort import Detection
import numpy as np

detection = Detection(
    position=np.array([x, y]),           # Required: [x, y] position
    confidence=0.9,                      # Detection confidence [0, 1]
    bbox=np.array([x1, y1, x2, y2]),     # Optional: bounding box
    embedding=np.array(...),             # Optional: feature embedding
    class_id=0,                          # Optional: object class
    id="detection_123"                   # Optional: detection identifier
)
```

### TrackedObject

```python
# Returned by tracker.update()
for tracked_obj in tracked_objects:
    print(f"ID: {tracked_obj.id}")
    print(f"Position: {tracked_obj.position}")
    print(f"Velocity: {tracked_obj.velocity}")
    print(f"Confidence: {tracked_obj.confidence}")
    print(f"Age: {tracked_obj.age}")
    print(f"Hits: {tracked_obj.hits}")
    print(f"Time since update: {tracked_obj.time_since_update}")
    print(f"State: {tracked_obj.state}")
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_distance` | 80.0 | Maximum distance for detection-track association |
| `high_score_threshold` | 0.8 | Threshold for high-confidence detections |
| `max_age` | 20 | Maximum frames to keep track alive without detections |
| `detection_conf_threshold` | 0.3 | Minimum confidence for detections |
| `use_embeddings` | True | Whether to use embedding features |
| `embedding_weight` | 0.3 | Weight for embedding similarity in cost function |
| `max_embeddings_per_track` | 15 | Maximum embeddings stored per track |
| `embedding_matching_method` | 'weighted_average' | Method for multi-embedding matching |
| `min_consecutive_detections` | 3 | Minimum consecutive detections to create track |
| `max_detection_gap` | 2 | Maximum gap between detections for same pending track |
| `reid_enabled` | True | Enable re-identification of lost tracks |
| `reid_max_distance` | 150.0 | Maximum distance for ReID |
| `reid_embedding_threshold` | 0.4 | Embedding threshold for ReID |
| `duplicate_detection_threshold` | 25.0 | Distance threshold for duplicate removal |

## Performance Optimizations

SwarmSort includes several performance optimizations:

1. **Numba JIT Compilation**: Core mathematical functions are compiled with Numba for maximum speed
2. **Vectorized Operations**: Efficient numpy-based matrix operations
3. **Adaptive Embedding Scaling**: Dynamic scaling of embedding distances for numerical stability
4. **Optimized Memory Usage**: Efficient data structures and memory management
5. **Parallel Processing**: Multi-threaded operations where beneficial

## Integration with SwarmTracker

When used within the SwarmTracker pipeline, SwarmSort automatically:

- Detects the SwarmTracker environment
- Uses SwarmTracker's detection and tracking data classes
- Integrates with SwarmTracker's configuration system
- Leverages SwarmTracker's embedding and feature extraction

To use within SwarmTracker:

```python
# In your SwarmTracker configuration
tracker_config = {
    'tracker_type': 'swarmsort',
    'swarmsort': {
        'max_distance': 80.0,
        'use_embeddings': True,
        'embedding_weight': 0.3
    }
}
```

## Examples

See the `examples/` directory for comprehensive usage examples:

- `basic_usage.py`: Complete examples with visualization
- Advanced tracking scenarios
- Configuration examples
- Performance benchmarks

## Testing

```bash
# Run tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=swarmsort --cov-report=html

# Run specific test
poetry run pytest tests/test_basic.py::test_basic_tracking
```

## Development

```bash
# Install development dependencies
poetry install --with dev

# Run linting
poetry run black swarmsort/
poetry run flake8 swarmsort/

# Run type checking
poetry run mypy swarmsort/
```

## Benchmarking

```bash
# Run benchmarks
poetry run pytest tests/ --benchmark-only
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use SwarmSort in your research, please cite:

```bibtex
@software{swarmsort,
    title={SwarmSort: High-Performance Multi-Object Tracking with Deep Learning},
    author={Charles Fosseprez},
    year={2024},
    url={https://github.com/your-org/swarmsort}
}
```