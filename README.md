[![Documentation Status](https://readthedocs.org/projects/swarmsort/badge/?version=latest)](https://swarmsort.readthedocs.io/en/latest/)
[![PyPI Version](https://img.shields.io/pypi/v/swarmsort.svg)](https://pypi.org/project/swarmsort/)
[![Python Version](https://img.shields.io/pypi/pyversions/swarmsort.svg)](https://pypi.org/project/swarmsort/)
[![CI Tests](https://github.com/cfosseprez/swarmsort/actions/workflows/ci.yml/badge.svg)](https://github.com/cfosseprez/swarmsort/actions/workflows/ci.yml)
[![GPL-2.0 License](https://img.shields.io/badge/License-GPL%202.0-blue.svg)](https://github.com/cfosseprez/swarmsort/blob/main/LICENSE)


![logo](https://raw.githubusercontent.com/cfosseprez/swarmsort/releases/download/0.0.0/logo-swarmsort-horizontal.jpg)

# SwarmSort

A high-performance standalone multi-object tracking library with GPU-accelerated embeddings. SwarmSort combines advanced computer vision techniques with uncertainty-based cost systems and intelligent collision handling for robust real-time tracking applications.

## Documentation

**[Full Documentation](https://swarmsort.readthedocs.io/en/latest/)**

## Features

- **Real-time multi-object tracking** with optimized algorithms
- **GPU-accelerated embedding extraction** using CuPy (optional)
- **Uncertainty-based cost system** for intelligent track association
- **Smart collision handling** with density-based embedding freezing
- **Hybrid assignment strategy** combining greedy and Hungarian algorithms
- **Advanced Kalman filtering** with simple and OC-SORT style options
- **Re-identification capabilities** for lost track recovery
- **Track lifecycle management** with alive/recently lost track separation
- **Comprehensive test suite** with 200+ tests

## Installation

### Standalone Installation (via Poetry)

```bash
git clone https://github.com/cfosseprez/swarmsort.git
cd swarmsort
poetry install
```

### Development Installation

```bash
git clone https://github.com/cfosseprez/swarmsort.git
cd swarmsort
poetry install --with dev
```

## Quick Start

### Basic Usage

```python
import numpy as np
from swarmsort import SwarmSortTracker, Detection

# Create tracker
tracker = SwarmSortTracker()

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
from swarmsort import SwarmSortTracker, SwarmSortConfig, Detection

# Configure tracker for embeddings
config = SwarmSortConfig(
    do_embeddings=True,  # Updated parameter name
    embedding_weight=1.0,
    embedding_matching_method='weighted_average'
)
tracker = SwarmSortTracker(config)

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
    # Core tracking parameters
    max_distance=150.0,                   # Maximum association distance
    detection_conf_threshold=0.0,         # Minimum confidence for detections
    max_track_age=30,                     # Maximum frames before track deletion
    
    # Kalman filter type
    kalman_type='simple',                 # 'simple' or 'oc' (OC-SORT style)
    
    # Uncertainty-based cost system
    uncertainty_weight=0.33,              # Weight for uncertainty penalties
    local_density_radius=150.0,           # Radius for computing local track density
    
    # Embedding parameters
    do_embeddings=True,                   # Enable embedding matching
    embedding_weight=1.0,                 # Weight of embeddings in cost function
    max_embeddings_per_track=15,          # Maximum embeddings stored per track
    embedding_matching_method='weighted_average', # 'best_match', 'average', 'weighted_average'
    
    # Smart collision handling
    collision_freeze_embeddings=True,     # Freeze embeddings in dense areas
    embedding_freeze_density=1,           # Freeze when ≥N tracks within radius
    
    # Assignment strategy
    assignment_strategy='hybrid',         # 'hungarian', 'greedy', or 'hybrid'
    greedy_threshold=30.0,                # Distance threshold for greedy assignment
    
    # Track initialization
    min_consecutive_detections=6,         # Minimum detections to create track
    max_detection_gap=2,                  # Maximum gap between detections
    pending_detection_distance=80.0,      # Distance threshold for pending detection matching
    
    # Re-identification
    reid_enabled=True,                    # Enable re-identification
    reid_max_distance=150.0,              # Maximum distance for ReID
    reid_embedding_threshold=0.3,         # Embedding threshold for ReID
)

tracker = SwarmSortTracker(config)
```

## Advanced Usage

### Different Configuration Methods

```python
from swarmsort import SwarmSortTracker, SwarmSortConfig

# Default tracker
tracker = SwarmSortTracker()

# With configuration object
config = SwarmSortConfig(max_distance=100.0, do_embeddings=True)
tracker = SwarmSortTracker(config)

# With dictionary config
tracker = SwarmSortTracker({'max_distance': 100.0, 'do_embeddings': True})
```

### Basic Standalone Usage

```python
from swarmsort import SwarmSortTracker, SwarmSortConfig

# SwarmSort is a standalone tracker - no special integration needed
tracker = SwarmSortTracker()

# Configure for specific use cases
config = SwarmSortConfig(
    do_embeddings=True,
    reid_enabled=True,
    max_distance=100.0,
    assignment_strategy='hybrid',  # Use hybrid assignment strategy
    uncertainty_weight=0.33         # Enable uncertainty-based costs
)
tracker_configured = SwarmSortTracker(config)
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
    print(f"Bbox: {tracked_obj.bbox}")

# Track lifecycle management
alive_tracks = tracker.update(detections)  # Only currently detected tracks
recently_lost = tracker.get_recently_lost_tracks(max_frames_lost=5)  # Recently lost tracks
all_active = tracker.get_all_active_tracks()  # Both alive and recently lost
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Core Tracking** | | |
| `max_distance` | 150.0 | Maximum distance for detection-track association |
| `detection_conf_threshold` | 0.0 | Minimum confidence for detections |
| `max_track_age` | 30 | Maximum frames to keep track alive without detections |
| **Kalman Filtering** | | |
| `kalman_type` | 'simple' | Kalman filter type: 'simple' or 'oc' (OC-SORT style) |
| **Uncertainty System** | | |
| `uncertainty_weight` | 0.33 | Weight for uncertainty penalties (0 = disabled) |
| `local_density_radius` | max_distance | Radius for computing local track density (defaults to max_distance) |
| **Embeddings** | | |
| `do_embeddings` | True | Whether to use embedding features |
| `embedding_weight` | 1.0 | Weight for embedding similarity in cost function |
| `max_embeddings_per_track` | 15 | Maximum embeddings stored per track |
| `embedding_matching_method` | 'weighted_average' | Method for multi-embedding matching |
| **Collision Handling** | | |
| `collision_freeze_embeddings` | True | Freeze embedding updates in dense areas |
| `embedding_freeze_density` | 1 | Freeze when ≥N tracks within radius |
| **Assignment Strategy** | | |
| `assignment_strategy` | 'hybrid' | Assignment method: 'hungarian', 'greedy', or 'hybrid' |
| `greedy_threshold` | 30.0 | Distance threshold for greedy assignment (default: max_distance/5) |
| **Track Initialization** | | |
| `min_consecutive_detections` | 6 | Minimum consecutive detections to create track |
| `max_detection_gap` | 2 | Maximum gap between detections |
| `pending_detection_distance` | 80.0 | Distance threshold for pending detection matching |
| **Re-identification** | | |
| `reid_enabled` | True | Enable re-identification of lost tracks |
| `reid_max_distance` | 150.0 | Maximum distance for ReID |
| `reid_embedding_threshold` | 0.3 | Embedding threshold for ReID |

## Performance Optimizations

SwarmSort includes several performance optimizations:

1. **Numba JIT Compilation**: Core mathematical functions are compiled with Numba for maximum speed
2. **Vectorized Operations**: Efficient numpy-based matrix operations replacing O(n²) loops
3. **Hybrid Assignment Strategy**: Combines fast greedy assignment with Hungarian algorithm fallback
4. **Uncertainty-Based Costs**: Intelligent track association using track age, density, and reliability
5. **Smart Embedding Freezing**: Reduces computational overhead in crowded scenarios
6. **Adaptive Embedding Scaling**: Dynamic scaling of embedding distances for numerical stability
7. **Optimized Memory Usage**: Efficient data structures and memory management

## GPU Acceleration

SwarmSort supports GPU acceleration for embedding extraction using CuPy:

```python
from swarmsort import is_gpu_available, SwarmSortTracker, SwarmSortConfig

if is_gpu_available():
    print("GPU acceleration available")
    # GPU will be used automatically for embedding operations
    config = SwarmSortConfig(do_embeddings=True)
    tracker = SwarmSortTracker(config)
else:
    print("Using CPU mode")
    tracker = SwarmSortTracker()
```

## Visualization Example

Here's a complete example showing how to visualize tracking results:

```python
import cv2
import numpy as np
from swarmsort import SwarmSortTracker, Detection, SwarmSortConfig

# Initialize tracker
config = SwarmSortConfig(
    do_embeddings=True,
    assignment_strategy='hybrid',
    uncertainty_weight=0.33
)
tracker = SwarmSortTracker(config)

# Function to draw tracking results
def draw_tracks(frame, tracked_objects, show_trails=True):
    """Draw bounding boxes and tracking information on frame."""
    # Store trail history (in production, store this outside the function)
    if not hasattr(draw_tracks, 'trails'):
        draw_tracks.trails = {}
    
    for obj in tracked_objects:
        # Get track color (consistent color per ID)
        color = np.random.RandomState(obj.id).randint(0, 255, 3).tolist()
        
        # Draw bounding box if available
        if obj.bbox is not None:
            x1, y1, x2, y2 = obj.bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID and confidence
            label = f"ID:{obj.id} ({obj.confidence:.2f})"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw center point
        cx, cy = obj.position.astype(int)
        cv2.circle(frame, (cx, cy), 5, color, -1)
        
        # Update and draw trail
        if show_trails:
            if obj.id not in draw_tracks.trails:
                draw_tracks.trails[obj.id] = []
            draw_tracks.trails[obj.id].append((cx, cy))
            
            # Keep only last 30 points
            draw_tracks.trails[obj.id] = draw_tracks.trails[obj.id][-30:]
            
            # Draw trail
            points = draw_tracks.trails[obj.id]
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], color, 2)
    
    # Clean up old trails
    active_ids = {obj.id for obj in tracked_objects}
    draw_tracks.trails = {k: v for k, v in draw_tracks.trails.items() 
                         if k in active_ids}
    
    return frame

# Example usage with video
cap = cv2.VideoCapture('video.mp4')  # Or use 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect objects (replace with your detector)
    # Here's a mock detection for demonstration
    detections = [
        Detection(
            position=np.array([100, 200]),
            confidence=0.9,
            bbox=np.array([80, 180, 120, 220])
        ),
        Detection(
            position=np.array([300, 400]),
            confidence=0.85,
            bbox=np.array([280, 380, 320, 420])
        )
    ]
    
    # Update tracker
    tracked_objects = tracker.update(detections)
    
    # Draw results
    frame = draw_tracks(frame, tracked_objects, show_trails=True)
    
    # Show recently lost tracks in different style
    recently_lost = tracker.get_recently_lost_tracks(max_frames_lost=5)
    for obj in recently_lost:
        cx, cy = obj.position.astype(int)
        cv2.circle(frame, (cx, cy), 8, (128, 128, 128), 1)  # Gray dashed circle
        cv2.putText(frame, f"Lost:{obj.id}", (cx-20, cy-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
    
    # Display frame
    cv2.imshow('SwarmSort Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Simple Visualization with Matplotlib

For a simpler visualization or for Jupyter notebooks:

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
from swarmsort import SwarmSortTracker, Detection

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 640)
ax.set_ylim(480, 0)  # Invert y-axis for image coordinates
ax.set_aspect('equal')
ax.set_title('SwarmSort Multi-Object Tracking')

tracker = SwarmSortTracker()
track_history = {}

def update_plot(frame_num):
    ax.clear()
    ax.set_xlim(0, 640)
    ax.set_ylim(480, 0)
    ax.set_title(f'Frame {frame_num}')
    
    # Generate mock detections (replace with real detections)
    detections = [
        Detection(
            position=np.array([320 + 100*np.sin(frame_num/10), 240]),
            confidence=0.9,
            bbox=np.array([300 + 100*np.sin(frame_num/10), 220, 
                          340 + 100*np.sin(frame_num/10), 260])
        ),
        Detection(
            position=np.array([200, 240 + 100*np.cos(frame_num/10)]),
            confidence=0.85,
            bbox=np.array([180, 220 + 100*np.cos(frame_num/10),
                          220, 260 + 100*np.cos(frame_num/10)])
        )
    ]
    
    # Update tracker
    tracked_objects = tracker.update(detections)
    
    # Plot tracked objects
    for obj in tracked_objects:
        # Get consistent color for track ID
        np.random.seed(obj.id)
        color = np.random.rand(3)
        
        # Draw bounding box
        if obj.bbox is not None:
            x1, y1, x2, y2 = obj.bbox
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                    linewidth=2, edgecolor=color, 
                                    facecolor='none')
            ax.add_patch(rect)
        
        # Draw center point
        ax.scatter(obj.position[0], obj.position[1], 
                  c=[color], s=100, marker='o')
        
        # Add ID label
        ax.text(obj.position[0], obj.position[1]-20, f'ID:{obj.id}',
               color=color, fontsize=12, ha='center', weight='bold')
        
        # Update history
        if obj.id not in track_history:
            track_history[obj.id] = []
        track_history[obj.id].append(obj.position.copy())
        
        # Draw trail
        if len(track_history[obj.id]) > 1:
            trail = np.array(track_history[obj.id])
            ax.plot(trail[:, 0], trail[:, 1], color=color, 
                   linewidth=2, alpha=0.5)
    
    # Clean old tracks
    active_ids = {obj.id for obj in tracked_objects}
    for track_id in list(track_history.keys()):
        if track_id not in active_ids:
            if len(track_history[track_id]) > 50:  # Remove very old tracks
                del track_history[track_id]

# Create animation
anim = FuncAnimation(fig, update_plot, frames=200, 
                    interval=50, repeat=True)
plt.show()

# To save as video:
# anim.save('tracking_visualization.mp4', writer='ffmpeg', fps=20)
```

## Advanced Features

### Track Lifecycle Management

SwarmSort provides fine-grained control over track states:

```python
# Get only currently detected tracks (alive)
alive_tracks = tracker.update(detections)

# Get recently lost tracks (useful for visualization)
recently_lost = tracker.get_recently_lost_tracks(max_frames_lost=5)

# Get all active tracks (alive + recently lost)
all_active = tracker.get_all_active_tracks(max_frames_lost=5)
```

### Uncertainty-Based Cost System

The tracker uses an intelligent cost system that considers:
- **Track age uncertainty**: Newer tracks have higher uncertainty
- **Local density**: Tracks in crowded areas have adjusted costs
- **Track reliability**: Based on detection history and consistency

### Smart Collision Handling

In dense scenarios, SwarmSort intelligently freezes embedding updates to prevent identity switches:

```python
config = SwarmSortConfig(
    collision_freeze_embeddings=True,  # Enable smart freezing
    embedding_freeze_density=1,        # Freeze when ≥1 other tracks nearby
    local_density_radius=150.0         # Define "nearby" radius
)
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

GPL 3.0 or later - see LICENSE file for details.

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
    url={https://github.com/cfosseprez/swarmsort}
}
```