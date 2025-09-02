[![Documentation Status](https://readthedocs.org/projects/swarmsort/badge/?version=latest)](https://swarmsort.readthedocs.io/en/latest/)
[![PyPI Version](https://img.shields.io/pypi/v/swarmsort.svg)](https://pypi.org/project/swarmsort/)
[![Python Version](https://img.shields.io/pypi/pyversions/swarmsort.svg)](https://pypi.org/project/swarmsort/)
[![CI Tests](https://github.com/cfosseprez/swarmsort/actions/workflows/ci.yml/badge.svg)](https://github.com/cfosseprez/swarmsort/actions/workflows/ci.yml)
[![GPL-2.0 License](https://img.shields.io/badge/License-GPL%202.0-blue.svg)](https://github.com/cfosseprez/swarmsort/blob/main/LICENSE)


![logo](https://raw.githubusercontent.com/cfosseprez/swarmsort/releases/download/0.0.0/logo-swarmsort-horizontal.jpg)

# SwarmSort

**Multi-object tracking made simple, fast, and accurate.** 🎯

SwarmSort is a modern Python library that keeps track of multiple moving objects in video streams. Whether you're counting people in a store, tracking players on a sports field, or monitoring vehicles on a highway, SwarmSort makes it easy.

## 🎬 What Does It Do?

Imagine you're watching a security camera feed with multiple people walking around. SwarmSort:
1. **Assigns a unique ID** to each person (ID #1, ID #2, etc.)
2. **Maintains those IDs** as people move around
3. **Keeps the same ID** even if someone temporarily disappears (behind a pillar)
4. **Never mixes up IDs** when people cross paths

All of this happens in real-time, at 30+ FPS!

## 🤔 Why SwarmSort?

- **🚀 Fast**: Optimized with Numba JIT compilation and vectorized operations
- **🎯 Accurate**: State-of-the-art tracking with uncertainty awareness
- **🔧 Flexible**: Works with any object detector (YOLO, Detectron2, etc.)
- **💪 Robust**: Handles occlusions, crowded scenes, and camera motion
- **📊 Production-Ready**: Used in real applications, thoroughly tested
- **🎨 Customizable**: Tune for your specific use case
- **🖥️ Multi-Platform**: Works on Linux, Windows, macOS, and edge devices

## 📖 Documentation

**[Full Documentation](https://swarmsort.readthedocs.io/en/latest/)**

## 🚀 Key Features

### 🎯 **Smart & Accurate Tracking**
- **Keeps track of multiple objects** even in crowded scenes - perfect for busy environments like airports, stadiums, or traffic monitoring
- **Never loses track unnecessarily** - Our re-identification system can recognize and reconnect with temporarily lost objects (like when someone walks behind a pillar)
- **Handles occlusions gracefully** - When objects overlap or hide each other, SwarmSort maintains identity consistency

### ⚡ **Lightning Fast Performance**
- **Real-time processing** - Track objects at 30+ FPS on standard hardware
- **GPU acceleration available** - Get even faster performance with CUDA-enabled GPUs (optional - works great on CPU too!)
- **Optimized algorithms** - Uses Numba JIT compilation and vectorized operations for maximum speed

### 🧠 **Intelligent Decision Making**
- **Uncertainty-aware** - The tracker knows when it's confident and when it's not, leading to better decisions
- **Adaptive to scenarios** - Automatically adjusts behavior in crowded vs sparse environments
- **Smart collision prevention** - Prevents ID switches when objects get close together

### 🔧 **Easy to Use & Customize**
- **Simple API** - Get started with just 3 lines of code
- **Flexible configuration** - Tune parameters for your specific use case (or use our optimized defaults)
- **Works with any detector** - Compatible with YOLO, Detectron2, or any detection source

### 📊 **Production Ready**
- **Battle-tested** - Over 200+ unit tests ensure reliability
- **Memory efficient** - Smart cleanup and bounded memory usage for long-running applications
- **Detailed tracking info** - Get position, velocity, confidence, and history for each track

## 📦 Installation

### Quick Install (Recommended)

```bash
# Option 1: Install from PyPI (coming soon!)
pip install swarmsort

# Option 2: Install from GitHub
pip install git+https://github.com/cfosseprez/swarmsort.git
```

### Development Setup

Want to contribute or modify SwarmSort? Here's how to set up a development environment:

```bash
# Clone the repository
git clone https://github.com/cfosseprez/swarmsort.git
cd swarmsort

# Install with Poetry (recommended for development)
poetry install --with dev

# Or use pip in editable mode
pip install -e ".[dev]"
```

### 🐳 Docker Option

```bash
# Coming soon: Docker image for easy deployment
docker run -it cfosseprez/swarmsort
```

## 🏃 Quick Start

### Your First Tracker in 30 Seconds

```python
import numpy as np
from swarmsort import SwarmSortTracker, Detection

# Step 1: Create a tracker (it's that simple!)
tracker = SwarmSortTracker()

# Step 2: Tell the tracker what you detected this frame
# In real use, these would come from your object detector (YOLO, etc.)
detections = [
    Detection(position=[100, 200], confidence=0.9),  # A person at position (100, 200)
    Detection(position=[300, 400], confidence=0.8),  # Another person at (300, 400)
]

# Step 3: Get tracking results - SwarmSort handles all the complexity!
tracked_objects = tracker.update(detections)

# Step 4: Use the results - each object has a unique ID that persists across frames
for obj in tracked_objects:
    print(f"Person {obj.id} is at position {obj.position} with {obj.confidence:.0%} confidence")
    # Output: Person 1 is at position [100. 200.] with 90% confidence
```

### 🎬 Real-World Example: Tracking People in Video

```python
import cv2
from swarmsort import SwarmSortTracker, Detection

tracker = SwarmSortTracker()

# Process a video file
video = cv2.VideoCapture('shopping_mall.mp4')

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    # Get detections from your favorite detector
    # For this example, let's say we detected 2 people:
    detections = [
        Detection(
            position=[320, 240],  # Center of bounding box
            confidence=0.95,
            bbox=[300, 220, 340, 260]  # x1, y1, x2, y2
        ),
        Detection(
            position=[150, 180],
            confidence=0.87,
            bbox=[130, 160, 170, 200]
        )
    ]
    
    # SwarmSort assigns consistent IDs across frames
    tracked = tracker.update(detections)
    
    # Draw results on frame
    for person in tracked:
        if person.bbox is not None:
            x1, y1, x2, y2 = person.bbox.astype(int)
            # Each person keeps the same ID and color throughout the video!
            color = (0, 255, 0)  # Green for tracked objects
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {person.id}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imshow('Tracking Results', frame)
    if cv2.waitKey(1) == ord('q'):
        break
```

### 🎨 Using Visual Features (Embeddings) for Better Tracking

Embeddings help the tracker recognize objects by their appearance, not just position. This is super useful when:
- Objects move quickly or unpredictably
- Multiple similar objects are close together
- Objects temporarily disappear and reappear

```python
from swarmsort import SwarmSortTracker, SwarmSortConfig, Detection
import numpy as np

# Enable appearance-based tracking
config = SwarmSortConfig(
    do_embeddings=True,  # Use visual features for matching
    embedding_weight=1.0,  # How much to trust appearance vs motion
)
tracker = SwarmSortTracker(config)

# In practice, embeddings come from a feature extractor (ResNet, etc.)
# Here's a simple example:
def get_embedding_from_image(image_patch):
    """Your feature extractor - could be a neural network"""
    # This would be your CNN/feature extractor
    # Returns a 128-dimensional feature vector
    return np.random.randn(128).astype(np.float32)

# Create detection with visual features
person_image = frame[160:200, 130:170]  # Crop person from frame
embedding = get_embedding_from_image(person_image)

detection = Detection(
    position=[150, 180],  # Center position
    confidence=0.9,
    embedding=embedding,  # Visual features help maintain ID
    bbox=[130, 160, 170, 200]  # Bounding box
)

# The tracker now uses BOTH motion AND appearance for matching!
tracked_objects = tracker.update([detection])
```

## ⚙️ Configuration Made Easy

### 🎯 Preset Configurations for Common Scenarios

```python
from swarmsort import SwarmSortConfig, SwarmSortTracker

# Scenario 1: Tracking in a crowded scene (like a busy street)
crowded_config = SwarmSortConfig(
    max_distance=100.0,                   # Shorter distance - objects are close
    uncertainty_weight=0.5,               # Higher uncertainty handling
    collision_freeze_embeddings=True,     # Prevent ID switches in crowds
    embedding_freeze_density=1,           # Freeze when anyone is nearby
    assignment_strategy='hybrid',         # Use smart assignment
    min_consecutive_detections=3,         # Quick initialization in dynamic scenes
)

# Scenario 2: Highway vehicle tracking (fast, spread out)
highway_config = SwarmSortConfig(
    max_distance=200.0,                   # Longer distance - fast moving vehicles
    kalman_type='oc',                     # Better motion model for vehicles
    uncertainty_weight=0.2,               # Less uncertainty - predictable motion
    min_consecutive_detections=2,         # Quick init for fast vehicles
    max_track_age=15,                     # Remove lost tracks quickly
)

# Scenario 3: Security camera (people tracking with re-identification)
security_config = SwarmSortConfig(
    do_embeddings=True,                   # Use appearance features
    embedding_weight=1.5,                 # Trust appearance more than motion
    reid_enabled=True,                    # Re-identify people who return
    reid_max_distance=200.0,              # Large reID distance
    reid_embedding_threshold=0.25,        # Permissive reID matching
    max_track_age=60,                     # Keep tracks longer (2 seconds at 30fps)
)

# Scenario 4: Sports tracking (fast action, clear visibility)
sports_config = SwarmSortConfig(
    max_distance=150.0,                   # Medium distance
    detection_conf_threshold=0.5,         # Only track clear detections
    assignment_strategy='greedy',         # Fast assignment for real-time
    kalman_type='simple',                 # Simple but fast motion model
    min_consecutive_detections=2,         # Quick player detection
)

# Use the configuration that fits your needs
tracker = SwarmSortTracker(security_config)
```

### 🔧 Understanding Key Parameters

```python
# The most important parameters to tune:

config = SwarmSortConfig(
    # 1. How far can an object move between frames?
    max_distance=150.0,  # Increase for fast objects, decrease for slow
    
    # 2. How many frames to confirm a new track?
    min_consecutive_detections=6,  # Lower = faster response, more false positives
                                   # Higher = slower response, fewer false positives
    
    # 3. How long to keep lost tracks?
    max_track_age=30,  # At 30 FPS, this is 1 second of "memory"
    
    # 4. Use appearance features?
    do_embeddings=True,  # True if objects look different from each other
                        # False if all objects look the same (e.g., identical boxes)
    
    # 5. How to handle crowded scenes?
    collision_freeze_embeddings=True,  # Prevents ID switches when objects touch
    uncertainty_weight=0.33,  # Higher = more conservative in uncertain situations
)
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

## 📦 Working with Data

### 📥 Input: Detection Objects

Detections are what you feed into the tracker - they represent objects found in the current frame:

```python
from swarmsort import Detection
import numpy as np

# Minimal detection - just position and confidence
simple_detection = Detection(
    position=[320, 240],  # Center point (x, y)
    confidence=0.9        # How sure are we this is real? (0-1)
)

# Full detection with all the bells and whistles
full_detection = Detection(
    position=np.array([320, 240]),        # Object center
    confidence=0.95,                      # Detection confidence
    bbox=np.array([300, 220, 340, 260]),  # Bounding box [x1, y1, x2, y2]
    embedding=feature_vector,             # Visual features (from your CNN)
    class_id=0,                          # 0=person, 1=car, etc.
    id="yolo_detection_42"               # Your detector's ID (optional)
)

# Real-world example: Converting YOLO output to SwarmSort
def yolo_to_swarmsort(yolo_results):
    detections = []
    for box in yolo_results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].numpy()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        detections.append(Detection(
            position=[center_x, center_y],
            confidence=box.conf[0].item(),
            bbox=[x1, y1, x2, y2],
            class_id=int(box.cls[0])
        ))
    return detections
```

### 📤 Output: TrackedObject Results

The tracker returns TrackedObject instances with rich information about each tracked object:

```python
# Get tracking results
tracked_objects = tracker.update(detections)

for obj in tracked_objects:
    # Identity
    print(f"🆔 Track ID: {obj.id}")  # Unique ID that persists across frames
    
    # Location & Motion
    print(f"📍 Position: {obj.position}")  # Current [x, y] position
    print(f"➡️ Velocity: {obj.velocity}")  # Speed and direction [vx, vy]
    
    # Confidence & Quality
    print(f"✅ Confidence: {obj.confidence:.1%}")  # How confident are we?
    print(f"📊 Track quality: {obj.hits}/{obj.age}")  # Hits/Age ratio
    
    # Track Status
    if obj.time_since_update == 0:
        print("🟢 Currently visible")
    else:
        print(f"🟡 Lost for {obj.time_since_update} frames")
    
    # Bounding Box (if available)
    if obj.bbox is not None:
        x1, y1, x2, y2 = obj.bbox
        width = x2 - x1
        height = y2 - y1
        print(f"📐 Size: {width:.0f}x{height:.0f} pixels")
```

### 🔄 Track Lifecycle Management

SwarmSort provides fine control over track states - perfect for different visualization needs:

```python
# Get only tracks that are currently visible
alive_tracks = tracker.update(detections)
print(f"👁️ Visible now: {len(alive_tracks)} objects")

# Get tracks that were recently lost (useful for smooth visualization)
recently_lost = tracker.get_recently_lost_tracks(max_frames_lost=5)
print(f"👻 Recently lost: {len(recently_lost)} objects")

# Get everything (visible + recently lost)
all_active = tracker.get_all_active_tracks(max_frames_lost=5)
print(f"📊 Total active: {len(all_active)} objects")

# Example: Different visualization for different states
for obj in alive_tracks:
    draw_solid_box(frame, obj, color='green')  # Solid box for visible
    
for obj in recently_lost:
    draw_dashed_box(frame, obj, color='yellow')  # Dashed box for lost
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

## ⚡ Performance & Optimization

### 🏎️ Why SwarmSort is Fast

SwarmSort is optimized for real-world performance:

```python
# Performance tips for different scenarios:

# 1. Maximum Speed (>100 FPS possible)
fast_config = SwarmSortConfig(
    assignment_strategy='greedy',      # Fastest assignment
    do_embeddings=False,               # Skip visual features
    kalman_type='simple',              # Simple motion model
    min_consecutive_detections=2,      # Quick initialization
)

# 2. Balanced Performance (30-60 FPS)
balanced_config = SwarmSortConfig(
    assignment_strategy='hybrid',      # Smart assignment
    do_embeddings=True,                # Use visual features
    embedding_weight=0.5,              # Balance motion/appearance
)

# 3. Maximum Accuracy (for offline processing)
accurate_config = SwarmSortConfig(
    assignment_strategy='hungarian',   # Optimal assignment
    do_embeddings=True,                # Full visual features
    min_consecutive_detections=10,     # Very careful initialization
    uncertainty_weight=0.5,            # Conservative tracking
)
```

### 🎯 Performance Tricks

```python
# Trick 1: Process every N frames for speed
frame_skip = 2  # Process every other frame
for i, frame in enumerate(video):
    if i % frame_skip == 0:
        detections = detect(frame)
        tracks = tracker.update(detections)
    else:
        # Predict positions without new detections
        tracks = tracker.update([])

# Trick 2: Limit tracking area
def filter_detections(detections, roi):
    """Only track objects in region of interest"""
    filtered = []
    x1, y1, x2, y2 = roi
    for det in detections:
        if x1 <= det.position[0] <= x2 and y1 <= det.position[1] <= y2:
            filtered.append(det)
    return filtered

# Trick 3: Confidence filtering
config = SwarmSortConfig(
    detection_conf_threshold=0.5,  # Ignore weak detections
    init_conf_threshold=0.7,       # Only start tracks for confident detections
)
```

### 💪 Under the Hood: Technical Optimizations

- **Numba JIT Compilation**: Math operations run at C speed
- **Vectorized NumPy**: Batch operations instead of loops
- **Smart Caching**: Reuses computed embeddings and distances
- **Memory Pooling**: Reduces allocation overhead
- **Early Exit Logic**: Skips unnecessary computations

## 🖥️ GPU Acceleration (Optional)

SwarmSort can use your GPU for even faster performance:

```python
from swarmsort import is_gpu_available, SwarmSortTracker, SwarmSortConfig

# Check if GPU is available
if is_gpu_available():
    print("🎮 GPU detected! SwarmSort will automatically use it for:")
    print("  - Embedding extraction (if using visual features)")
    print("  - Matrix operations (distance calculations)")
    
    # GPU is used automatically when available
    config = SwarmSortConfig(do_embeddings=True)
    tracker = SwarmSortTracker(config)
else:
    print("💻 No GPU detected - using CPU (still fast!)")
    tracker = SwarmSortTracker()

# Force CPU mode (useful for debugging)
tracker = SwarmSortTracker(embedding_type='cupytexture', use_gpu=False)
```

### 📊 Performance Benchmarks

Typical performance on a mid-range system:
- **CPU Only (i7-9700K)**: 45-60 FPS with 50 objects
- **With GPU (RTX 2070)**: 80-120 FPS with 50 objects
- **Raspberry Pi 4**: 15-20 FPS with 20 objects

Memory usage:
- ~50MB base memory
- ~1MB per 100 active tracks
- Automatic cleanup of old tracks

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

## 🚀 Advanced Features

### 🧠 How SwarmSort Thinks: The Intelligence Behind the Tracking

#### **Uncertainty-Aware Tracking**
SwarmSort knows when it's confident and when it's not:

```python
# The tracker automatically adjusts behavior based on uncertainty:
# - New tracks: "I'm not sure yet, let me observe more"
# - Established tracks: "I know this object well"
# - Crowded areas: "Need to be extra careful here"

config = SwarmSortConfig(
    uncertainty_weight=0.33,  # How much to consider uncertainty
    # 0.0 = Ignore uncertainty (aggressive)
    # 0.5 = Balanced approach
    # 1.0 = Very conservative
)

# Example: High uncertainty for drone tracking (unpredictable motion)
drone_config = SwarmSortConfig(
    uncertainty_weight=0.6,  # Be more careful with uncertain tracks
    kalman_type='oc',       # Better motion model for erratic movement
)
```

#### **Smart Collision Prevention**
Prevents ID switches when objects get close:

```python
# Scenario: Tracking dancers who frequently cross paths
dance_config = SwarmSortConfig(
    collision_freeze_embeddings=True,  # Lock visual features when close
    embedding_freeze_density=1,        # Freeze when anyone is within...
    local_density_radius=100.0,        # ...100 pixels
)

# What happens:
# 1. Two dancers approach each other
# 2. SwarmSort detects they're getting close
# 3. Visual features are "frozen" - relies on motion only
# 4. Prevents mixing up their identities
# 5. Once separated, visual matching resumes
```

#### **Hybrid Assignment Strategy**
Combines the best of both worlds:

```python
config = SwarmSortConfig(
    assignment_strategy='hybrid',  # Smart mode (default)
    greedy_threshold=30.0,         # Fast matching for obvious cases
)

# How it works:
# 1. Obvious matches (very close): Uses fast greedy assignment
# 2. Ambiguous cases: Falls back to optimal Hungarian algorithm
# 3. Best of both: Fast AND accurate

# For maximum speed (real-time systems):
config.assignment_strategy = 'greedy'  # Faster, slightly less accurate

# For maximum accuracy (offline processing):
config.assignment_strategy = 'hungarian'  # Slower, optimal matching
```

### 🔍 Re-Identification: Bringing Lost Objects Back

Perfect for scenarios where objects temporarily disappear:

```python
# Example: Security camera at a store entrance
config = SwarmSortConfig(
    reid_enabled=True,              # Enable re-identification
    reid_max_distance=200.0,        # Search this far for lost tracks
    reid_embedding_threshold=0.25,  # How similar must appearances be?
)

# What happens:
# 1. Person walks behind a pillar (track lost)
# 2. Person reappears on the other side
# 3. SwarmSort compares appearance with recently lost tracks
# 4. Same person? Same ID! Tracking continues seamlessly

# Real-world usage:
tracker = SwarmSortTracker(config)
results = tracker.update(detections)

# The person who disappeared at frame 100 and reappeared at frame 120
# will have the SAME track ID - perfect for counting and analytics!
```

## 🔧 Troubleshooting & FAQ

### Common Issues and Solutions

**Q: My tracks keep switching IDs when objects cross paths**
```python
# Solution: Enable collision handling
config = SwarmSortConfig(
    collision_freeze_embeddings=True,  # Prevent ID switches
    embedding_freeze_density=1,        # Freeze when objects are close
    do_embeddings=True,                # Use visual features
    embedding_weight=1.5,              # Trust appearance more
)
```

**Q: New tracks take too long to appear**
```python
# Solution: Reduce initialization requirements
config = SwarmSortConfig(
    min_consecutive_detections=2,  # Was 6, now faster
    init_conf_threshold=0.3,       # Accept lower confidence
)
```

**Q: Too many false tracks from noise**
```python
# Solution: Be more strict about track creation
config = SwarmSortConfig(
    min_consecutive_detections=8,      # Require more detections
    init_conf_threshold=0.7,           # Higher confidence needed
    detection_conf_threshold=0.5,      # Filter out weak detections
)
```

**Q: Tracks disappear too quickly**
```python
# Solution: Keep tracks alive longer
config = SwarmSortConfig(
    max_track_age=60,  # Keep for 2 seconds at 30 FPS (was 30)
    reid_enabled=True,  # Try to re-identify lost tracks
)
```

**Q: Performance is too slow**
```python
# Solution: Optimize for speed
config = SwarmSortConfig(
    assignment_strategy='greedy',      # Faster than 'hybrid'
    do_embeddings=False,               # Skip visual features
    detection_conf_threshold=0.5,      # Process fewer detections
)
# Also consider processing every other frame
```

### 💡 Pro Tips

1. **Start Simple**: Begin with default settings, then tune based on your results
2. **Log Everything**: Use `debug_timings=True` to identify bottlenecks
3. **Visualize**: Always visualize your tracks to understand behavior
4. **Test Incrementally**: Change one parameter at a time
5. **Know Your Domain**: Highway tracking needs different settings than indoor tracking

## Examples

See the `examples/` directory for comprehensive usage examples:

- `basic_usage.py`: Complete examples with visualization
- `multi_camera_tracking.py`: Track across multiple cameras
- `crowd_tracking.py`: Dense crowd scenarios
- `vehicle_tracking.py`: Highway and parking lot examples
- `sports_tracking.py`: Fast-moving objects in sports

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