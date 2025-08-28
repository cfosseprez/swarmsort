"""
Object motion simulator for SwarmSort testing and visualization.

This module provides tools for creating realistic object trajectories,
motion patterns, and detection sequences for testing tracking algorithms.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import random

from .data_classes import Detection


class MotionType(Enum):
    """Types of motion patterns."""
    LINEAR = "linear"
    CIRCULAR = "circular"
    RANDOM_WALK = "random_walk"
    BROWNIAN = "brownian"
    SINUSOIDAL = "sinusoidal"
    SPIRAL = "spiral"
    STOP_AND_GO = "stop_and_go"
    FIGURE_EIGHT = "figure_eight"


@dataclass
class SimulatedObject:
    """Represents a simulated object with motion properties."""
    object_id: int
    initial_position: np.ndarray
    motion_type: MotionType
    motion_params: Dict[str, Any] = field(default_factory=dict)
    
    # Object properties
    size: Tuple[float, float] = (20.0, 20.0)  # width, height
    base_confidence: float = 0.9
    confidence_noise: float = 0.1
    class_id: int = 0
    
    # Motion state
    position: np.ndarray = field(init=False)
    velocity: np.ndarray = field(init=False)
    acceleration: np.ndarray = field(init=False)
    
    # Lifecycle
    spawn_frame: int = 0
    death_frame: int = -1  # -1 means never dies
    is_active: bool = True
    
    def __post_init__(self):
        self.position = self.initial_position.copy()
        self.velocity = np.zeros(2, dtype=np.float64)
        self.acceleration = np.zeros(2, dtype=np.float64)


@dataclass 
class SimulationConfig:
    """Configuration for the simulation environment."""
    # World bounds
    world_width: float = 800.0
    world_height: float = 600.0
    
    # Detection parameters
    detection_probability: float = 0.95
    false_positive_rate: float = 0.02
    missed_detection_rate: float = 0.05
    position_noise_std: float = 2.0
    bbox_noise_std: float = 1.0
    
    # Embedding simulation
    use_embeddings: bool = False
    embedding_dim: int = 128
    embedding_noise_std: float = 0.1
    
    # Occlusion simulation
    occlusion_probability: float = 0.1
    occlusion_duration_range: Tuple[int, int] = (5, 20)
    
    # Random seed
    random_seed: Optional[int] = None


class ObjectMotionSimulator:
    """Simulates realistic object motion patterns."""
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
            random.seed(self.config.random_seed)
        
        self.objects: List[SimulatedObject] = []
        self.frame_count = 0
        self.occlusion_state: Dict[int, int] = {}  # object_id -> frames_remaining_occluded
        
        # Base embeddings for each object (if using embeddings)
        self.base_embeddings: Dict[int, np.ndarray] = {}
    
    def add_object(self, obj: SimulatedObject) -> None:
        """Add an object to the simulation."""
        self.objects.append(obj)
        
        # Generate base embedding if needed
        if self.config.use_embeddings:
            self.base_embeddings[obj.object_id] = np.random.randn(self.config.embedding_dim)
    
    def create_linear_motion_object(self, 
                                  object_id: int,
                                  start_pos: Tuple[float, float],
                                  velocity: Tuple[float, float],
                                  **kwargs) -> SimulatedObject:
        """Create an object with linear motion."""
        return SimulatedObject(
            object_id=object_id,
            initial_position=np.array(start_pos, dtype=np.float64),
            motion_type=MotionType.LINEAR,
            motion_params={"velocity": np.array(velocity, dtype=np.float64)},
            **kwargs
        )
    
    def create_circular_motion_object(self,
                                    object_id: int,
                                    center: Tuple[float, float],
                                    radius: float,
                                    angular_velocity: float,
                                    start_angle: float = 0.0,
                                    **kwargs) -> SimulatedObject:
        """Create an object with circular motion."""
        start_pos = (
            center[0] + radius * np.cos(start_angle),
            center[1] + radius * np.sin(start_angle)
        )
        return SimulatedObject(
            object_id=object_id,
            initial_position=np.array(start_pos, dtype=np.float64),
            motion_type=MotionType.CIRCULAR,
            motion_params={
                "center": np.array(center, dtype=np.float64),
                "radius": radius,
                "angular_velocity": angular_velocity,
                "current_angle": start_angle
            },
            **kwargs
        )
    
    def create_random_walk_object(self,
                                object_id: int,
                                start_pos: Tuple[float, float],
                                step_size: float = 0.3,
                                boundary_behavior: str = "bounce",
                                **kwargs) -> SimulatedObject:
        """Create an object with random walk motion."""
        return SimulatedObject(
            object_id=object_id,
            initial_position=np.array(start_pos, dtype=np.float64),
            motion_type=MotionType.RANDOM_WALK,
            motion_params={
                "step_size": step_size,
                "boundary_behavior": boundary_behavior
            },
            **kwargs
        )
    
    def create_figure_eight_object(self,
                                 object_id: int,
                                 center: Tuple[float, float],
                                 width: float,
                                 height: float,
                                 period: float,
                                 **kwargs) -> SimulatedObject:
        """Create an object following a figure-eight pattern."""
        return SimulatedObject(
            object_id=object_id,
            initial_position=np.array(center, dtype=np.float64),
            motion_type=MotionType.FIGURE_EIGHT,
            motion_params={
                "center": np.array(center, dtype=np.float64),
                "width": width,
                "height": height,
                "period": period,
                "t": 0.0
            },
            **kwargs
        )
    
    def update_object_motion(self, obj: SimulatedObject, dt: float = 1.0) -> None:
        """Update object position based on motion type."""
        if not obj.is_active:
            return
            
        if obj.motion_type == MotionType.LINEAR:
            velocity = obj.motion_params["velocity"]
            obj.position += velocity * dt
            obj.velocity = velocity
            
        elif obj.motion_type == MotionType.CIRCULAR:
            center = obj.motion_params["center"]
            radius = obj.motion_params["radius"]
            angular_vel = obj.motion_params["angular_velocity"]
            
            obj.motion_params["current_angle"] += angular_vel * dt
            angle = obj.motion_params["current_angle"]
            
            new_pos = center + radius * np.array([np.cos(angle), np.sin(angle)])
            obj.velocity = (new_pos - obj.position) / dt
            obj.position = new_pos
            
        elif obj.motion_type == MotionType.RANDOM_WALK:
            step_size = obj.motion_params["step_size"]
            boundary_behavior = obj.motion_params.get("boundary_behavior", "bounce")
            
            # Initialize direction if not set
            if "current_direction" not in obj.motion_params:
                obj.motion_params["current_direction"] = np.random.uniform(0, 2 * np.pi)
            
            # Smooth random walk with momentum
            current_dir = obj.motion_params["current_direction"]
            
            # Extremely small random change in direction for very smooth motion
            direction_change = np.random.normal(0, 0.02)  # Extremely small direction changes
            new_direction = current_dir + direction_change
            obj.motion_params["current_direction"] = new_direction
            
            # Calculate step
            step = step_size * np.array([np.cos(new_direction), np.sin(new_direction)])
            new_pos = obj.position + step
            
            # Handle boundaries
            if boundary_behavior == "bounce":
                if new_pos[0] < 0 or new_pos[0] > self.config.world_width:
                    step[0] = -step[0]
                    # Reverse horizontal direction for smoother bouncing
                    obj.motion_params["current_direction"] = np.pi - obj.motion_params["current_direction"]
                if new_pos[1] < 0 or new_pos[1] > self.config.world_height:
                    step[1] = -step[1]
                    # Reverse vertical direction for smoother bouncing
                    obj.motion_params["current_direction"] = -obj.motion_params["current_direction"]
                new_pos = obj.position + step
            elif boundary_behavior == "wrap":
                new_pos[0] = new_pos[0] % self.config.world_width
                new_pos[1] = new_pos[1] % self.config.world_height
            
            obj.velocity = step / dt
            obj.position = new_pos
            
        elif obj.motion_type == MotionType.FIGURE_EIGHT:
            center = obj.motion_params["center"]
            width = obj.motion_params["width"]
            height = obj.motion_params["height"]
            period = obj.motion_params["period"]
            
            obj.motion_params["t"] += dt
            t = obj.motion_params["t"] * 2 * np.pi / period
            
            # Lemniscate (figure-eight) parametric equations
            x = center[0] + width * np.sin(t) / (1 + np.cos(t)**2)
            y = center[1] + height * np.sin(t) * np.cos(t) / (1 + np.cos(t)**2)
            
            new_pos = np.array([x, y])
            obj.velocity = (new_pos - obj.position) / dt
            obj.position = new_pos
            
        elif obj.motion_type == MotionType.BROWNIAN:
            noise_scale = obj.motion_params.get("noise_scale", 1.0)
            noise = np.random.normal(0, noise_scale, 2)
            obj.position += noise * dt
            obj.velocity = noise
        
        # Ensure objects stay within bounds (optional)
        obj.position[0] = np.clip(obj.position[0], 0, self.config.world_width)
        obj.position[1] = np.clip(obj.position[1], 0, self.config.world_height)
    
    def generate_detection(self, obj: SimulatedObject) -> Optional[Detection]:
        """Generate a detection from an object, including noise and missing detections."""
        # Check if object should be detected
        if not obj.is_active:
            return None
            
        # Lifecycle checks
        if self.frame_count < obj.spawn_frame:
            return None
        if obj.death_frame > 0 and self.frame_count >= obj.death_frame:
            obj.is_active = False
            return None
        
        # Check occlusion
        if obj.object_id in self.occlusion_state:
            self.occlusion_state[obj.object_id] -= 1
            if self.occlusion_state[obj.object_id] <= 0:
                del self.occlusion_state[obj.object_id]
            else:
                return None  # Still occluded
        
        # Random occlusion start
        if np.random.random() < self.config.occlusion_probability:
            duration = np.random.randint(*self.config.occlusion_duration_range)
            self.occlusion_state[obj.object_id] = duration
            return None
        
        # Miss detection randomly
        if np.random.random() > self.config.detection_probability:
            return None
        
        # Add noise to position
        noisy_position = obj.position + np.random.normal(0, self.config.position_noise_std, 2)
        
        # Generate bounding box with noise
        bbox = np.array([
            noisy_position[0] - obj.size[0]/2 + np.random.normal(0, self.config.bbox_noise_std),
            noisy_position[1] - obj.size[1]/2 + np.random.normal(0, self.config.bbox_noise_std),
            noisy_position[0] + obj.size[0]/2 + np.random.normal(0, self.config.bbox_noise_std),
            noisy_position[1] + obj.size[1]/2 + np.random.normal(0, self.config.bbox_noise_std)
        ])
        
        # Generate confidence with noise
        confidence = np.clip(
            obj.base_confidence + np.random.normal(0, obj.confidence_noise),
            0.0, 1.0
        )
        
        # Generate embedding if needed
        embedding = None
        if self.config.use_embeddings and obj.object_id in self.base_embeddings:
            base_emb = self.base_embeddings[obj.object_id]
            embedding = base_emb + np.random.normal(0, self.config.embedding_noise_std, len(base_emb))
        
        return Detection(
            position=noisy_position,
            confidence=confidence,
            bbox=bbox,
            embedding=embedding,
            class_id=obj.class_id,
            id=f"det_{obj.object_id}_{self.frame_count}"
        )
    
    def generate_false_positives(self, num_fps: int = None) -> List[Detection]:
        """Generate false positive detections."""
        if num_fps is None:
            # Poisson process for false positives
            expected_fps = self.config.false_positive_rate * self.config.world_width * self.config.world_height / 10000
            num_fps = np.random.poisson(expected_fps)
        
        false_positives = []
        for i in range(num_fps):
            pos = np.array([
                np.random.uniform(0, self.config.world_width),
                np.random.uniform(0, self.config.world_height)
            ])
            
            # Random size
            size = np.random.uniform(10, 30, 2)
            bbox = np.array([
                pos[0] - size[0]/2, pos[1] - size[1]/2,
                pos[0] + size[0]/2, pos[1] + size[1]/2
            ])
            
            # Low confidence for false positives
            confidence = np.random.uniform(0.1, 0.6)
            
            # Random embedding if needed
            embedding = None
            if self.config.use_embeddings:
                embedding = np.random.randn(self.config.embedding_dim)
            
            false_positives.append(Detection(
                position=pos,
                confidence=confidence,
                bbox=bbox,
                embedding=embedding,
                class_id=np.random.randint(0, 3),
                id=f"fp_{self.frame_count}_{i}"
            ))
        
        return false_positives
    
    def step(self) -> List[Detection]:
        """Advance simulation by one time step and return detections."""
        detections = []
        
        # Update all objects
        for obj in self.objects:
            self.update_object_motion(obj)
            detection = self.generate_detection(obj)
            if detection is not None:
                detections.append(detection)
        
        # Add false positives
        false_positives = self.generate_false_positives()
        detections.extend(false_positives)
        
        self.frame_count += 1
        return detections
    
    def run_simulation(self, num_frames: int) -> List[List[Detection]]:
        """Run the simulation for a specified number of frames."""
        all_detections = []
        for _ in range(num_frames):
            detections = self.step()
            all_detections.append(detections)
        return all_detections
    
    def reset(self):
        """Reset simulation state."""
        self.frame_count = 0
        self.occlusion_state.clear()
        
        # Reset object positions
        for obj in self.objects:
            obj.position = obj.initial_position.copy()
            obj.velocity = np.zeros(2, dtype=np.float64)
            obj.is_active = True


def create_demo_scenario(scenario_name: str = "crossing_paths") -> ObjectMotionSimulator:
    """Create predefined demo scenarios."""
    config = SimulationConfig(
        detection_probability=0.95,
        false_positive_rate=0.01,
        position_noise_std=1.5,
        use_embeddings=True
    )
    
    sim = ObjectMotionSimulator(config)
    
    if scenario_name == "crossing_paths":
        # Two objects crossing paths
        sim.add_object(sim.create_linear_motion_object(
            object_id=1, start_pos=(50, 300), velocity=(2, 0), class_id=0
        ))
        sim.add_object(sim.create_linear_motion_object(
            object_id=2, start_pos=(400, 50), velocity=(0, 2), class_id=1
        ))
        
    elif scenario_name == "circular_dance":
        # Multiple objects in circular motion
        center = (400, 300)
        for i in range(4):
            angle = i * np.pi / 2
            sim.add_object(sim.create_circular_motion_object(
                object_id=i+1, center=center, radius=100,
                angular_velocity=0.05, start_angle=angle, class_id=i
            ))
            
    elif scenario_name == "mixed_motions":
        # Various motion types
        sim.add_object(sim.create_linear_motion_object(
            object_id=1, start_pos=(100, 100), velocity=(1.5, 0.5)
        ))
        sim.add_object(sim.create_circular_motion_object(
            object_id=2, center=(400, 300), radius=80, angular_velocity=0.03
        ))
        sim.add_object(sim.create_random_walk_object(
            object_id=3, start_pos=(600, 200), step_size=3.0
        ))
        sim.add_object(sim.create_figure_eight_object(
            object_id=4, center=(200, 400), width=60, height=40, period=100
        ))
    
    return sim