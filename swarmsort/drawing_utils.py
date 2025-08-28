"""
Visualization utilities for SwarmSort tracking results.

This module provides tools for drawing tracking results, creating visualizations,
and generating video output from tracking sequences.
"""
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import colorsys
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

from .data_classes import Detection, TrackedObject


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""

    frame_width: int = 800
    frame_height: int = 600
    background_color: Tuple[int, int, int] = (0, 0, 0)  # Black
    detection_color: Tuple[int, int, int] = (128, 128, 128)  # Gray
    track_thickness: int = 2
    detection_thickness: int = 1
    font_scale: float = 0.5
    font_thickness: int = 1
    show_ids: bool = True
    show_confidences: bool = True
    show_velocities: bool = False
    show_trails: bool = True
    trail_length: int = 20
    bbox_alpha: float = 0.3


class ColorManager:
    """Manages consistent colors for track IDs."""

    def __init__(self, saturation: float = 0.8, value: float = 1.0):
        self.saturation = saturation
        self.value = value
        self.color_cache: Dict[int, Tuple[int, int, int]] = {}

    def get_color(self, track_id: int) -> Tuple[int, int, int]:
        """Get a consistent color for a track ID."""
        if track_id not in self.color_cache:
            # Generate color based on track ID
            hue = (track_id * 137.5) % 360  # Golden angle approximation
            rgb = colorsys.hsv_to_rgb(hue / 360, self.saturation, self.value)
            # Convert to 0-255 range
            self.color_cache[track_id] = tuple(int(c * 255) for c in rgb)

        return self.color_cache[track_id]

    def reset(self):
        """Clear color cache."""
        self.color_cache.clear()


class TrackingVisualizer:
    """Main visualization class for tracking results."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.color_manager = ColorManager()
        self.trail_history: Dict[int, List[Tuple[float, float]]] = {}

        if not MATPLOTLIB_AVAILABLE and not OPENCV_AVAILABLE:
            raise ImportError(
                "Either matplotlib or opencv-python must be installed for visualization"
            )

    def draw_frame_matplotlib(
        self, detections: List[Detection], tracks: List[TrackedObject], frame_num: int, ax=None
    ) -> None:
        """Draw a single frame using matplotlib."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for this function")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        ax.clear()
        ax.set_xlim(0, self.config.frame_width)
        ax.set_ylim(0, self.config.frame_height)
        ax.set_aspect("equal")
        ax.invert_yaxis()  # Match image coordinates
        ax.set_title(f"Frame {frame_num} - {len(tracks)} tracks, {len(detections)} detections")

        # Draw detections
        for det in detections:
            x, y = det.position
            circle = plt.Circle((x, y), 2, color="gray", alpha=0.5)
            ax.add_patch(circle)

            if det.bbox is not None and len(det.bbox) >= 4:
                bbox = patches.Rectangle(
                    (det.bbox[0], det.bbox[1]),
                    det.bbox[2] - det.bbox[0],
                    det.bbox[3] - det.bbox[1],
                    linewidth=1,
                    edgecolor="gray",
                    facecolor="none",
                    alpha=0.3,
                )
                ax.add_patch(bbox)

        # Draw tracks
        for track in tracks:
            color = np.array(self.color_manager.get_color(track.id)) / 255.0
            x, y = track.position

            # Update trail history
            if track.id not in self.trail_history:
                self.trail_history[track.id] = []
            self.trail_history[track.id].append((x, y))
            if len(self.trail_history[track.id]) > self.config.trail_length:
                self.trail_history[track.id].pop(0)

            # Draw trail
            if self.config.show_trails and len(self.trail_history[track.id]) > 1:
                trail_points = np.array(self.trail_history[track.id])
                ax.plot(trail_points[:, 0], trail_points[:, 1], color=color, alpha=0.5, linewidth=1)

            # Draw current position
            circle = plt.Circle((x, y), 8, color=color, alpha=0.8)
            ax.add_patch(circle)

            # Draw bounding box
            if track.bbox is not None and len(track.bbox) >= 4:
                bbox = patches.Rectangle(
                    (track.bbox[0], track.bbox[1]),
                    track.bbox[2] - track.bbox[0],
                    track.bbox[3] - track.bbox[1],
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                    alpha=0.7,
                )
                ax.add_patch(bbox)

            # Draw ID and info
            if self.config.show_ids:
                text = f"ID:{track.id}"
                if self.config.show_confidences:
                    text += f"\nConf:{track.confidence:.2f}"
                if self.config.show_velocities:
                    text += f"\nVel:({track.velocity[0]:.1f},{track.velocity[1]:.1f})"

                ax.text(
                    x + 10,
                    y - 10,
                    text,
                    fontsize=8,
                    color=color,
                    weight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                )

    def draw_frame_opencv(
        self,
        detections: List[Detection],
        tracks: List[TrackedObject],
        frame_num: int,
        frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Draw a single frame using OpenCV."""
        if not OPENCV_AVAILABLE:
            raise ImportError("opencv-python is required for this function")

        if frame is None:
            frame = np.zeros((self.config.frame_height, self.config.frame_width, 3), dtype=np.uint8)
            frame[:] = self.config.background_color
        else:
            frame = frame.copy()

        # Draw detections
        for det in detections:
            x, y = int(det.position[0]), int(det.position[1])
            cv2.circle(frame, (x, y), 5, self.config.detection_color, -1)

            if det.bbox is not None and len(det.bbox) >= 4:
                x1, y1, x2, y2 = [int(coord) for coord in det.bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.config.detection_color, 1)

        # Draw tracks
        for track in tracks:
            color = self.color_manager.get_color(track.id)
            x, y = int(track.position[0]), int(track.position[1])

            # Update trail history
            if track.id not in self.trail_history:
                self.trail_history[track.id] = []
            self.trail_history[track.id].append((x, y))
            if len(self.trail_history[track.id]) > self.config.trail_length:
                self.trail_history[track.id].pop(0)

            # Draw trail
            if self.config.show_trails and len(self.trail_history[track.id]) > 1:
                trail_points = np.array(self.trail_history[track.id], dtype=np.int32)
                for i in range(1, len(trail_points)):
                    alpha = i / len(trail_points)
                    trail_color = tuple(int(c * alpha) for c in color)
                    cv2.line(
                        frame, tuple(trail_points[i - 1]), tuple(trail_points[i]), trail_color, 1
                    )

            # Draw current position
            cv2.circle(frame, (x, y), 8, color, -1)
            cv2.circle(frame, (x, y), 8, (255, 255, 255), 1)

            # Draw bounding box
            if track.bbox is not None and len(track.bbox) >= 4:
                x1, y1, x2, y2 = [int(coord) for coord in track.bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.config.track_thickness)

            # Draw ID and info
            if self.config.show_ids:
                text = f"ID:{track.id}"
                if self.config.show_confidences:
                    text += f" C:{track.confidence:.2f}"

                # Text background
                text_size = cv2.getTextSize(
                    text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.font_scale,
                    self.config.font_thickness,
                )[0]
                cv2.rectangle(
                    frame,
                    (x + 10, y - 25),
                    (x + 10 + text_size[0], y - 25 + text_size[1] + 5),
                    (0, 0, 0),
                    -1,
                )

                # Text
                cv2.putText(
                    frame,
                    text,
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.font_scale,
                    color,
                    self.config.font_thickness,
                )

        # Frame info
        info_text = f"Frame: {frame_num} | Tracks: {len(tracks)} | Detections: {len(detections)}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def create_video_sequence(
        self,
        detection_sequence: List[List[Detection]],
        track_sequence: List[List[TrackedObject]],
        output_path: str = "tracking_output.mp4",
        fps: int = 30,
    ) -> None:
        """Create a video from detection and track sequences."""
        if not OPENCV_AVAILABLE:
            raise ImportError("opencv-python is required for video creation")

        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            output_path, fourcc, fps, (self.config.frame_width, self.config.frame_height)
        )

        try:
            for frame_num, (detections, tracks) in enumerate(
                zip(detection_sequence, track_sequence)
            ):
                frame = self.draw_frame_opencv(detections, tracks, frame_num)
                video_writer.write(frame)

                # Progress indicator
                if frame_num % 50 == 0:
                    print(f"Processing frame {frame_num}...")

            print(f"Video saved to {output_path}")

        finally:
            video_writer.release()

    def show_sequence_matplotlib(
        self,
        detection_sequence: List[List[Detection]],
        track_sequence: List[List[TrackedObject]],
        interval: int = 100,
        save_path: Optional[str] = None,
    ) -> None:
        """Show animated sequence using matplotlib."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for this function")

        fig, ax = plt.subplots(figsize=(12, 9))

        def animate(frame_num):
            if frame_num < len(detection_sequence) and frame_num < len(track_sequence):
                self.draw_frame_matplotlib(
                    detection_sequence[frame_num], track_sequence[frame_num], frame_num, ax
                )

        anim = FuncAnimation(
            fig, animate, frames=len(detection_sequence), interval=interval, blit=False, repeat=True
        )

        if save_path:
            anim.save(save_path, writer="pillow", fps=1000 // interval)
            print(f"Animation saved to {save_path}")
        else:
            plt.show()

        return anim

    def reset(self):
        """Reset visualization state."""
        self.trail_history.clear()
        self.color_manager.reset()


def quick_visualize(
    detections: List[Detection],
    tracks: List[TrackedObject],
    frame_num: int = 0,
    method: str = "matplotlib",
) -> None:
    """Quick visualization function for debugging."""
    visualizer = TrackingVisualizer()

    if method == "matplotlib" and MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(10, 8))
        visualizer.draw_frame_matplotlib(detections, tracks, frame_num, ax)
        plt.show()
    elif method == "opencv" and OPENCV_AVAILABLE:
        frame = visualizer.draw_frame_opencv(detections, tracks, frame_num)
        cv2.imshow(f"Tracking Frame {frame_num}", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Requested visualization method not available. Install matplotlib or opencv-python.")
