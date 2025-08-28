"""
Performance benchmarking and timing utilities for SwarmSort.

This module provides tools for measuring tracking performance, analyzing
timing characteristics, and generating performance reports.
"""
import time
import numpy as np
import psutil
import gc
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
import json
from pathlib import Path

from .data_classes import Detection, TrackedObject
from .core import SwarmSortTracker
from .config import SwarmSortConfig


@dataclass
class FrameTimingResult:
    """Timing results for a single frame."""

    frame_number: int
    num_detections: int
    num_tracks: int
    processing_time_ms: float
    memory_usage_mb: float
    cpu_percent: float

    # Detailed timing breakdown
    detection_preprocessing_ms: float = 0.0
    association_ms: float = 0.0
    track_update_ms: float = 0.0
    track_management_ms: float = 0.0


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""

    total_frames: int
    total_time_ms: float
    average_fps: float
    frame_results: List[FrameTimingResult] = field(default_factory=list)

    # Summary statistics
    avg_processing_time_ms: float = 0.0
    std_processing_time_ms: float = 0.0
    min_processing_time_ms: float = 0.0
    max_processing_time_ms: float = 0.0

    # Memory statistics
    avg_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0

    # Track statistics
    total_tracks_created: int = 0
    avg_active_tracks: float = 0.0
    max_active_tracks: int = 0

    # Configuration used
    config: Optional[Dict[str, Any]] = None

    def compute_summary_stats(self):
        """Compute summary statistics from frame results."""
        if not self.frame_results:
            return

        times = [f.processing_time_ms for f in self.frame_results]
        memories = [f.memory_usage_mb for f in self.frame_results]
        tracks = [f.num_tracks for f in self.frame_results]

        self.avg_processing_time_ms = np.mean(times)
        self.std_processing_time_ms = np.std(times)
        self.min_processing_time_ms = np.min(times)
        self.max_processing_time_ms = np.max(times)

        self.avg_memory_mb = np.mean(memories)
        self.peak_memory_mb = np.max(memories)

        self.avg_active_tracks = np.mean(tracks)
        self.max_active_tracks = np.max(tracks)

        if self.total_time_ms > 0:
            self.average_fps = (self.total_frames * 1000.0) / self.total_time_ms


class PerformanceProfiler:
    """Detailed performance profiler for tracking operations."""

    def __init__(self, enable_detailed_timing: bool = True):
        self.enable_detailed_timing = enable_detailed_timing
        self.timing_stack: List[Tuple[str, float]] = []
        self.current_frame_timings: Dict[str, float] = {}

    @contextmanager
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        if not self.enable_detailed_timing:
            yield
            return

        start_time = time.perf_counter()
        self.timing_stack.append((operation_name, start_time))

        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000.0
            self.current_frame_timings[operation_name] = duration_ms

            # Remove from stack
            if self.timing_stack and self.timing_stack[-1][0] == operation_name:
                self.timing_stack.pop()

    def get_frame_timings(self) -> Dict[str, float]:
        """Get timing results for the current frame."""
        return self.current_frame_timings.copy()

    def reset_frame(self):
        """Reset timing data for new frame."""
        self.current_frame_timings.clear()


class TrackingBenchmark:
    """Main benchmarking class for tracking performance."""

    def __init__(
        self, tracker: Optional[SwarmSortTracker] = None, enable_detailed_profiling: bool = True
    ):
        self.tracker = tracker
        self.profiler = PerformanceProfiler(enable_detailed_profiling)
        self.process = psutil.Process()

        # Results storage
        self.frame_results: List[FrameTimingResult] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def set_tracker(self, tracker: SwarmSortTracker):
        """Set the tracker to benchmark."""
        self.tracker = tracker

    def benchmark_frame(
        self, detections: List[Detection], frame_number: int
    ) -> Tuple[List[TrackedObject], FrameTimingResult]:
        """Benchmark processing of a single frame."""
        if self.tracker is None:
            raise ValueError("No tracker set for benchmarking")

        # Memory and CPU before processing
        gc.collect()  # Force garbage collection for consistent memory readings
        memory_before = self.process.memory_info().rss / (1024 * 1024)  # MB
        cpu_percent = self.process.cpu_percent()

        # Reset profiler for this frame
        self.profiler.reset_frame()

        # Time the overall processing
        frame_start = time.perf_counter()

        with self.profiler.time_operation("detection_preprocessing"):
            # Simulate preprocessing if needed
            pass

        with self.profiler.time_operation("tracking_update"):
            tracked_objects = self.tracker.update(detections)

        frame_end = time.perf_counter()

        # Memory after processing
        memory_after = self.process.memory_info().rss / (1024 * 1024)  # MB

        # Get detailed timings
        detailed_timings = self.profiler.get_frame_timings()

        # Create timing result
        result = FrameTimingResult(
            frame_number=frame_number,
            num_detections=len(detections),
            num_tracks=len(tracked_objects),
            processing_time_ms=(frame_end - frame_start) * 1000.0,
            memory_usage_mb=max(memory_before, memory_after),
            cpu_percent=cpu_percent,
            detection_preprocessing_ms=detailed_timings.get("detection_preprocessing", 0.0),
            association_ms=detailed_timings.get("association", 0.0),
            track_update_ms=detailed_timings.get("tracking_update", 0.0),
            track_management_ms=detailed_timings.get("track_management", 0.0),
        )

        self.frame_results.append(result)
        return tracked_objects, result

    def benchmark_sequence(
        self,
        detection_sequence: List[List[Detection]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BenchmarkResult:
        """Benchmark processing of a complete detection sequence."""
        if self.tracker is None:
            raise ValueError("No tracker set for benchmarking")

        # Reset tracker and results
        self.tracker.reset()
        self.frame_results.clear()

        # Start timing
        self.start_time = time.perf_counter()

        # Process all frames
        all_tracks = []
        for frame_num, detections in enumerate(detection_sequence):
            tracks, _ = self.benchmark_frame(detections, frame_num)
            all_tracks.append(tracks)

            if progress_callback:
                progress_callback(frame_num + 1, len(detection_sequence))

        # End timing
        self.end_time = time.perf_counter()

        # Create benchmark result
        total_time_ms = (self.end_time - self.start_time) * 1000.0

        result = BenchmarkResult(
            total_frames=len(detection_sequence),
            total_time_ms=total_time_ms,
            average_fps=0.0,  # Will be computed
            frame_results=self.frame_results.copy(),
            config=self.tracker.config.to_dict() if self.tracker.config else None,
        )

        # Compute summary statistics
        result.compute_summary_stats()

        # Get tracker statistics for track creation count
        tracker_stats = self.tracker.get_statistics()
        result.total_tracks_created = tracker_stats.get("total_tracks_created", 0)

        return result

    def save_results(self, result: BenchmarkResult, output_path: str):
        """Save benchmark results to JSON file."""
        # Convert to serializable format
        data = {
            "total_frames": result.total_frames,
            "total_time_ms": result.total_time_ms,
            "average_fps": result.average_fps,
            "avg_processing_time_ms": result.avg_processing_time_ms,
            "std_processing_time_ms": result.std_processing_time_ms,
            "min_processing_time_ms": result.min_processing_time_ms,
            "max_processing_time_ms": result.max_processing_time_ms,
            "avg_memory_mb": result.avg_memory_mb,
            "peak_memory_mb": result.peak_memory_mb,
            "total_tracks_created": result.total_tracks_created,
            "avg_active_tracks": result.avg_active_tracks,
            "max_active_tracks": result.max_active_tracks,
            "config": result.config,
            "frame_results": [
                {
                    "frame_number": f.frame_number,
                    "num_detections": f.num_detections,
                    "num_tracks": f.num_tracks,
                    "processing_time_ms": f.processing_time_ms,
                    "memory_usage_mb": f.memory_usage_mb,
                    "cpu_percent": f.cpu_percent,
                }
                for f in result.frame_results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Benchmark results saved to {output_path}")


class ScalabilityBenchmark:
    """Benchmark tracking performance with varying numbers of objects."""

    def __init__(self):
        self.results: Dict[str, BenchmarkResult] = {}

    def run_object_count_scalability(
        self,
        detection_generator: Callable[[int, int], List[List[Detection]]],
        object_counts: List[int],
        num_frames: int = 300,
        config: Optional[SwarmSortConfig] = None,
    ) -> Dict[str, BenchmarkResult]:
        """Run scalability test with varying object counts."""
        results = {}

        for count in object_counts:
            print(f"Testing with {count} objects...")

            # Generate detection sequence
            detection_sequence = detection_generator(count, num_frames)

            # Create tracker
            tracker = SwarmSortTracker(config)

            # Run benchmark
            benchmark = TrackingBenchmark(tracker)

            def progress_callback(frame, total):
                if frame % 50 == 0:
                    print(f"  Frame {frame}/{total}")

            result = benchmark.benchmark_sequence(detection_sequence, progress_callback)
            results[f"{count}_objects"] = result

            print(f"  Average FPS: {result.average_fps:.1f}")
            print(f"  Average processing time: {result.avg_processing_time_ms:.2f}ms")
            print(f"  Peak memory: {result.peak_memory_mb:.1f}MB")
            print()

        self.results.update(results)
        return results

    def run_frame_count_scalability(
        self,
        detection_generator: Callable[[int, int], List[List[Detection]]],
        frame_counts: List[int],
        num_objects: int = 10,
        config: Optional[SwarmSortConfig] = None,
    ) -> Dict[str, BenchmarkResult]:
        """Run scalability test with varying frame counts."""
        results = {}

        for count in frame_counts:
            print(f"Testing with {count} frames...")

            # Generate detection sequence
            detection_sequence = detection_generator(num_objects, count)

            # Create tracker
            tracker = SwarmSortTracker(config)

            # Run benchmark
            benchmark = TrackingBenchmark(tracker)
            result = benchmark.benchmark_sequence(detection_sequence)
            results[f"{count}_frames"] = result

            print(f"  Average FPS: {result.average_fps:.1f}")
            print(f"  Total time: {result.total_time_ms/1000:.2f}s")
            print()

        self.results.update(results)
        return results

    def generate_performance_report(self, output_path: str = "performance_report.json"):
        """Generate a comprehensive performance report."""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "python_version": psutil.version_info
                if hasattr(psutil, "version_info")
                else "unknown",
            },
            "results": {},
        }

        for test_name, result in self.results.items():
            report["results"][test_name] = {
                "total_frames": result.total_frames,
                "average_fps": result.average_fps,
                "avg_processing_time_ms": result.avg_processing_time_ms,
                "std_processing_time_ms": result.std_processing_time_ms,
                "peak_memory_mb": result.peak_memory_mb,
                "total_tracks_created": result.total_tracks_created,
                "avg_active_tracks": result.avg_active_tracks,
            }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Performance report saved to {output_path}")
        return report


def quick_benchmark(
    detection_sequence: List[List[Detection]],
    config: Optional[SwarmSortConfig] = None,
    verbose: bool = True,
) -> BenchmarkResult:
    """Quick benchmark function for simple testing."""
    tracker = SwarmSortTracker(config)
    benchmark = TrackingBenchmark(tracker)

    if verbose:

        def progress_callback(frame, total):
            if frame % 50 == 0 or frame == total:
                print(f"Processing frame {frame}/{total}")

        result = benchmark.benchmark_sequence(detection_sequence, progress_callback)
    else:
        result = benchmark.benchmark_sequence(detection_sequence)

    if verbose:
        print(f"\nBenchmark Results:")
        print(f"Total frames: {result.total_frames}")
        print(f"Average FPS: {result.average_fps:.1f}")
        print(f"Average processing time: {result.avg_processing_time_ms:.2f}ms")
        print(f"Peak memory usage: {result.peak_memory_mb:.1f}MB")
        print(f"Total tracks created: {result.total_tracks_created}")

    return result


@contextmanager
def timing_context(name: str = "Operation"):
    """Simple context manager for timing operations."""
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        duration_ms = (end - start) * 1000.0
        print(f"{name} took {duration_ms:.2f}ms")
