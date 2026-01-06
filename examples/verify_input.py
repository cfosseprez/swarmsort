#!/usr/bin/env python3
"""
Input verification utilities for SwarmSort tracker.

This module provides comprehensive validation functions to ensure inputs
to the SwarmSort tracker are correct and properly formatted.
"""

import sys
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Tuple
import warnings

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarmsort import SwarmSortTracker, SwarmSortConfig, Detection
from swarmsort.data_classes import TrackedObject


class InputVerifier:
    """Comprehensive input verification for SwarmSort tracker."""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the verifier.
        
        Args:
            verbose: If True, print detailed validation messages
        """
        self.verbose = verbose
        self.errors = []
        self.warnings = []
    
    def _log(self, message: str, level: str = "info"):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            prefix = {
                "info": "ℹ️ ",
                "success": "✅",
                "warning": "⚠️ ",
                "error": "❌"
            }.get(level, "")
            print(f"{prefix} {message}")
    
    def _add_error(self, error: str):
        """Add an error to the list."""
        self.errors.append(error)
        self._log(error, "error")
    
    def _add_warning(self, warning: str):
        """Add a warning to the list."""
        self.warnings.append(warning)
        self._log(warning, "warning")
    
    def verify_detection(self, detection: Detection, index: Optional[int] = None) -> bool:
        """
        Verify a single detection object.
        
        Args:
            detection: Detection object to verify
            index: Optional index for error messages
            
        Returns:
            True if detection is valid, False otherwise
        """
        prefix = f"Detection {index}: " if index is not None else "Detection: "
        is_valid = True
        
        # Check required attributes
        if not hasattr(detection, 'position') or detection.position is None:
            self._add_error(f"{prefix}Missing required 'position' attribute")
            is_valid = False
        else:
            # Verify position format
            if not isinstance(detection.position, np.ndarray):
                self._add_error(f"{prefix}Position must be numpy array, got {type(detection.position)}")
                is_valid = False
            elif detection.position.shape not in [(2,), (2, 1)]:
                self._add_error(f"{prefix}Position shape must be (2,) or (2,1), got {detection.position.shape}")
                is_valid = False
            elif not np.isfinite(detection.position).all():
                self._add_error(f"{prefix}Position contains non-finite values: {detection.position}")
                is_valid = False
        
        # Check confidence
        if not hasattr(detection, 'confidence') or detection.confidence is None:
            self._add_error(f"{prefix}Missing required 'confidence' attribute")
            is_valid = False
        else:
            if not isinstance(detection.confidence, (float, np.floating, int)):
                self._add_error(f"{prefix}Confidence must be numeric, got {type(detection.confidence)}")
                is_valid = False
            elif not 0 <= detection.confidence <= 1:
                self._add_warning(f"{prefix}Confidence should be in [0,1], got {detection.confidence}")
        
        # Check optional bbox
        if hasattr(detection, 'bbox') and detection.bbox is not None:
            if not isinstance(detection.bbox, np.ndarray):
                self._add_error(f"{prefix}Bbox must be numpy array, got {type(detection.bbox)}")
                is_valid = False
            elif detection.bbox.shape not in [(4,), (4, 1)]:
                self._add_error(f"{prefix}Bbox shape must be (4,), got {detection.bbox.shape}")
                is_valid = False
            elif len(detection.bbox) == 4:
                x1, y1, x2, y2 = detection.bbox
                if x2 <= x1 or y2 <= y1:
                    self._add_warning(f"{prefix}Invalid bbox: x2<=x1 or y2<=y1: {detection.bbox}")
        
        # Check optional embedding
        if hasattr(detection, 'embedding') and detection.embedding is not None:
            if not isinstance(detection.embedding, np.ndarray):
                self._add_error(f"{prefix}Embedding must be numpy array, got {type(detection.embedding)}")
                is_valid = False
            elif detection.embedding.ndim != 1:
                self._add_error(f"{prefix}Embedding must be 1D array, got shape {detection.embedding.shape}")
                is_valid = False
            elif not np.isfinite(detection.embedding).all():
                self._add_error(f"{prefix}Embedding contains non-finite values")
                is_valid = False
            elif detection.embedding.dtype not in [np.float32, np.float64]:
                self._add_warning(f"{prefix}Embedding should be float32/float64, got {detection.embedding.dtype}")
        
        # Check optional class_id
        if hasattr(detection, 'class_id') and detection.class_id is not None:
            if not isinstance(detection.class_id, (int, np.integer)):
                self._add_warning(f"{prefix}class_id should be integer, got {type(detection.class_id)}")
            elif detection.class_id < 0:
                self._add_warning(f"{prefix}class_id should be non-negative, got {detection.class_id}")
        
        return is_valid
    
    def verify_detections_batch(self, detections: List[Detection]) -> bool:
        """
        Verify a batch of detections.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            True if all detections are valid, False otherwise
        """
        self._log(f"Verifying {len(detections)} detections...", "info")
        
        if not isinstance(detections, list):
            self._add_error(f"Detections must be a list, got {type(detections)}")
            return False
        
        if len(detections) == 0:
            self._add_warning("Empty detection list provided")
            return True
        
        all_valid = True
        embedding_dims = set()
        
        for i, det in enumerate(detections):
            if not isinstance(det, Detection):
                self._add_error(f"Detection {i}: Must be Detection object, got {type(det)}")
                all_valid = False
                continue
            
            if not self.verify_detection(det, i):
                all_valid = False
            
            # Check embedding consistency
            if hasattr(det, 'embedding') and det.embedding is not None:
                embedding_dims.add(det.embedding.shape[0])
        
        # Check embedding dimension consistency
        if len(embedding_dims) > 1:
            self._add_error(f"Inconsistent embedding dimensions: {embedding_dims}")
            all_valid = False
        elif len(embedding_dims) == 1:
            self._log(f"All embeddings have consistent dimension: {embedding_dims.pop()}", "success")
        
        return all_valid
    
    def verify_config(self, config: SwarmSortConfig) -> bool:
        """
        Verify SwarmSort configuration.
        
        Args:
            config: SwarmSortConfig object
            
        Returns:
            True if configuration is valid, False otherwise
        """
        self._log("Verifying configuration...", "info")
        is_valid = True
        
        if not isinstance(config, SwarmSortConfig):
            self._add_error(f"Config must be SwarmSortConfig object, got {type(config)}")
            return False
        
        # Check critical parameters
        if config.max_distance <= 0:
            self._add_error(f"max_distance must be positive, got {config.max_distance}")
            is_valid = False
        
        if config.assignment_strategy not in ["hungarian", "greedy", "hybrid"]:
            self._add_error(f"Invalid assignment_strategy: {config.assignment_strategy}")
            is_valid = False
        
        if config.kalman_type not in ["simple", "oc"]:
            self._add_error(f"Invalid kalman_type: {config.kalman_type}")
            is_valid = False
        
        if not 0 <= config.uncertainty_weight <= 1:
            self._add_warning(f"uncertainty_weight should be in [0,1], got {config.uncertainty_weight}")
        
        if not 0 <= config.embedding_weight <= 10:
            self._add_warning(f"embedding_weight typically in [0,10], got {config.embedding_weight}")
        
        if config.min_consecutive_detections < 1:
            self._add_error(f"min_consecutive_detections must be >= 1, got {config.min_consecutive_detections}")
            is_valid = False
        
        if config.max_track_age < 1:
            self._add_error(f"max_track_age must be >= 1, got {config.max_track_age}")
            is_valid = False
        
        # Check logical consistency
        if config.do_embeddings and config.embedding_weight == 0:
            self._add_warning("do_embeddings=True but embedding_weight=0, embeddings won't affect tracking")
        
        if config.reid_enabled and not config.do_embeddings:
            self._add_warning("reid_enabled=True but do_embeddings=False, ReID won't work")
        
        if config.greedy_threshold > config.max_distance:
            self._add_warning(f"greedy_threshold ({config.greedy_threshold}) > max_distance ({config.max_distance})")
        
        self._log("Configuration validation complete", "success" if is_valid else "error")
        return is_valid
    
    def verify_tracker_update_inputs(
        self, 
        detections: List[Detection],
        config: Optional[SwarmSortConfig] = None
    ) -> bool:
        """
        Verify inputs for tracker.update() call.
        
        Args:
            detections: List of detections to verify
            config: Optional config to verify
            
        Returns:
            True if all inputs are valid, False otherwise
        """
        self.errors = []
        self.warnings = []
        
        self._log("="*60, "info")
        self._log("SwarmSort Input Verification", "info")
        self._log("="*60, "info")
        
        all_valid = True
        
        # Verify config if provided
        if config is not None:
            if not self.verify_config(config):
                all_valid = False
        
        # Verify detections
        if not self.verify_detections_batch(detections):
            all_valid = False
        
        # Summary
        self._log("="*60, "info")
        if all_valid and not self.warnings:
            self._log("All inputs are valid! ✅", "success")
        elif all_valid and self.warnings:
            self._log(f"Inputs valid with {len(self.warnings)} warnings ⚠️", "warning")
        else:
            self._log(f"Found {len(self.errors)} errors ❌", "error")
        
        return all_valid
    
    def verify_tracked_objects(self, tracked_objects: List[TrackedObject]) -> bool:
        """
        Verify tracked objects returned by the tracker.
        
        Args:
            tracked_objects: List of TrackedObject instances
            
        Returns:
            True if all objects are valid, False otherwise
        """
        self._log(f"Verifying {len(tracked_objects)} tracked objects...", "info")
        is_valid = True
        
        if not isinstance(tracked_objects, list):
            self._add_error(f"Tracked objects must be a list, got {type(tracked_objects)}")
            return False
        
        track_ids = set()
        for i, obj in enumerate(tracked_objects):
            if not hasattr(obj, 'id'):
                self._add_error(f"TrackedObject {i}: Missing 'id' attribute")
                is_valid = False
            else:
                if obj.id in track_ids:
                    self._add_error(f"Duplicate track ID: {obj.id}")
                    is_valid = False
                track_ids.add(obj.id)
            
            if not hasattr(obj, 'position'):
                self._add_error(f"TrackedObject {i}: Missing 'position' attribute")
                is_valid = False
            
            if not hasattr(obj, 'confidence'):
                self._add_error(f"TrackedObject {i}: Missing 'confidence' attribute")
                is_valid = False
            
            # Check state consistency
            if hasattr(obj, 'time_since_update'):
                if obj.time_since_update < 0:
                    self._add_error(f"TrackedObject {obj.id}: Invalid time_since_update: {obj.time_since_update}")
                    is_valid = False
        
        return is_valid


def create_test_detections(
    num_detections: int = 10,
    with_embeddings: bool = True,
    embedding_dim: int = 128,
    with_errors: bool = False
) -> List[Detection]:
    """
    Create test detections for verification.
    
    Args:
        num_detections: Number of detections to create
        with_embeddings: Whether to include embeddings
        embedding_dim: Dimension of embeddings
        with_errors: If True, introduce some errors for testing
        
    Returns:
        List of Detection objects
    """
    detections = []
    
    for i in range(num_detections):
        # Create position
        if with_errors and i == 0:
            # Invalid position for testing
            position = np.array([np.nan, 100])
        else:
            position = np.random.uniform(0, 500, 2).astype(np.float32)
        
        # Create confidence
        if with_errors and i == 1:
            confidence = 1.5  # Out of range
        else:
            confidence = np.random.uniform(0.5, 1.0)
        
        # Create bbox
        if i % 2 == 0:  # Only half have bboxes
            if with_errors and i == 2:
                # Invalid bbox (x2 < x1)
                bbox = np.array([100, 100, 50, 150], dtype=np.float32)
            else:
                x, y = position
                size = np.random.uniform(20, 50)
                bbox = np.array([x-size/2, y-size/2, x+size/2, y+size/2], dtype=np.float32)
        else:
            bbox = None
        
        # Create embedding
        if with_embeddings:
            if with_errors and i == 3:
                # Wrong dimension
                embedding = np.random.randn(embedding_dim + 10).astype(np.float32)
            elif with_errors and i == 4:
                # Contains NaN
                embedding = np.random.randn(embedding_dim).astype(np.float32)
                embedding[0] = np.nan
            else:
                embedding = np.random.randn(embedding_dim).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)  # Normalize
        else:
            embedding = None
        
        det = Detection(
            position=position,
            confidence=confidence,
            bbox=bbox,
            embedding=embedding,
            class_id=i % 3,
            id=f"det_{i}"
        )
        detections.append(det)
    
    return detections


def main():
    """Main function demonstrating input verification."""
    print("\n" + "="*70)
    print(" SwarmSort Input Verification Tool")
    print("="*70)
    
    verifier = InputVerifier(verbose=True)
    
    # Test 1: Valid inputs
    print("\n1. Testing VALID inputs:")
    print("-" * 40)
    valid_detections = create_test_detections(
        num_detections=5,
        with_embeddings=True,
        with_errors=False
    )
    config = SwarmSortConfig(
        max_distance=150.0,
        do_embeddings=True,
        assignment_strategy="hybrid"
    )
    result = verifier.verify_tracker_update_inputs(valid_detections, config)
    
    # Test 2: Invalid inputs
    print("\n2. Testing INVALID inputs (with errors):")
    print("-" * 40)
    invalid_detections = create_test_detections(
        num_detections=5,
        with_embeddings=True,
        with_errors=True
    )
    verifier.errors = []
    verifier.warnings = []
    result = verifier.verify_tracker_update_inputs(invalid_detections)
    
    # Test 3: Configuration issues
    print("\n3. Testing configuration warnings:")
    print("-" * 40)
    config_with_issues = SwarmSortConfig(
        max_distance=150.0,
        do_embeddings=True,
        embedding_weight=0.0,  # Warning: embeddings enabled but weight is 0
        reid_enabled=True,
        assignment_strategy="hybrid"
    )
    verifier.errors = []
    verifier.warnings = []
    result = verifier.verify_config(config_with_issues)
    
    # Test 4: Run actual tracking and verify outputs
    print("\n4. Testing with actual tracker:")
    print("-" * 40)
    tracker = SwarmSortTracker(SwarmSortConfig(
        max_distance=150.0,
        do_embeddings=False
    ))
    
    detections = create_test_detections(
        num_detections=10,
        with_embeddings=False,
        with_errors=False
    )
    
    # Verify inputs
    verifier.errors = []
    verifier.warnings = []
    if verifier.verify_tracker_update_inputs(detections):
        # Run tracking
        tracked_objects = tracker.update(detections)
        
        # Verify outputs
        print("\nVerifying tracker outputs:")
        verifier.verify_tracked_objects(tracked_objects)
        print(f"Tracked {len(tracked_objects)} objects successfully")
    
    # Test 5: Embedding consistency check
    print("\n5. Testing embedding consistency:")
    print("-" * 40)
    mixed_embeddings = []
    for i in range(5):
        dim = 128 if i < 3 else 256  # Inconsistent dimensions
        det = Detection(
            position=np.random.uniform(0, 500, 2).astype(np.float32),
            confidence=0.9,
            embedding=np.random.randn(dim).astype(np.float32)
        )
        mixed_embeddings.append(det)
    
    verifier.errors = []
    verifier.warnings = []
    verifier.verify_detections_batch(mixed_embeddings)
    
    # Summary
    print("\n" + "="*70)
    print(" Verification Complete!")
    print("="*70)
    print("\nUsage in your code:")
    print("-" * 40)
    print("""
from verify_input import InputVerifier

# Create verifier
verifier = InputVerifier(verbose=True)

# Verify before tracking
if verifier.verify_tracker_update_inputs(detections, config):
    tracked_objects = tracker.update(detections)
else:
    print(f"Input validation failed with {len(verifier.errors)} errors")
    """)


if __name__ == "__main__":
    main()