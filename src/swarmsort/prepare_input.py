"""
Input preparation and verification utilities for SwarmSort.

This module provides optimized conversion functions from various detection formats
(YOLO v8/v11, etc.) to SwarmSort's Detection format, along with input verification.
"""
# ============================================================================
# STANDARD IMPORTS
# ============================================================================
import numpy as np
from typing import List, Union, Optional, Tuple, Any
import warnings
# ============================================================================
# LOGGER
# ============================================================================
from loguru import logger
# ============================================================================
# Internal imports
# ============================================================================
from .data_classes import Detection


def yolo_to_detections(
    yolo_results: Any,
    image_shape: Optional[Tuple[int, int]] = None,
    confidence_threshold: float = 0.0,
    class_filter: Optional[List[int]] = None,
    extract_embeddings: bool = False,
) -> List[Detection]:
    """
    Convert YOLO v8/v11 detection results to SwarmSort Detection format.
    
    Works with both stream=True and stream=False YOLO predictions.
    
    Args:
        yolo_results: YOLO results object from model.predict() or model.track()
                     - Single Results object (one frame)
                     - NOT a generator/list of results (use in loop for that)
        image_shape: (height, width) of the image. If None, extracts from results
        confidence_threshold: Minimum confidence to include detection
        class_filter: List of class IDs to include. None means all classes
        extract_embeddings: If True, attempts to extract visual features if available
    
    Returns:
        List of Detection objects ready for SwarmSort tracking
        
    Examples:
        >>> from ultralytics import YOLO
        >>> model = YOLO('yolov8n.pt')
        >>> 
        >>> # Option 1: stream=False (loads all frames to memory)
        >>> results = model.predict('video.mp4', stream=False)
        >>> for result in results:  # Iterate over pre-loaded results
        >>>     detections = yolo_to_detections(result)
        >>>     tracked = tracker.update(detections)
        >>> 
        >>> # Option 2: stream=True (memory efficient, processes one at a time)
        >>> results = model.predict('video.mp4', stream=True)
        >>> for result in results:  # Generator, loads one frame at a time
        >>>     detections = yolo_to_detections(result)
        >>>     tracked = tracker.update(detections)
    """
    detections = []
    
    # Handle both single result and list of results
    if not isinstance(yolo_results, list):
        yolo_results = [yolo_results]
    
    for result in yolo_results:
        # Extract boxes (works for both v8 and v11)
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            
            # Get data as numpy arrays for vectorized operations
            if hasattr(boxes, 'xyxy'):
                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
            else:
                continue
                
            if hasattr(boxes, 'conf'):
                confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf
            else:
                confs = np.ones(len(xyxy))
            
            if hasattr(boxes, 'cls'):
                classes = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else boxes.cls
            else:
                classes = np.zeros(len(xyxy))
            
            # Vectorized filtering
            valid_mask = confs >= confidence_threshold
            if class_filter is not None:
                class_mask = np.isin(classes, class_filter)
                valid_mask = valid_mask & class_mask
            
            # Extract valid detections
            valid_xyxy = xyxy[valid_mask]
            valid_confs = confs[valid_mask]
            
            # Convert to Detection objects
            for i in range(len(valid_xyxy)):
                x1, y1, x2, y2 = valid_xyxy[i]
                
                # Calculate center position
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                
                # Create Detection
                det = Detection(
                    position=np.array([cx, cy], dtype=np.float32),
                    confidence=float(valid_confs[i]),
                    bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                    embedding=None  # YOLO doesn't provide embeddings by default
                )
                detections.append(det)
    
    return detections


def yolo_to_detections_batch(
    yolo_results_list: List[Any],
    confidence_threshold: float = 0.0,
    class_filter: Optional[List[int]] = None,
) -> List[List[Detection]]:
    """
    Convert a pre-loaded list of YOLO results to SwarmSort format.
    
    NOTE: This is mainly useful for offline analysis where you want to
    process all detections first before tracking. For real-time tracking,
    use yolo_to_detections() in your frame loop instead.
    
    Args:
        yolo_results_list: List of YOLO results (pre-loaded, not a generator)
        confidence_threshold: Minimum confidence threshold
        class_filter: Optional class filter
        
    Returns:
        List of detection lists (one list per frame)
        
    Example (offline analysis):
        >>> # Load all results first (uses more memory)
        >>> model = YOLO('yolov8n.pt')
        >>> results = model.predict('video.mp4', stream=False)  # All frames in memory
        >>> all_detections = yolo_to_detections_batch(results)
        >>> 
        >>> # Now you can analyze detections before tracking
        >>> print(f"Total detections: {sum(len(d) for d in all_detections)}")
        >>> 
        >>> # Then track
        >>> for frame_detections in all_detections:
        >>>     tracked = tracker.update(frame_detections)
    
    For real-time/streaming, just use yolo_to_detections() directly:
        >>> for result in model.predict('video.mp4', stream=True):
        >>>     detections = yolo_to_detections(result)
        >>>     tracked = tracker.update(detections)
    """
    return [
        yolo_to_detections(result, confidence_threshold=confidence_threshold, 
                          class_filter=class_filter)
        for result in yolo_results_list
    ]


def numpy_to_detections(
    boxes: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    embeddings: Optional[np.ndarray] = None,
    format: str = 'xyxy'
) -> List[Detection]:
    """
    Convert numpy arrays to SwarmSort Detection format.
    
    Ultra-fast conversion for custom detection pipelines.
    
    Args:
        boxes: Array of bounding boxes. Shape (N, 4)
        confidences: Array of confidence scores. Shape (N,)
        embeddings: Optional embedding vectors. Shape (N, embedding_dim)
        format: Box format - 'xyxy', 'xywh', or 'cxcywh'
        
    Returns:
        List of Detection objects
        
    Example:
        >>> boxes = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
        >>> confs = np.array([0.9, 0.85])
        >>> detections = numpy_to_detections(boxes, confs)
    """
    if len(boxes) == 0:
        return []
    
    # Ensure float32 for efficiency
    boxes = np.asarray(boxes, dtype=np.float32)
    
    # Convert box format if needed
    if format == 'xywh':
        # Convert from [x, y, w, h] to [x1, y1, x2, y2]
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = x1 + boxes[:, 2]
        y2 = y1 + boxes[:, 3]
        boxes = np.stack([x1, y1, x2, y2], axis=1)
    elif format == 'cxcywh':
        # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
        cx = boxes[:, 0]
        cy = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)
    
    # Calculate centers (vectorized)
    centers = np.stack([
        (boxes[:, 0] + boxes[:, 2]) / 2,
        (boxes[:, 1] + boxes[:, 3]) / 2
    ], axis=1)
    
    # Handle confidences
    if confidences is None:
        confidences = np.ones(len(boxes), dtype=np.float32)
    else:
        confidences = np.asarray(confidences, dtype=np.float32)
    
    # Create Detection objects (optimized loop)
    detections = []
    for i in range(len(boxes)):
        det = Detection(
            position=centers[i],
            confidence=confidences[i],
            bbox=boxes[i],
            embedding=embeddings[i] if embeddings is not None else None
        )
        detections.append(det)
    
    return detections


def verify_detections(
    detections: List[Detection],
    image_shape: Optional[Tuple[int, int]] = None,
    auto_fix: bool = False,
    raise_on_error: bool = False
) -> Tuple[List[Detection], List[str]]:
    """
    Verify and optionally fix detection inputs.
    
    Checks for common issues and can auto-fix them.
    
    Args:
        detections: List of Detection objects to verify
        image_shape: (height, width) to check if detections are within bounds
        auto_fix: If True, attempts to fix issues (clip coords, normalize, etc.)
        raise_on_error: If True, raises exception on critical errors
        
    Returns:
        Tuple of (verified_detections, list_of_warnings)
        
    Example:
        >>> detections, warnings = verify_detections(detections, image_shape=(720, 1280))
        >>> if warnings:
        >>>     print(f"Found {len(warnings)} issues")
    """
    warnings_list = []
    verified = []
    
    if not detections:
        warnings_list.append("Empty detection list")
        return detections, warnings_list
    
    for i, det in enumerate(detections):
        issues = []
        
        # Check Detection type
        if not isinstance(det, Detection):
            issues.append(f"Detection {i}: Not a Detection object")
            if raise_on_error:
                raise TypeError(f"Detection {i} is not a Detection object")
            continue
        
        # Check position
        if det.position is None:
            issues.append(f"Detection {i}: Missing position")
            if raise_on_error:
                raise ValueError(f"Detection {i} has no position")
            continue
        
        pos = np.asarray(det.position, dtype=np.float32)
        if pos.shape != (2,):
            issues.append(f"Detection {i}: Invalid position shape {pos.shape}, expected (2,)")
            if raise_on_error:
                raise ValueError(f"Detection {i} has invalid position shape")
            continue
        
        # Check for NaN/Inf
        if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
            issues.append(f"Detection {i}: Position contains NaN or Inf")
            if raise_on_error:
                raise ValueError(f"Detection {i} position contains invalid values")
            continue
        
        # Check bounds if image_shape provided
        if image_shape is not None:
            h, w = image_shape
            if auto_fix:
                # Clip to image bounds
                pos[0] = np.clip(pos[0], 0, w - 1)
                pos[1] = np.clip(pos[1], 0, h - 1)
                det.position = pos
            else:
                if pos[0] < 0 or pos[0] >= w or pos[1] < 0 or pos[1] >= h:
                    issues.append(f"Detection {i}: Position {pos} outside image bounds {image_shape}")
        
        # Check confidence
        if det.confidence < 0 or det.confidence > 1:
            if auto_fix:
                det.confidence = np.clip(det.confidence, 0, 1)
            else:
                issues.append(f"Detection {i}: Confidence {det.confidence} not in [0, 1]")
        
        # Check bbox if present
        if det.bbox is not None:
            bbox = np.asarray(det.bbox, dtype=np.float32)
            if bbox.shape != (4,):
                issues.append(f"Detection {i}: Invalid bbox shape {bbox.shape}")
            elif image_shape is not None and auto_fix:
                h, w = image_shape
                bbox[0] = np.clip(bbox[0], 0, w - 1)
                bbox[1] = np.clip(bbox[1], 0, h - 1)
                bbox[2] = np.clip(bbox[2], 0, w - 1)
                bbox[3] = np.clip(bbox[3], 0, h - 1)
                det.bbox = bbox
        
        # Check embedding if present
        if det.embedding is not None:
            emb = np.asarray(det.embedding, dtype=np.float32)
            if len(emb.shape) != 1:
                issues.append(f"Detection {i}: Embedding must be 1D vector")
            elif np.any(np.isnan(emb)) or np.any(np.isinf(emb)):
                issues.append(f"Detection {i}: Embedding contains NaN or Inf")
            else:
                # Normalize embedding if needed
                if auto_fix:
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        det.embedding = emb / norm
        
        # Add warnings
        warnings_list.extend(issues)
        
        # Add to verified list if no critical issues
        if not any("Missing position" in issue for issue in issues):
            verified.append(det)
    
    return verified, warnings_list


def prepare_detections(
    raw_detections: Union[List[Detection], np.ndarray, Any],
    source_format: str = 'auto',
    **kwargs
) -> List[Detection]:
    """
    Universal detection preparation function.
    
    Automatically converts and verifies detections from various formats.
    
    Args:
        raw_detections: Raw detection data in any supported format
        source_format: Format hint - 'auto', 'yolo', 'numpy', 'detection'
        **kwargs: Additional arguments passed to conversion functions
        
    Returns:
        List of verified Detection objects ready for tracking
        
    Example:
        >>> # Auto-detect format and convert
        >>> detections = prepare_detections(yolo_results)
        >>> tracked = tracker.update(detections)
    """
    # Auto-detect format
    if source_format == 'auto':
        if isinstance(raw_detections, list) and len(raw_detections) > 0:
            if isinstance(raw_detections[0], Detection):
                source_format = 'detection'
            elif hasattr(raw_detections[0], 'boxes'):
                source_format = 'yolo'
            else:
                source_format = 'unknown'
        elif isinstance(raw_detections, np.ndarray):
            source_format = 'numpy'
        elif hasattr(raw_detections, 'boxes'):
            source_format = 'yolo'
        else:
            source_format = 'unknown'
    
    # Convert based on format
    if source_format == 'detection':
        detections = raw_detections
    elif source_format == 'yolo':
        detections = yolo_to_detections(raw_detections, **kwargs)
    elif source_format == 'numpy':
        detections = numpy_to_detections(raw_detections, **kwargs)
    else:
        raise ValueError(f"Unknown detection format: {source_format}")
    
    # Verify and auto-fix
    verified_detections, warnings = verify_detections(
        detections, 
        auto_fix=True,
        **kwargs
    )
    
    if warnings:
        logger.debug(f"Detection preparation found {len(warnings)} issues (auto-fixed)")
    
    return verified_detections