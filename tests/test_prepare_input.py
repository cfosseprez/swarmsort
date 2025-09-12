"""
Tests for input preparation and conversion utilities.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from swarmsort.data_classes import Detection
from swarmsort.prepare_input import (
    yolo_to_detections,
    yolo_to_detections_batch,
    numpy_to_detections,
    verify_detections,
    prepare_detections,
)


class TestYOLOConversion:
    """Test YOLO to Detection conversion."""
    
    def create_mock_yolo_result(self, num_detections=3):
        """Create a mock YOLO result object."""
        result = Mock()
        
        # Create mock boxes
        boxes = Mock()
        boxes.xyxy = np.array([
            [100, 100, 200, 200],
            [300, 150, 400, 250],
            [500, 200, 600, 300]
        ][:num_detections], dtype=np.float32)
        
        boxes.conf = np.array([0.9, 0.85, 0.7][:num_detections], dtype=np.float32)
        boxes.cls = np.array([0, 1, 0][:num_detections], dtype=np.float32)
        
        # Add cpu() method for tensor compatibility
        for attr in ['xyxy', 'conf', 'cls']:
            val = getattr(boxes, attr)
            mock_tensor = Mock()
            mock_tensor.cpu = Mock(return_value=Mock(numpy=Mock(return_value=val)))
            setattr(boxes, attr, mock_tensor)
        
        result.boxes = boxes
        return result
    
    def test_yolo_basic_conversion(self):
        """Test basic YOLO result conversion."""
        mock_result = self.create_mock_yolo_result()
        
        detections = yolo_to_detections(mock_result)
        
        assert len(detections) == 3
        assert all(isinstance(d, Detection) for d in detections)
        
        # Check first detection
        det = detections[0]
        assert np.allclose(det.position, [150, 150])  # Center of [100,100,200,200]
        assert np.isclose(det.confidence, 0.9)  # Use isclose for float comparison
        assert np.allclose(det.bbox, [100, 100, 200, 200])
    
    def test_yolo_confidence_filtering(self):
        """Test confidence threshold filtering."""
        mock_result = self.create_mock_yolo_result()
        
        detections = yolo_to_detections(mock_result, confidence_threshold=0.8)
        
        assert len(detections) == 2  # Only conf >= 0.8
        assert all(d.confidence >= 0.8 for d in detections)
    
    def test_yolo_class_filtering(self):
        """Test class-based filtering."""
        mock_result = self.create_mock_yolo_result()
        
        detections = yolo_to_detections(mock_result, class_filter=[0])
        
        assert len(detections) == 2  # Only class 0
    
    def test_yolo_empty_result(self):
        """Test handling of empty YOLO results."""
        result = Mock()
        result.boxes = None
        
        detections = yolo_to_detections(result)
        assert len(detections) == 0
    
    def test_yolo_batch_conversion(self):
        """Test batch conversion of multiple frames."""
        results = [self.create_mock_yolo_result(i+1) for i in range(3)]
        
        all_detections = yolo_to_detections_batch(results)
        
        assert len(all_detections) == 3
        assert len(all_detections[0]) == 1
        assert len(all_detections[1]) == 2
        assert len(all_detections[2]) == 3


class TestNumpyConversion:
    """Test numpy array to Detection conversion."""
    
    def test_numpy_xyxy_format(self):
        """Test conversion from xyxy format."""
        boxes = np.array([
            [100, 100, 200, 200],
            [300, 150, 400, 250]
        ], dtype=np.float32)
        
        detections = numpy_to_detections(boxes, format='xyxy')
        
        assert len(detections) == 2
        assert np.allclose(detections[0].position, [150, 150])
        assert np.allclose(detections[0].bbox, [100, 100, 200, 200])
    
    def test_numpy_xywh_format(self):
        """Test conversion from xywh format."""
        boxes = np.array([
            [100, 100, 100, 100],  # x, y, w, h
            [300, 150, 100, 100]
        ], dtype=np.float32)
        
        detections = numpy_to_detections(boxes, format='xywh')
        
        assert len(detections) == 2
        assert np.allclose(detections[0].position, [150, 150])
        assert np.allclose(detections[0].bbox, [100, 100, 200, 200])
    
    def test_numpy_cxcywh_format(self):
        """Test conversion from center format."""
        boxes = np.array([
            [150, 150, 100, 100],  # cx, cy, w, h
            [350, 200, 100, 100]
        ], dtype=np.float32)
        
        detections = numpy_to_detections(boxes, format='cxcywh')
        
        assert len(detections) == 2
        assert np.allclose(detections[0].position, [150, 150])
        assert np.allclose(detections[0].bbox, [100, 100, 200, 200])
    
    def test_numpy_with_confidences(self):
        """Test conversion with confidence scores."""
        boxes = np.array([[100, 100, 200, 200]])
        confs = np.array([0.95])
        
        detections = numpy_to_detections(boxes, confidences=confs)
        
        assert np.isclose(detections[0].confidence, 0.95)
    
    def test_numpy_with_embeddings(self):
        """Test conversion with embedding vectors."""
        boxes = np.array([[100, 100, 200, 200]])
        embeddings = np.random.randn(1, 128).astype(np.float32)
        
        detections = numpy_to_detections(boxes, embeddings=embeddings)
        
        assert detections[0].embedding is not None
        assert detections[0].embedding.shape == (128,)
    
    def test_numpy_empty_boxes(self):
        """Test handling of empty box array."""
        boxes = np.array([]).reshape(0, 4)
        
        detections = numpy_to_detections(boxes)
        assert len(detections) == 0


class TestVerification:
    """Test detection verification and auto-fixing."""
    
    def test_verify_valid_detections(self):
        """Test verification of valid detections."""
        detections = [
            Detection(position=np.array([100, 100]), confidence=0.9),
            Detection(position=np.array([200, 200]), confidence=0.8)
        ]
        
        verified, warnings = verify_detections(detections)
        
        assert len(verified) == 2
        assert len(warnings) == 0
    
    def test_verify_out_of_bounds(self):
        """Test detection of out-of-bounds positions."""
        detections = [
            Detection(position=np.array([100, 100]), confidence=0.9),
            Detection(position=np.array([1500, 100]), confidence=0.8)  # Out of bounds
        ]
        
        verified, warnings = verify_detections(
            detections, 
            image_shape=(720, 1280),
            auto_fix=False
        )
        
        assert len(warnings) > 0
        assert any("outside image bounds" in w for w in warnings)
    
    def test_verify_auto_fix_bounds(self):
        """Test auto-fixing of out-of-bounds positions."""
        detections = [
            Detection(position=np.array([-10, 100]), confidence=0.9),
            Detection(position=np.array([1500, 800]), confidence=0.8)
        ]
        
        verified, warnings = verify_detections(
            detections,
            image_shape=(720, 1280),
            auto_fix=True
        )
        
        assert len(verified) == 2
        # Check positions were clipped
        assert verified[0].position[0] >= 0
        assert verified[1].position[0] <= 1279
        assert verified[1].position[1] <= 719
    
    def test_verify_invalid_confidence(self):
        """Test detection of invalid confidence scores."""
        detections = [
            Detection(position=np.array([100, 100]), confidence=1.5),  # > 1
            Detection(position=np.array([200, 200]), confidence=-0.1)  # < 0
        ]
        
        verified, warnings = verify_detections(detections, auto_fix=False)
        
        assert len(warnings) >= 2
        assert any("not in [0, 1]" in w for w in warnings)
    
    def test_verify_auto_fix_confidence(self):
        """Test auto-fixing of confidence scores."""
        detections = [
            Detection(position=np.array([100, 100]), confidence=1.5),
            Detection(position=np.array([200, 200]), confidence=-0.1)
        ]
        
        verified, warnings = verify_detections(detections, auto_fix=True)
        
        assert verified[0].confidence == 1.0
        assert verified[1].confidence == 0.0
    
    def test_verify_nan_values(self):
        """Test detection of NaN values - they should be excluded."""
        # Add one valid detection to ensure the function works
        detections = [
            Detection(position=np.array([np.nan, 100]), confidence=0.9),  # Invalid
            Detection(position=np.array([100, np.inf]), confidence=0.8),  # Invalid  
            Detection(position=np.array([200, 200]), confidence=0.7)      # Valid
        ]
        
        verified, warnings = verify_detections(detections)
        
        # Invalid detections should be excluded
        assert len(verified) == 1  # Only the valid detection
        assert verified[0].confidence == 0.7  # The valid one
        
        # The current implementation doesn't generate warnings for NaN/Inf
        # This could be improved in the future
    
    def test_verify_invalid_bbox_shape(self):
        """Test detection of invalid bbox shapes."""
        det = Detection(position=np.array([100, 100]), confidence=0.9)
        det.bbox = np.array([100, 100])  # Wrong shape
        
        verified, warnings = verify_detections([det])
        
        assert len(warnings) > 0
        assert any("Invalid bbox shape" in w for w in warnings)
    
    def test_verify_embedding_normalization(self):
        """Test embedding normalization during verification."""
        embedding = np.array([3, 4], dtype=np.float32)  # Norm = 5
        det = Detection(
            position=np.array([100, 100]), 
            confidence=0.9,
            embedding=embedding
        )
        
        verified, warnings = verify_detections([det], auto_fix=True)
        
        # Check embedding was normalized
        norm = np.linalg.norm(verified[0].embedding)
        assert np.isclose(norm, 1.0)
    
    def test_verify_empty_list(self):
        """Test handling of empty detection list."""
        verified, warnings = verify_detections([])
        
        assert len(verified) == 0
        assert len(warnings) == 1
        assert "Empty detection list" in warnings[0]
    
    def test_verify_raise_on_error(self):
        """Test raising exceptions on critical errors."""
        det = Mock()  # Not a Detection object
        
        with pytest.raises(TypeError):
            verify_detections([det], raise_on_error=True)


class TestPrepareDetections:
    """Test universal prepare_detections function."""
    
    def test_prepare_detection_objects(self):
        """Test preparation of Detection objects."""
        detections = [
            Detection(position=np.array([100, 100]), confidence=0.9),
            Detection(position=np.array([200, 200]), confidence=0.8)
        ]
        
        prepared = prepare_detections(detections, source_format='auto')
        
        assert len(prepared) == 2
        assert all(isinstance(d, Detection) for d in prepared)
    
    def test_prepare_numpy_auto_detect(self):
        """Test auto-detection of numpy format."""
        boxes = np.array([[100, 100, 200, 200]])
        
        prepared = prepare_detections(boxes, source_format='auto')
        
        assert len(prepared) == 1
        assert isinstance(prepared[0], Detection)
    
    def test_prepare_with_verification(self):
        """Test that prepare_detections includes verification."""
        # Create detection with issues
        det = Detection(position=np.array([100, 100]), confidence=1.5)
        
        prepared = prepare_detections([det])
        
        # Should auto-fix confidence
        assert prepared[0].confidence == 1.0
    
    def test_prepare_unknown_format(self):
        """Test handling of unknown format."""
        with pytest.raises(ValueError, match="Unknown detection format"):
            prepare_detections("invalid_data", source_format='unknown')


class TestPerformance:
    """Test performance-related aspects."""
    
    def test_large_batch_conversion(self):
        """Test conversion of large batches."""
        # Create large batch
        boxes = np.random.rand(1000, 4) * 1000
        confs = np.random.rand(1000)
        
        detections = numpy_to_detections(boxes, confidences=confs)
        
        assert len(detections) == 1000
        assert all(isinstance(d, Detection) for d in detections)
    
    def test_vectorized_operations(self):
        """Test that operations are vectorized."""
        # Create batch with known values
        boxes = np.array([
            [0, 0, 100, 100],
            [100, 100, 200, 200],
            [200, 200, 300, 300]
        ])
        
        detections = numpy_to_detections(boxes, format='xyxy')
        
        # Check all centers computed correctly
        expected_centers = [[50, 50], [150, 150], [250, 250]]
        for det, expected in zip(detections, expected_centers):
            assert np.allclose(det.position, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])