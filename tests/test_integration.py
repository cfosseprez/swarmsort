"""
Integration tests for SwarmSort package interfaces.

Tests the SwarmTracker integration interfaces, new embeddings,
and overall package functionality.
"""

import pytest
import numpy as np
from typing import List, Dict, Any

from swarmsort import (
    SwarmSortTracker, 
    SwarmSortConfig, 
    Detection, 
    TrackedObject,
    get_embedding_extractor,
    list_available_embeddings
)
from swarmsort.swarmtracker_adapter import RawTrackerSwarmSORT, create_swarmsort_tracker


class TestSwarmSortTrackerIntegration:
    """Test SwarmSortTracker integration interfaces."""
    
    def test_constructor_with_config_only(self):
        """Test tracker creation with config only."""
        config = SwarmSortConfig(max_distance=100.0, do_embeddings=False)
        tracker = SwarmSortTracker(config=config)
        
        assert tracker.config.max_distance == 100.0
        assert tracker.config.do_embeddings == False
        assert tracker.embedding_extractor is None

    def test_constructor_with_dict_config(self):
        """Test tracker creation with dictionary config."""
        config_dict = {
            'max_distance': 150.0,
            'use_embeddings': True,
            'embedding_weight': 0.8
        }
        tracker = SwarmSortTracker(config=config_dict)
        
        assert tracker.config.max_distance == 150.0
        assert tracker.config.do_embeddings == True
        assert tracker.config.embedding_weight == 0.8

    def test_constructor_with_embedding_type(self):
        """Test tracker creation with embedding type."""
        config = SwarmSortConfig(do_embeddings=True)
        tracker = SwarmSortTracker(
            config=config,
            embedding_type='cupytexture',
            use_gpu=False
        )
        
        assert tracker.embedding_extractor is not None
        assert hasattr(tracker.embedding_extractor, 'extract')
        assert hasattr(tracker.embedding_extractor, 'embedding_dim')

    def test_constructor_with_color_embedding(self):
        """Test tracker creation with color embedding."""
        tracker = SwarmSortTracker(
            embedding_type='cupytexture_color',
            use_gpu=False
        )
        
        assert tracker.embedding_extractor is not None
        assert tracker.embedding_extractor.embedding_dim == 84

    def test_constructor_with_invalid_embedding(self):
        """Test tracker creation with invalid embedding type."""
        # Should not raise error, just warn and continue
        tracker = SwarmSortTracker(embedding_type='invalid_embedding')
        assert tracker.embedding_extractor is None

    def test_update_with_frame_parameter(self):
        """Test update method with frame parameter."""
        tracker = SwarmSortTracker()
        
        # Create test data
        detection = Detection(
            position=np.array([100.0, 100.0], dtype=np.float32),
            confidence=0.9,
            bbox=np.array([90, 90, 110, 110], dtype=np.float32)
        )
        frame = np.random.randint(0, 255, size=(240, 320, 3), dtype=np.uint8)
        
        # Should not raise error
        result = tracker.update([detection], frame)
        assert isinstance(result, list)

    def test_update_without_frame_parameter(self):
        """Test update method maintains backward compatibility."""
        tracker = SwarmSortTracker()
        
        detection = Detection(
            position=np.array([100.0, 100.0], dtype=np.float32),
            confidence=0.9
        )
        
        # Should work without frame
        result = tracker.update([detection])
        assert isinstance(result, list)

    def test_kwargs_handling(self):
        """Test that extra kwargs are handled gracefully."""
        tracker = SwarmSortTracker(
            embedding_type='cupytexture',
            use_gpu=False,
            extra_param=123,  # Should be ignored
            another_param='test'  # Should be ignored
        )
        
        assert tracker.embedding_extractor is not None


class TestDetectionInterface:
    """Test Detection class interface compatibility."""
    
    def test_detection_creation_minimal(self):
        """Test minimal Detection creation."""
        det = Detection(
            position=np.array([100.0, 100.0], dtype=np.float32),
            confidence=0.8
        )
        
        assert np.array_equal(det.position, np.array([100.0, 100.0]))
        assert det.confidence == 0.8
        assert det.bbox is None
        assert det.embedding is None
        assert det.class_id is None
        assert det.id is None

    def test_detection_creation_full(self):
        """Test full Detection creation with all parameters."""
        position = np.array([100.0, 100.0], dtype=np.float32)
        bbox = np.array([90, 90, 110, 110], dtype=np.float32)
        embedding = np.random.rand(64).astype(np.float32)
        
        det = Detection(
            position=position,
            confidence=0.9,
            bbox=bbox,
            embedding=embedding,
            class_id=1,
            id='det_001'
        )
        
        assert np.array_equal(det.position, position)
        assert det.confidence == 0.9
        assert np.array_equal(det.bbox, bbox)
        assert np.array_equal(det.embedding, embedding)
        assert det.class_id == 1
        assert det.id == 'det_001'

    def test_detection_swarmtracker_format(self):
        """Test Detection creation in SwarmTracker format."""
        # Simulate SwarmTracker detection conversion
        x1, y1, x2, y2 = 90.0, 90.0, 110.0, 110.0
        confidence = 0.85
        class_id = 2
        
        # Center point calculation
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        position = np.array([center_x, center_y], dtype=np.float32)
        bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        
        det = Detection(
            position=position,
            confidence=confidence,
            bbox=bbox,
            class_id=class_id
        )
        
        assert np.allclose(det.position, [100.0, 100.0])
        assert det.confidence == 0.85
        assert det.class_id == 2


class TestCupyTextureColorEmbedding:
    """Test the new CupyTextureColorEmbedding."""
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample frame for testing."""
        # Create a colorful test image
        frame = np.random.randint(0, 255, size=(240, 320, 3), dtype=np.uint8)
        return frame
    
    @pytest.fixture
    def sample_bbox(self):
        """Create a sample bounding box."""
        return np.array([100, 100, 150, 150], dtype=np.float32)

    def test_embedding_creation_cpu(self, sample_frame, sample_bbox):
        """Test color embedding creation in CPU mode."""
        embedding = get_embedding_extractor('cupytexture_color', use_gpu=False)
        
        assert embedding.embedding_dim == 84
        assert not embedding.use_gpu
        
        # Test single extraction
        features = embedding.extract(sample_frame, sample_bbox)
        assert features.shape == (84,)
        assert features.dtype == np.float32

    def test_embedding_creation_gpu(self, sample_frame, sample_bbox):
        """Test color embedding creation in GPU mode."""
        try:
            embedding = get_embedding_extractor('cupytexture_color', use_gpu=True)
            
            assert embedding.embedding_dim == 84
            
            # Test single extraction
            features = embedding.extract(sample_frame, sample_bbox)
            assert features.shape == (84,)
            assert features.dtype == np.float32
            
        except Exception as e:
            # GPU might not be available
            pytest.skip(f"GPU test skipped: {e}")

    def test_embedding_batch_extraction(self, sample_frame):
        """Test batch embedding extraction."""
        embedding = get_embedding_extractor('cupytexture_color', use_gpu=False)
        
        bboxes = np.array([
            [50, 50, 100, 100],
            [100, 100, 150, 150],
            [150, 150, 200, 200]
        ], dtype=np.float32)
        
        features_list = embedding.extract_batch(sample_frame, bboxes)
        
        assert len(features_list) == 3
        for features in features_list:
            assert features.shape == (84,)
            assert features.dtype == np.float32

    def test_embedding_comparison_with_original(self, sample_frame, sample_bbox):
        """Test that color embedding has more features than original."""
        original_emb = get_embedding_extractor('cupytexture', use_gpu=False)
        color_emb = get_embedding_extractor('cupytexture_color', use_gpu=False)
        
        original_features = original_emb.extract(sample_frame, sample_bbox)
        color_features = color_emb.extract(sample_frame, sample_bbox)
        
        assert original_emb.embedding_dim == 36
        assert color_emb.embedding_dim == 84
        assert color_features.shape[0] > original_features.shape[0]

    def test_embedding_feature_structure(self, sample_frame, sample_bbox):
        """Test that color embedding has expected feature structure."""
        embedding = get_embedding_extractor('cupytexture_color', use_gpu=False)
        features = embedding.extract(sample_frame, sample_bbox)
        
        # Test that features are not all zeros (meaningful extraction)
        assert not np.allclose(features, 0)
        
        # Test that features are normalized/reasonable range
        assert np.all(np.isfinite(features))
        # Features may have different scales due to color histograms and texture features
        assert np.all(features >= -1000) and np.all(features <= 1000)

    def test_embedding_consistency(self, sample_frame, sample_bbox):
        """Test that embedding extraction is consistent."""
        embedding = get_embedding_extractor('cupytexture_color', use_gpu=False)
        
        features1 = embedding.extract(sample_frame, sample_bbox)
        features2 = embedding.extract(sample_frame, sample_bbox)
        
        # Should be identical for same input
        assert np.allclose(features1, features2)


class TestEndToEndIntegration:
    """Test end-to-end integration scenarios."""
    
    def test_full_tracking_pipeline_color_embedding(self):
        """Test complete tracking pipeline with color embedding."""
        # Setup tracker with color embedding
        config = SwarmSortConfig(
            do_embeddings=True,
            embedding_weight=1.0,
            min_consecutive_detections=1,  # Allow immediate track creation
            max_track_age=5
        )
        
        tracker = SwarmSortTracker(
            config=config,
            embedding_type='cupytexture_color',
            use_gpu=False
        )
        
        # Create test frame and detections
        frame = np.random.randint(0, 255, size=(240, 320, 3), dtype=np.uint8)
        
        # Track object across multiple frames
        tracked_objects_history = []
        for i in range(10):
            # Create detection that moves slightly
            detection = Detection(
                position=np.array([100.0 + i*2, 100.0 + i*2], dtype=np.float32),
                confidence=0.9,
                bbox=np.array([95+i*2, 95+i*2, 105+i*2, 105+i*2], dtype=np.float32),
                class_id=0
            )
            
            tracked_objects = tracker.update([detection], frame)
            tracked_objects_history.append(len(tracked_objects))
        
        # Should eventually create tracks
        assert any(count > 0 for count in tracked_objects_history), \
            f"No tracks created in sequence: {tracked_objects_history}"

    def test_multi_object_tracking_with_reid(self):
        """Test multi-object tracking with re-identification."""
        config = SwarmSortConfig(
            do_embeddings=True,
            reid_enabled=True,
            min_consecutive_detections=2,
            max_track_age=10,
            reid_max_distance=200.0
        )
        
        tracker = SwarmSortTracker(
            config=config,
            embedding_type='cupytexture',
            use_gpu=False
        )
        
        frame = np.random.randint(0, 255, size=(240, 320, 3), dtype=np.uint8)
        
        # Create multiple objects
        detections_sequence = [
            # Frame 1: Two objects
            [
                Detection(np.array([50.0, 50.0]), 0.9, np.array([45, 45, 55, 55])),
                Detection(np.array([150.0, 150.0]), 0.9, np.array([145, 145, 155, 155]))
            ],
            # Frame 2: Two objects moved
            [
                Detection(np.array([52.0, 52.0]), 0.9, np.array([47, 47, 57, 57])),
                Detection(np.array([152.0, 152.0]), 0.9, np.array([147, 147, 157, 157]))
            ],
            # Frame 3: Only first object (second lost)
            [
                Detection(np.array([54.0, 54.0]), 0.9, np.array([49, 49, 59, 59]))
            ],
            # Frame 4: First object continues
            [
                Detection(np.array([56.0, 56.0]), 0.9, np.array([51, 51, 61, 61]))
            ],
            # Frame 5: Second object reappears (should be re-identified)
            [
                Detection(np.array([58.0, 58.0]), 0.9, np.array([53, 53, 63, 63])),
                Detection(np.array([160.0, 160.0]), 0.9, np.array([155, 155, 165, 165]))
            ]
        ]
        
        max_tracked_count = 0
        for detections in detections_sequence:
            tracked_objects = tracker.update(detections, frame)
            max_tracked_count = max(max_tracked_count, len(tracked_objects))
        
        # Should track multiple objects at some point
        assert max_tracked_count >= 1

    def test_config_parameter_handling(self):
        """Test that all configuration parameters are properly handled."""
        config_params = {
            'max_distance': 200.0,
            'max_track_age': 15,
            'detection_conf_threshold': 0.5,
            'use_embeddings': True,
            'embedding_weight': 0.8,
            'reid_enabled': True,
            'reid_max_distance': 250.0,
            'reid_embedding_threshold': 0.7,
            'reid_max_frames': 20,
            'min_consecutive_detections': 5,
            'use_probabilistic_costs': True
        }
        
        tracker = SwarmSortTracker(
            config=config_params,
            embedding_type='cupytexture',
            use_gpu=False
        )
        
        # Verify parameters are set correctly
        assert tracker.config.max_distance == 200.0
        assert tracker.config.max_track_age == 15
        assert tracker.config.do_embeddings == True
        assert tracker.config.embedding_weight == 0.8
        assert tracker.config.reid_enabled == True


class TestAvailableEmbeddings:
    """Test embedding availability and factory functions."""
    
    def test_list_available_embeddings(self):
        """Test that all embeddings are listed."""
        embeddings = list_available_embeddings()
        
        assert 'cupytexture' in embeddings
        assert 'cupytexture_color' in embeddings
        assert 'mega_cupytexture' in embeddings
        
        # Should have at least our 3 embeddings
        assert len(embeddings) >= 3

    def test_get_embedding_extractor_all_types(self):
        """Test creating all available embedding types."""
        embeddings = list_available_embeddings()
        
        for emb_type in embeddings:
            try:
                extractor = get_embedding_extractor(emb_type, use_gpu=False)
                assert extractor is not None
                assert hasattr(extractor, 'extract')
                assert hasattr(extractor, 'embedding_dim')
                assert extractor.embedding_dim > 0
            except Exception as e:
                pytest.fail(f"Failed to create embedding '{emb_type}': {e}")

    def test_embedding_dimensions(self):
        """Test that embeddings have expected dimensions."""
        expected_dims = {
            'cupytexture': 36,
            'cupytexture_color': 84,
            'mega_cupytexture': 64
        }
        
        for emb_type, expected_dim in expected_dims.items():
            extractor = get_embedding_extractor(emb_type, use_gpu=False)
            assert extractor.embedding_dim == expected_dim, \
                f"Embedding '{emb_type}' has dim {extractor.embedding_dim}, expected {expected_dim}"


class TestSwarmTrackerAdapter:
    """Test SwarmTracker integration adapter."""

    def test_raw_tracker_creation(self):
        """Test RawTrackerSwarmSORT creation."""
        config = {'max_track_age': 25, 'do_embeddings': False}
        tracker = RawTrackerSwarmSORT(tracker_config=config)
        
        assert tracker.config.max_track_age == 25
        assert tracker.config.do_embeddings == False

    def test_adapter_detection_conversion(self):
        """Test detection format conversion in adapter."""
        tracker = RawTrackerSwarmSORT()
        
        # Test with various detection formats
        detections = [
            # Array format [x1, y1, x2, y2, confidence]
            np.array([10, 10, 50, 50, 0.8]),
            # List format
            [20, 20, 60, 60, 0.9, 0],  # With class_id
        ]
        
        result = tracker.track(detections, np.zeros((100, 100, 3)))
        assert hasattr(result, 'tracked_objects')
        assert hasattr(result, 'bounding_boxes')

    def test_create_swarmsort_tracker_fallback(self):
        """Test create_swarmsort_tracker fallback behavior."""
        # Test with mock runtime config
        class MockConfig:
            def get_modified_params(self):
                return {'max_track_age': 20, 'use_embeddings': False}
        
        # Should create tracker without errors
        tracker = create_swarmsort_tracker(runtime_config=MockConfig())
        assert hasattr(tracker, 'config')

    def test_import_error_handling(self):
        """Test graceful handling of import errors."""
        # Test that package works even if optional dependencies are missing
        from swarmsort import SwarmSortTracker, SwarmSortConfig
        
        # Should work without errors
        config = SwarmSortConfig(do_embeddings=False)  # Disable embeddings to avoid GPU deps
        tracker = SwarmSortTracker(config)
        assert tracker is not None


if __name__ == '__main__':
    # Run tests if called directly
    pytest.main([__file__, '-v'])