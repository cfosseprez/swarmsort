"""
Tests for SwarmSort configuration system.
"""
import pytest
import numpy as np
import tempfile
import yaml

from swarmsort import SwarmSortConfig
from swarmsort.config import merge_config_with_priority, load_local_yaml_config


class TestConfigurationSystem:
    """Test configuration loading and validation."""

    def test_config_validation_errors(self):
        """Test configuration validation with invalid values."""
        with pytest.raises(ValueError, match="max_track_age must be at least 1"):
            config = SwarmSortConfig(max_track_age=0)
            config.validate()
            
        with pytest.raises(ValueError, match="init_conf_threshold must be between 0 and 1"):
            config = SwarmSortConfig(init_conf_threshold=2.0)
            config.validate()
            
        with pytest.raises(ValueError, match="detection_conf_threshold must be between 0 and 1"):
            config = SwarmSortConfig(detection_conf_threshold=-0.1)
            config.validate()
            
        with pytest.raises(ValueError, match="embedding_weight must be non-negative"):
            config = SwarmSortConfig(use_embeddings=True, embedding_weight=-1.0)
            config.validate()
            
        with pytest.raises(ValueError, match="min_consecutive_detections must be at least 1"):
            config = SwarmSortConfig(min_consecutive_detections=0)
            config.validate()

    def test_yaml_config_loading(self):
        """Test YAML configuration loading functionality."""
        # Create a temporary YAML file
        config_data = {
            'max_track_age': 25,
            'init_conf_threshold': 0.1,
            'use_embeddings': False,
            'reid_enabled': False,
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            yaml_path = f.name
        
        try:
            # Test loading YAML config
            loaded_config = load_local_yaml_config('test_config.yaml', caller_file=yaml_path)
            # Should return empty dict since file doesn't exist in expected locations
            assert isinstance(loaded_config, dict)
            
        finally:
            import os
            os.unlink(yaml_path)

    def test_config_merging(self):
        """Test configuration merging with priority."""
        runtime_config = {
            'max_track_age': 15,
            'use_embeddings': False,
        }
        
        # Test merging
        merged_config = merge_config_with_priority(
            default_config=SwarmSortConfig,
            runtime_config=runtime_config,
            verbose_parameters=False
        )
        
        assert merged_config.max_track_age == 15
        assert merged_config.use_embeddings == False
        assert merged_config.max_distance == 80.0  # Should keep default

    def test_config_serialization(self):
        """Test configuration serialization/deserialization."""
        config = SwarmSortConfig(
            max_track_age=25,
            use_embeddings=True,
            reid_enabled=False,
            init_conf_threshold=0.1
        )
        
        # Test to_dict method (if available)
        if hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
            assert config_dict['max_track_age'] == 25
            assert config_dict['use_embeddings'] == True
            assert config_dict['reid_enabled'] == False
        
        # Test basic attribute access
        assert config.max_track_age == 25
        assert config.use_embeddings == True
        assert config.reid_enabled == False
        assert config.init_conf_threshold == 0.1

    def test_version_info(self):
        """Test package version information."""
        import swarmsort
        assert hasattr(swarmsort, '__version__')
        assert isinstance(swarmsort.__version__, str)

    def test_package_imports(self):
        """Test that all main package imports work."""
        from swarmsort import (
            SwarmSortTracker, 
            SwarmSortConfig, 
            Detection, 
            TrackedObject
        )
        
        # All should be importable
        assert SwarmSortTracker is not None
        assert SwarmSortConfig is not None
        assert Detection is not None
        assert TrackedObject is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])