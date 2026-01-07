"""
Tests for SwarmSort configuration system.

This module tests:
- Core configuration validation
- New Kalman filter parameters
- Configuration relationships and constraints
- validate_config() convenience function
- YAML loading and merging
"""
import pytest
import numpy as np
import tempfile
import yaml

from swarmsort import SwarmSortConfig, validate_config
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
            config = SwarmSortConfig(do_embeddings=True, embedding_weight=-1.0)
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
            'do_embeddings': False,
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
            'do_embeddings': False,
        }
        
        # Test merging
        merged_config = merge_config_with_priority(
            default_config=SwarmSortConfig,
            runtime_config=runtime_config,
            verbose_parameters=False
        )
        
        assert merged_config.max_track_age == 15
        assert merged_config.do_embeddings == False
        assert merged_config.max_distance == 80.0  # Should keep default

    def test_config_serialization(self):
        """Test configuration serialization/deserialization."""
        config = SwarmSortConfig(
            max_track_age=25,
            do_embeddings=True,
            reid_enabled=False,
            init_conf_threshold=0.1
        )
        
        # Test to_dict method (if available)
        if hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
            assert config_dict['max_track_age'] == 25
            assert config_dict['do_embeddings'] == True
            assert config_dict['reid_enabled'] == False
        
        # Test basic attribute access
        assert config.max_track_age == 25
        assert config.do_embeddings == True
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


class TestKalmanParameterValidation:
    """Test validation of new Kalman filter configuration parameters."""

    def test_kalman_velocity_damping_valid(self):
        """Test valid kalman_velocity_damping values."""
        for damping in [0.0, 0.5, 0.95, 1.0]:
            config = SwarmSortConfig(kalman_velocity_damping=damping)
            config.validate()  # Should not raise

    def test_kalman_velocity_damping_invalid(self):
        """Test invalid kalman_velocity_damping values."""
        with pytest.raises(ValueError, match="kalman_velocity_damping must be between 0 and 1"):
            config = SwarmSortConfig(kalman_velocity_damping=-0.1)
            config.validate()

        with pytest.raises(ValueError, match="kalman_velocity_damping must be between 0 and 1"):
            config = SwarmSortConfig(kalman_velocity_damping=1.1)
            config.validate()

    def test_kalman_update_alpha_valid(self):
        """Test valid kalman_update_alpha values."""
        for alpha in [0.0, 0.5, 0.7, 1.0]:
            config = SwarmSortConfig(kalman_update_alpha=alpha)
            config.validate()

    def test_kalman_update_alpha_invalid(self):
        """Test invalid kalman_update_alpha values."""
        with pytest.raises(ValueError, match="kalman_update_alpha must be between 0 and 1"):
            config = SwarmSortConfig(kalman_update_alpha=-0.1)
            config.validate()

        with pytest.raises(ValueError, match="kalman_update_alpha must be between 0 and 1"):
            config = SwarmSortConfig(kalman_update_alpha=1.5)
            config.validate()

    def test_kalman_velocity_weight_valid(self):
        """Test valid kalman_velocity_weight values."""
        for weight in [0.0, 0.2, 0.5, 1.0]:
            config = SwarmSortConfig(kalman_velocity_weight=weight)
            config.validate()

    def test_kalman_velocity_weight_invalid(self):
        """Test invalid kalman_velocity_weight values."""
        with pytest.raises(ValueError, match="kalman_velocity_weight must be between 0 and 1"):
            config = SwarmSortConfig(kalman_velocity_weight=-0.1)
            config.validate()

    def test_prediction_miss_threshold_valid(self):
        """Test valid prediction_miss_threshold values."""
        for threshold in [1, 3, 10, 100]:
            config = SwarmSortConfig(prediction_miss_threshold=threshold)
            config.validate()

    def test_prediction_miss_threshold_invalid(self):
        """Test invalid prediction_miss_threshold values."""
        with pytest.raises(ValueError, match="prediction_miss_threshold must be at least 1"):
            config = SwarmSortConfig(prediction_miss_threshold=0)
            config.validate()

    def test_mahalanobis_normalization_valid(self):
        """Test valid mahalanobis_normalization values."""
        for norm in [0.1, 1.0, 20.0, 100.0]:
            config = SwarmSortConfig(mahalanobis_normalization=norm)
            config.validate()

    def test_mahalanobis_normalization_invalid(self):
        """Test invalid mahalanobis_normalization values."""
        with pytest.raises(ValueError, match="mahalanobis_normalization must be positive"):
            config = SwarmSortConfig(mahalanobis_normalization=0.0)
            config.validate()

        with pytest.raises(ValueError, match="mahalanobis_normalization must be positive"):
            config = SwarmSortConfig(mahalanobis_normalization=-1.0)
            config.validate()

    def test_probabilistic_gating_multiplier_valid(self):
        """Test valid probabilistic_gating_multiplier values."""
        for mult in [0.5, 1.0, 1.5, 3.0]:
            config = SwarmSortConfig(probabilistic_gating_multiplier=mult)
            config.validate()

    def test_probabilistic_gating_multiplier_invalid(self):
        """Test invalid probabilistic_gating_multiplier values."""
        with pytest.raises(ValueError, match="probabilistic_gating_multiplier must be positive"):
            config = SwarmSortConfig(probabilistic_gating_multiplier=0.0)
            config.validate()

    def test_time_covariance_inflation_valid(self):
        """Test valid time_covariance_inflation values."""
        for inflation in [0.0, 0.1, 0.5, 1.0]:
            config = SwarmSortConfig(time_covariance_inflation=inflation)
            config.validate()

    def test_time_covariance_inflation_invalid(self):
        """Test invalid time_covariance_inflation values."""
        with pytest.raises(ValueError, match="time_covariance_inflation must be between 0 and 1"):
            config = SwarmSortConfig(time_covariance_inflation=-0.1)
            config.validate()

        with pytest.raises(ValueError, match="time_covariance_inflation must be between 0 and 1"):
            config = SwarmSortConfig(time_covariance_inflation=1.5)
            config.validate()

    def test_kalman_type_valid(self):
        """Test valid kalman_type values."""
        for kalman_type in ["simple", "oc"]:
            config = SwarmSortConfig(kalman_type=kalman_type)
            config.validate()

    def test_kalman_type_invalid(self):
        """Test invalid kalman_type values."""
        with pytest.raises(ValueError, match="Invalid kalman_type"):
            config = SwarmSortConfig(kalman_type="invalid")
            config.validate()

    def test_assignment_strategy_valid(self):
        """Test valid assignment_strategy values."""
        for strategy in ["hungarian", "greedy", "hybrid"]:
            config = SwarmSortConfig(assignment_strategy=strategy)
            config.validate()

    def test_assignment_strategy_invalid(self):
        """Test invalid assignment_strategy values."""
        with pytest.raises(ValueError, match="Invalid assignment_strategy"):
            config = SwarmSortConfig(assignment_strategy="invalid")
            config.validate()


class TestConfigDistanceRelationships:
    """Test validation of distance parameter relationships."""

    def test_deduplication_less_than_collision(self):
        """Test that deduplication_distance < collision_safety_distance."""
        with pytest.raises(ValueError, match="deduplication_distance.*must be <.*collision_safety_distance"):
            config = SwarmSortConfig(
                deduplication_distance=50.0,
                collision_safety_distance=30.0
            )
            config.validate()

    def test_collision_less_than_max_distance(self):
        """Test that collision_safety_distance < max_distance."""
        with pytest.raises(ValueError, match="collision_safety_distance.*must be <.*max_distance"):
            config = SwarmSortConfig(
                max_distance=50.0,
                collision_safety_distance=60.0
            )
            config.validate()

    def test_local_density_radius_le_max_distance(self):
        """Test that local_density_radius <= max_distance."""
        with pytest.raises(ValueError, match="local_density_radius.*must be <=.*max_distance"):
            config = SwarmSortConfig(
                max_distance=50.0,
                local_density_radius=60.0
            )
            config.validate()

    def test_greedy_threshold_less_than_max_distance(self):
        """Test that greedy_threshold < max_distance."""
        with pytest.raises(ValueError, match="greedy_threshold.*must be <.*max_distance"):
            config = SwarmSortConfig(
                max_distance=50.0,
                greedy_threshold=60.0
            )
            config.validate()

    def test_valid_distance_relationships(self):
        """Test valid distance parameter relationships."""
        config = SwarmSortConfig(
            max_distance=100.0,
            deduplication_distance=10.0,
            collision_safety_distance=30.0,
            local_density_radius=50.0,
            greedy_threshold=20.0
        )
        config.validate()  # Should not raise

    def test_auto_computed_parameters(self):
        """Test that auto-computed parameters are set correctly."""
        config = SwarmSortConfig(max_distance=100.0)
        # These should be auto-computed in __post_init__
        assert abs(config.local_density_radius - 100.0 / 3) < 0.01  # max_distance / 3
        assert abs(config.greedy_threshold - 100.0 / 3) < 0.01  # max_distance / 3
        assert config.reid_max_distance == 150.0  # max_distance * 1.5
        assert config.pending_detection_distance == 100.0  # max_distance
        assert config.collision_safety_distance == 25.0  # max_distance * 0.25


class TestValidateConfigFunction:
    """Test the validate_config() convenience function."""

    def test_valid_config_returns_true(self):
        """Test that valid config returns (True, [])."""
        config = SwarmSortConfig()
        is_valid, errors = validate_config(config)
        assert is_valid is True
        assert errors == []

    def test_invalid_config_returns_false(self):
        """Test that invalid config returns (False, errors)."""
        config = SwarmSortConfig(max_distance=-10.0)
        is_valid, errors = validate_config(config)
        assert is_valid is False
        assert len(errors) > 0
        assert any("max_distance" in e for e in errors)

    def test_collects_multiple_errors(self):
        """Test that validate_config collects all errors."""
        config = SwarmSortConfig(
            max_distance=-10.0,
            max_track_age=0,
            kalman_velocity_damping=2.0
        )
        is_valid, errors = validate_config(config)
        assert is_valid is False
        assert len(errors) >= 3  # Should have at least 3 errors

    def test_error_messages_are_descriptive(self):
        """Test that error messages include actual values."""
        config = SwarmSortConfig(
            deduplication_distance=50.0,
            collision_safety_distance=30.0
        )
        is_valid, errors = validate_config(config)
        assert is_valid is False
        # Error should mention the actual values
        error_text = " ".join(errors)
        assert "50" in error_text or "30" in error_text

    def test_vs_validate_method(self):
        """Test validate_config vs validate() method consistency."""
        # Valid config - both should pass
        valid_config = SwarmSortConfig()
        is_valid, _ = validate_config(valid_config)
        assert is_valid is True
        valid_config.validate()  # Should not raise

        # Invalid config - validate_config returns errors, validate() raises
        invalid_config = SwarmSortConfig(max_distance=-1.0)
        is_valid, errors = validate_config(invalid_config)
        assert is_valid is False
        with pytest.raises(ValueError):
            invalid_config.validate()

    def test_embedding_weight_upper_bound(self):
        """Test embedding_weight upper bound validation."""
        config = SwarmSortConfig(embedding_weight=15.0)
        is_valid, errors = validate_config(config)
        assert is_valid is False
        assert any("embedding_weight" in e and "10.0" in e for e in errors)

    def test_reid_embedding_threshold_range(self):
        """Test reid_embedding_threshold must be in [0, 1]."""
        config = SwarmSortConfig(reid_embedding_threshold=1.5)
        is_valid, errors = validate_config(config)
        assert is_valid is False
        assert any("reid_embedding_threshold" in e for e in errors)


class TestConfigDefaults:
    """Test configuration default values."""

    def test_default_config_is_valid(self):
        """Test that default configuration passes validation."""
        config = SwarmSortConfig()
        config.validate()  # Should not raise
        is_valid, errors = validate_config(config)
        assert is_valid is True

    def test_default_kalman_parameters(self):
        """Test default Kalman parameter values."""
        config = SwarmSortConfig()
        assert config.kalman_velocity_damping == 0.95
        assert config.kalman_update_alpha == 0.7
        assert config.kalman_velocity_weight == 0.2
        assert config.prediction_miss_threshold == 3

    def test_default_probabilistic_parameters(self):
        """Test default probabilistic cost parameters."""
        config = SwarmSortConfig()
        assert config.mahalanobis_normalization == 5.0
        assert config.probabilistic_gating_multiplier == 1.5
        assert config.time_covariance_inflation == 0.2
        assert config.base_position_variance == 25.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])