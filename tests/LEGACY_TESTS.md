# Test Files Documentation

This document describes the test structure for SwarmSort.

## Active Test Files

The following tests cover current functionality and are actively maintained:

| File | Purpose | Test Classes |
|------|---------|--------------|
| `test_assignment.py` | Assignment algorithms (greedy, Hungarian, hybrid) | Core matching tests |
| `test_config.py` | Configuration validation and defaults | `TestConfigurationSystem`, `TestKalmanParameterValidation`, `TestConfigDistanceRelationships`, `TestValidateConfigFunction`, `TestConfigDefaults` |
| `test_embeddings.py` | Embedding extraction and distance computation | Embedding tests |
| `test_integration.py` | End-to-end tracking scenarios | `TestBasicTracking`, `TestEmbeddingBasedTracking`, `TestReIdentification`, `TestInputValidation`, `TestPerformanceAndStability` |
| `test_kalman_filters.py` | Kalman filter predict/update cycles | `TestSimpleKalmanFilter`, `TestConfigurableKalmanParameters`, `TestKalmanFilterValidation_Partial` |
| `test_track_state.py` | Track state management and lifecycle | `TestFastTrackStateInitialization`, `TestFastTrackStateUpdates_Partial`, `TestPendingDetection` |
| `test_swarmtracker_integration.py` | SwarmTracker pipeline integration | SwarmTracker adapter tests |
| `test_tracker_comprehensive.py` | Comprehensive tracker validation | `TestBasicTracking`, `TestEmbeddingTracking`, `TestAdvancedFeatures`, `TestConfigurationModes` |
| `conftest.py` | Pytest fixtures and shared test utilities | - |

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_assignment.py -v

# Run with coverage
pytest tests/ --cov=swarmsort --cov-report=html
```

## Historical Context

During SwarmSort development, the API evolved through several iterations:

1. **Assignment module refactoring** - Functions were renamed (e.g., `greedy_assignment` -> `numba_greedy_assignment`)
2. **Track state redesign** - Attributes like `lost_tracks` were removed
3. **Embedding API changes** - `set_embedding_params()` signature changed
4. **Kalman filter simplification** - OC-SORT implementation was refactored

Legacy test files that were no longer compatible with the current API have been removed. The current test suite provides comprehensive coverage of the active functionality.

## Contributing New Tests

When adding new tests:
1. Place in this `tests/` directory
2. Use `test_*.py` naming convention
3. Import from `swarmsort` package (not relative imports)
4. Use fixtures from `conftest.py` where applicable
