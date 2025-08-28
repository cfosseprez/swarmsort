# SwarmSort Testing Guide

This document provides comprehensive information about testing the SwarmSort standalone package.

## Test Suite Overview

The SwarmSort test suite consists of **200+ tests** organized into 6 main categories:

### 1. **Unit Tests** (`test_basic.py`, `test_core.py`)
- **31 core unit tests**
- Tests individual components and functions
- Fast execution (< 2 seconds)
- Core Numba JIT functions
- Data classes and configuration
- Error handling

### 2. **Integration Tests** (`test_integration.py`) 
- **24 integration tests**
- End-to-end tracking scenarios
- Multi-object tracking
- Configuration integration
- Factory function testing

### 3. **Performance Tests** (`test_performance.py`)
- **15 performance benchmarks**
- Numba function performance
- Scalability testing
- Memory usage validation
- Different problem sizes

### 4. **Stress & Edge Case Tests** (`test_stress_edge_cases.py`)
- **25 stress tests**
- Edge case inputs (NaN, infinity, extreme values)
- High load scenarios (100+ objects)
- Memory pressure testing
- Error recovery

### 5. **Embedding-Specific Tests** (`test_embedding_specific.py`)
- **35 embedding tests**
- Distance scaling algorithms
- Embedding-based tracking
- Multi-class scenarios
- Re-identification testing

### 6. **Integration & Long-Running Tests**
- Comprehensive stress scenarios
- Long sequence stability
- Memory leak detection
- Configuration consistency

## Test Categories by Markers

Tests are organized using pytest markers:

```bash
# Unit tests only
pytest -m unit

# Integration tests
pytest -m integration

# Performance benchmarks  
pytest -m performance

# Stress tests
pytest -m stress

# Embedding-specific tests
pytest -m embedding

# Slow tests (>5 seconds)
pytest -m slow

# Benchmark tests with timing
pytest -m benchmark
```

## Running Tests

### Quick Start

```bash
# Install dependencies
poetry install --with dev

# Run all tests
pytest

# Run with coverage
pytest --cov=swarmsort --cov-report=html
```

### Using the Test Runner

The `run_tests.py` script provides convenient test execution:

```bash
# Run all tests
python run_tests.py --mode all

# Quick test run (core tests only)
python run_tests.py --mode all --quick

# Run specific test categories
python run_tests.py --mode unit
python run_tests.py --mode integration
python run_tests.py --mode performance
python run_tests.py --mode embedding

# CI mode (optimized for continuous integration)
python run_tests.py --mode ci

# Code quality checks
python run_tests.py --mode lint
```

### GitHub Actions CI/CD

The `.github/workflows/test.yml` provides comprehensive automated testing:

- **Multi-platform**: Ubuntu, Windows, macOS
- **Multi-Python**: 3.9, 3.10, 3.11, 3.12
- **Parallel execution**: Multiple test jobs
- **Code quality**: Linting, formatting, type checking
- **Security**: Dependency scanning, security analysis
- **Performance**: Benchmark tracking
- **Coverage**: Code coverage reporting

## Test Configuration

### pytest.ini
Configured with:
- Strict markers and configuration
- Coverage reporting (HTML, XML, terminal)
- 80% coverage threshold
- Proper test discovery patterns

### Coverage Goals
- **Target**: 80%+ code coverage
- **Reports**: HTML (`htmlcov/`), XML (`coverage.xml`), terminal
- **Exclusions**: Test files, `__pycache__`, virtual environments

## Performance Benchmarks

### Benchmark Categories

1. **Numba Function Performance**
   - Cosine similarity computation
   - Embedding distance calculations
   - Cost matrix operations

2. **Tracker Performance** 
   - Single frame updates
   - Scalability with detection count
   - Long sequence processing

3. **Memory Benchmarks**
   - Memory leak detection
   - Large embedding handling
   - Track history management

4. **Configuration Impact**
   - Embedding vs no-embedding performance
   - Different matching methods
   - History length impact

### Running Benchmarks

```bash
# Run all performance tests with benchmarking
pytest tests/test_performance.py -m benchmark --benchmark-only

# Compare different configurations
pytest tests/test_performance.py::test_embedding_vs_no_embedding_performance

# Memory profiling
pytest tests/test_performance.py::TestMemoryBenchmarks
```

## Stress Testing Scenarios

### Edge Case Testing

1. **Input Validation**
   - Empty detection sequences
   - Invalid coordinates (NaN, infinity)
   - Extreme confidence values
   - Malformed bounding boxes

2. **High Load Scenarios**
   - 100+ simultaneous objects
   - Very long sequences (500+ frames)
   - Dense object clusters
   - Rapid appearance/disappearance

3. **Resource Constraints**
   - Memory pressure testing
   - Extreme configuration values
   - Large embedding dimensions
   - Numerical instability scenarios

### Comprehensive Stress Test

The `test_comprehensive_stress_scenario` combines:
- 25 objects with different behaviors
- 100 frames of tracking
- Multiple motion patterns
- Noise and occlusions
- Cross-platform validation

## Embedding Testing

### Embedding Scaler Tests

1. **Algorithm Validation**
   - 11 different scaling methods
   - Numerical stability testing
   - Online adaptation validation
   - Edge case handling

2. **Integration Testing**
   - Embedding-based association
   - Multi-class tracking
   - Re-identification scenarios
   - Distance computation accuracy

### Scaling Methods Tested

- `robust_minmax`: Percentile-based robust scaling
- `min_robustmax`: Asymmetric robust scaling  
- `zscore`: Standard z-score normalization
- `robust_zscore`: Median-MAD based scaling
- `arcsinh`: Inverse hyperbolic sine transformation
- `beta`: Beta distribution CDF mapping
- `quantile`: Empirical CDF mapping
- And more...

## Test Data & Fixtures

### Provided Fixtures

- `default_config`: Standard configuration for testing
- `embedding_config`: Optimized for embedding tests
- `sample_detections`: Pre-generated detection sequences
- `consistent_embeddings`: Reproducible embedding templates
- `tracking_scenario`: Configurable multi-object scenarios
- `benchmark_data`: Large-scale performance test data

### Synthetic Data Generation

Tests use reproducible synthetic data:
- Configurable number of objects and frames
- Different motion patterns (linear, circular, random)
- Realistic noise and occlusion simulation
- Multi-class scenarios with distinct embeddings

## Continuous Integration

### GitHub Actions Workflow

The CI pipeline includes:

1. **Code Quality Jobs**
   - Black formatting
   - Flake8 linting  
   - MyPy type checking
   - Import sorting (isort)

2. **Test Matrix**
   - 3 operating systems
   - 4 Python versions
   - Parallel test execution
   - Coverage reporting

3. **Security & Performance**
   - Dependency security scanning
   - Performance regression detection
   - Memory leak validation

4. **Extended Testing**
   - Stress tests on main branch
   - Long-running scenarios
   - Integration with SwarmTracker

### Workflow Triggers

- Push to `main` or `develop` branches
- Pull requests
- Manual dispatch
- Scheduled runs (optional)

## Test Development Guidelines

### Adding New Tests

1. **Choose appropriate category**:
   - Unit tests for individual functions
   - Integration tests for component interaction
   - Performance tests for speed/memory validation
   - Stress tests for edge cases

2. **Use proper markers**:
   ```python
   @pytest.mark.unit
   def test_my_function():
       pass
   
   @pytest.mark.slow
   @pytest.mark.integration
   def test_long_scenario():
       pass
   ```

3. **Follow naming conventions**:
   - `test_*.py` files
   - `Test*` classes  
   - `test_*` methods
   - Descriptive names

4. **Use fixtures appropriately**:
   ```python
   def test_tracker_functionality(basic_tracker, sample_detections):
       result = basic_tracker.update(sample_detections)
       assert isinstance(result, list)
   ```

### Performance Test Guidelines

1. **Use pytest-benchmark** for timing:
   ```python
   def test_function_performance(benchmark):
       result = benchmark(my_function, arg1, arg2)
       assert result is not None
   ```

2. **Include warm-up** for Numba functions:
   ```python
   # Warm up JIT compilation
   for _ in range(3):
       my_numba_function(test_input)
   
   # Then benchmark
   result = benchmark(my_numba_function, test_input)
   ```

3. **Test scalability** with parametrized tests:
   ```python
   @pytest.mark.parametrize("size", [10, 50, 100, 500])
   def test_scaling(benchmark, size):
       data = generate_data(size)
       benchmark(process_data, data)
   ```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure package is installed in development mode
   pip install -e .
   # Or with Poetry
   poetry install
   ```

2. **Missing Dependencies**
   ```bash
   # Install test dependencies
   poetry install --with dev
   ```

3. **Windows Path Issues**
   - Use forward slashes in test paths
   - Be aware of case sensitivity differences

4. **Memory Issues in Stress Tests**
   - Increase timeout for slow tests
   - Monitor system resources during testing
   - Use `--timeout=300` for long-running tests

### Performance Issues

1. **Slow Test Execution**
   - Use `pytest-xdist` for parallel execution:
     ```bash
     pytest -n auto  # Use all CPU cores
     ```
   - Skip slow tests during development:
     ```bash
     pytest -m "not slow"
     ```

2. **Numba Compilation Delays**
   - First run includes JIT compilation overhead
   - Subsequent runs are much faster
   - Consider using `cache=True` in `@nb.njit` decorators

### CI/CD Issues

1. **Platform-Specific Failures**
   - Check OS-specific test conditions
   - Verify file path handling
   - Test timeout adjustments for different platforms

2. **Dependency Conflicts**
   - Keep `pyproject.toml` dependencies up to date
   - Use version ranges for compatibility
   - Test with minimum supported versions

## Test Metrics & Reporting

### Coverage Reports

- **Terminal**: Real-time coverage feedback
- **HTML**: Detailed line-by-line coverage (`htmlcov/index.html`)
- **XML**: Machine-readable format for CI integration

### Performance Tracking

- Benchmark results stored in JSON format
- Historical performance comparison
- Regression detection for critical functions

### Test Execution Statistics

Typical test execution times:
- **Unit tests**: ~5 seconds
- **Integration tests**: ~15 seconds  
- **Performance tests**: ~30 seconds
- **Stress tests**: ~60 seconds
- **All tests**: ~2-3 minutes

## Future Test Enhancements

### Planned Improvements

1. **Visual Testing**
   - Trajectory visualization validation
   - Embedding space visualization
   - Performance dashboard

2. **Property-Based Testing**
   - Hypothesis-based test generation
   - Automatic edge case discovery
   - Invariant validation

3. **Integration Testing**
   - Real video data testing
   - YOLO detector integration
   - End-to-end pipeline validation

4. **Performance Optimization**
   - GPU acceleration testing
   - Distributed processing validation
   - Real-time performance benchmarks

---

This comprehensive test suite ensures SwarmSort's reliability, performance, and correctness across diverse scenarios and platforms. The combination of unit tests, integration tests, performance benchmarks, and stress tests provides confidence in the package's robustness for production use.