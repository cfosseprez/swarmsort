# SwarmSort Performance Profiling

This directory contains profiling tools for analyzing SwarmSort performance and identifying bottlenecks.

## Tools

### 1. `profile_swarmsort.py` - Comprehensive Profiler
Full-featured profiler using cProfile with snakeviz visualization.

**Features:**
- Profiles basic and uncertainty-enabled configurations
- Generates detailed call graphs and timing reports
- Creates interactive visualizations with snakeviz
- Comparative analysis between configurations

**Usage:**
```bash
# Install snakeviz for visualization
pip install snakeviz

# Run comprehensive profiling
python profiling/profile_swarmsort.py
```

**Output:**
- `swarmsort_basic.prof` - Profile data for basic config
- `swarmsort_uncertainty.prof` - Profile data with uncertainty
- Text reports with detailed function statistics
- Interactive snakeviz visualizations

### 2. `quick_profile.py` - Lightweight Profiler
Fast profiling tool for quick bottleneck identification.

**Features:**
- Lightweight timing analysis
- Component breakdown testing
- No external dependencies
- Quick performance overview

**Usage:**
```bash
python profiling/quick_profile.py
```

**Output:**
- Console-based performance report
- Frame-by-frame timing analysis
- Component-level benchmarks

## Profiling Scenarios

Both tools test realistic scenarios:
- **110+ objects** (typical high-density scenario)
- **20 frames** of tracking
- **Varying detection counts** (±10 objects per frame)
- **Realistic spatial distribution** (grid with noise)
- **128-dimensional embeddings** (standard ReID embedding size)

## Key Metrics

The profilers analyze:
- **Total execution time** per frame
- **Cost matrix computation** time
- **Assignment algorithm** performance
- **Embedding processing** overhead
- **Track management** efficiency

## Performance Targets

Target performance for 110 objects:
- **Basic tracking**: <10ms per frame
- **With uncertainty**: <15ms per frame
- **Cost matrix**: <5ms
- **Assignment**: <3ms

## Interpreting Results

### Snakeviz Visualization
1. **Icicle diagram** shows call hierarchy and time distribution
2. **Sunburst chart** provides interactive exploration
3. **Function list** shows detailed statistics

### Common Bottlenecks
1. **Cost matrix computation** - Should be JIT-compiled with Numba
2. **Embedding similarity** - Vectorized operations preferred
3. **Assignment algorithms** - Hungarian vs. greedy trade-offs
4. **Memory allocation** - Array creation/copying overhead

## Optimization Tips

1. **Enable Numba JIT** for hot path functions
2. **Vectorize operations** instead of loops
3. **Cache expensive computations** (embeddings, distances)
4. **Use appropriate data types** (float32 vs float64)
5. **Minimize memory allocation** in tight loops

## Troubleshooting

### Performance Regression
If performance degrades:
1. Run `quick_profile.py` to identify slow components
2. Compare with previous profile data
3. Check for missing `@nb.njit` decorators
4. Verify vectorized operations are working

### Memory Issues
For large scenarios:
1. Monitor memory usage during profiling
2. Check for memory leaks in long runs
3. Profile garbage collection overhead

## Advanced Usage

### Custom Scenarios
Modify the test scenarios in either script:
```python
def create_custom_scenario(num_objects, complexity):
    # Custom detection patterns
    # Specific movement patterns  
    # Realistic occlusions/splits
```

### Integration with CI/CD
```bash
# Performance regression testing
python profiling/quick_profile.py > performance_baseline.txt
# Compare against baseline in CI
```