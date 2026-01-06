"""
Embedding distance scaler for SwarmSort.

This module provides advanced scaling methods for embedding distances
to improve tracking performance and numerical stability.
"""
import numpy as np
from scipy import stats
from loguru import logger


class EmbeddingDistanceScaler:
    """Enhanced embedding scaler with multiple scaling methods for comparison"""

    def __init__(
        self, method: str = "robust_minmax", update_rate: float = 0.05, min_samples: int = 200, 
        update_interval: int = 3
    ):
        self.method = method
        self.update_rate = update_rate
        self.min_samples = min_samples
        self.update_interval = update_interval
        self.update_counter = 0

        # Running statistics
        self.min_distance = None
        self.max_distance = None
        self.mean_distance = None
        self.std_distance = None
        self.sample_count = 0
        self.p5 = None
        self.p95 = None
        self.p1 = None
        self.p99 = None
        self.p10 = None
        self.p90 = None
        self.median = None
        self.iqr = None

        # For quantile-based methods
        self.q25 = None
        self.q75 = None

    def update_statistics(self, distances: np.ndarray):
        """Update running statistics with new distance samples - OPTIMIZED VERSION"""
        if len(distances) == 0:
            return

        # Only update statistics every N frames for performance (5-50x speedup)
        self.update_counter += 1
        if self.update_counter % self.update_interval != 0:
            return

        # Basic statistics (fast operations)
        current_min = np.min(distances)
        current_max = np.max(distances)
        current_mean = np.mean(distances)
        current_std = np.std(distances)
        
        # PERFORMANCE OPTIMIZATION: Only compute percentiles needed for the active scaling method
        if self.method in ["robust_minmax", "min_robustmax"]:
            # Most common case: only need p5 and p95 (massive speedup!)
            percentiles = np.percentile(distances, [5, 95])
            current_p5 = percentiles[0]
            current_p95 = percentiles[1]
            # Set unused percentiles to None to avoid computation
            current_p1 = current_p10 = current_p90 = current_p99 = None
            current_median = current_q25 = current_q75 = current_iqr = None
        else:
            # Fallback for other methods that might need more percentiles
            percentiles = np.percentile(distances, [1, 5, 10, 25, 50, 75, 90, 95, 99])
            current_p1 = percentiles[0]
            current_p5 = percentiles[1]
            current_p10 = percentiles[2]
            current_q25 = percentiles[3]
            current_median = percentiles[4]
            current_q75 = percentiles[5]
            current_p90 = percentiles[6]
            current_p95 = percentiles[7]
            current_p99 = percentiles[8]
            current_iqr = current_q75 - current_q25

        if self.sample_count == 0:
            # Initialize statistics
            self.min_distance = current_min
            self.max_distance = current_max
            self.mean_distance = current_mean
            self.std_distance = current_std
            self.p5 = current_p5
            self.p95 = current_p95
            # Only initialize percentiles that were computed
            if current_p1 is not None:
                self.p1 = current_p1
                self.p10 = current_p10
                self.p90 = current_p90
                self.p99 = current_p99
                self.median = current_median
                self.q25 = current_q25
                self.q75 = current_q75
                self.iqr = current_iqr
        else:
            # Update statistics with exponential moving average
            alpha = self.update_rate
            self.min_distance = min(self.min_distance, current_min)
            self.max_distance = max(self.max_distance, current_max)
            self.mean_distance = (1 - alpha) * self.mean_distance + alpha * current_mean
            self.std_distance = (1 - alpha) * self.std_distance + alpha * current_std
            self.p5 = (1 - alpha) * self.p5 + alpha * current_p5
            self.p95 = (1 - alpha) * self.p95 + alpha * current_p95
            # Only update percentiles that were computed
            if current_p1 is not None and self.p1 is not None:
                self.p1 = (1 - alpha) * self.p1 + alpha * current_p1
                self.p10 = (1 - alpha) * self.p10 + alpha * current_p10
                self.p90 = (1 - alpha) * self.p90 + alpha * current_p90
                self.p99 = (1 - alpha) * self.p99 + alpha * current_p99
                self.median = (1 - alpha) * self.median + alpha * current_median
                self.q25 = (1 - alpha) * self.q25 + alpha * current_q25
                self.q75 = (1 - alpha) * self.q75 + alpha * current_q75
                self.iqr = (1 - alpha) * self.iqr + alpha * current_iqr

        self.sample_count += len(distances)

    def scale_distances(self, distances: np.ndarray) -> np.ndarray:
        """Scale distances using the selected method"""
        if self.sample_count < self.min_samples:
            # Before enough samples: use simple scaling
            return np.clip(distances * 5.0, 0, 1)

        distances = np.array(distances, dtype=np.float64)

        try:
            if self.method == "robust_minmax":
                return self._robust_minmax(distances)
            elif self.method == "min_robustmax":
                return self._min_robustmax(distances)
            elif self.method == "zscore":
                return self._zscore_scaling(distances)
            elif self.method == "robust_zscore":
                return self._robust_zscore_scaling(distances)
            elif self.method == "arcsinh":
                return self._arcsinh_scaling(distances)
            elif self.method == "arcsinh_percentile":
                return self._arcsinh_percentile_scaling(distances)
            elif self.method == "beta":
                return self._beta_scaling(distances)
            elif self.method == "double_transform":
                return self._double_transformation_scaling(distances)
            elif self.method == "sqrt":
                return self._sqrt_scaling(distances)
            elif self.method == "quantile":
                return self._quantile_scaling(distances)
            elif self.method == "sigmoid":
                return self._sigmoid_scaling(distances)
            else:
                # Fallback to robust_minmax
                return self._robust_minmax(distances)

        except Exception as e:
            logger.warning(f"Scaling method {self.method} failed: {e}, using fallback")
            return np.clip(distances * 5.0, 0, 1)

    def _min_robustmax(self, distances):
        """Asymmetric scaling: actual minimum with robust maximum (P95)"""
        if self.min_distance is None or self.p95 is None:
            return np.clip(distances * 30.0, 0, 1)  # Fallback during initialization

        range_val = self.p95 - self.min_distance
        if range_val > 0:
            scaled = (distances - self.min_distance) / range_val
            scaled = np.clip(scaled, 0, 1)
            # Ensure good range utilization
            actual_range = np.max(scaled) - np.min(scaled)
            if actual_range < 0.5:
                scaled = (scaled - np.min(scaled)) / max(actual_range, 1e-6)
                scaled = scaled * 0.8 + 0.1
            return scaled
        else:
            return np.full_like(distances, 0.5)

    def _robust_minmax(self, distances):
        """Original robust min-max scaling using percentiles"""
        range_val = self.p95 - self.p5
        if range_val > 0:
            scaled = (distances - self.p5) / range_val
            scaled = np.clip(scaled, 0, 1)
            # Ensure good range utilization
            actual_range = np.max(scaled) - np.min(scaled)
            if actual_range < 0.5:
                scaled = (scaled - np.min(scaled)) / max(actual_range, 1e-6)
                scaled = scaled * 0.8 + 0.1
            return scaled
        else:
            return np.full_like(distances, 0.5)

    def _zscore_scaling(self, distances):
        """Standard z-score normalization -> sigmoid mapping to [0,1]"""
        if self.std_distance > 0:
            z_scores = (distances - self.mean_distance) / self.std_distance
            # Map z-scores to [0,1] using sigmoid
            scaled = 1 / (1 + np.exp(-z_scores))
            return scaled
        else:
            return np.full_like(distances, 0.5)

    def _robust_zscore_scaling(self, distances):
        """Robust z-score using median and MAD"""
        mad = np.median(np.abs(distances - self.median))
        if mad > 0:
            robust_z = 0.6745 * (distances - self.median) / mad
            # Map to [0,1] using sigmoid
            scaled = 1 / (1 + np.exp(-robust_z))
            return scaled
        else:
            return np.full_like(distances, 0.5)

    def _arcsinh_scaling(self, distances):
        """Arcsinh with full [0,1] range utilization"""
        # Center on median for symmetry
        centered = distances - self.median

        # Scale by robust measure
        if self.iqr > 0:
            scaled_input = centered / self.iqr
        elif self.std_distance > 0:
            scaled_input = centered / self.std_distance
        else:
            return np.full_like(distances, 0.5)

        # Apply arcsinh transformation
        arcsinh_values = np.arcsinh(scaled_input)

        # Map to [0,1] using the actual range in this batch for full utilization
        min_val = np.min(arcsinh_values)
        max_val = np.max(arcsinh_values)

        if max_val > min_val:
            scaled = (arcsinh_values - min_val) / (max_val - min_val)
        else:
            scaled = np.full_like(distances, 0.5)

        return scaled

    def _arcsinh_percentile_scaling(self, distances):
        """Arcsinh + percentile normalization for consistent [0,1] range"""
        # Apply arcsinh first
        arcsinh_distances = np.arcsinh(distances)

        # Then use percentile-based scaling for full range utilization
        p5_arcsinh = np.arcsinh(self.p5) if self.p5 is not None else np.min(arcsinh_distances)
        p95_arcsinh = np.arcsinh(self.p95) if self.p95 is not None else np.max(arcsinh_distances)

        range_val = p95_arcsinh - p5_arcsinh
        if range_val > 0:
            scaled = (arcsinh_distances - p5_arcsinh) / range_val
            scaled = np.clip(scaled, 0, 1)

            # Ensure good range utilization
            actual_range = np.max(scaled) - np.min(scaled)
            if actual_range < 0.5:
                scaled = (scaled - np.min(scaled)) / max(actual_range, 1e-6)
                scaled = scaled * 0.8 + 0.1
            return scaled
        else:
            return np.full_like(distances, 0.5)

    def _beta_scaling(self, distances):
        """Beta CDF transformation - always gives full [0,1] range"""
        # Normalize distances to [0,1] first
        range_val = self.p95 - self.p5
        if range_val > 0:
            normalized = np.clip((distances - self.p5) / range_val, 1e-6, 1 - 1e-6)

            alpha, beta = 0.8, 0.8  # Slight U-shape for better discrimination
            from scipy.stats import beta as beta_dist

            scaled = beta_dist.cdf(normalized, alpha, beta)
            return scaled
        else:
            return np.full_like(distances, 0.5)

    def _double_transformation_scaling(self, distances):
        """Two-stage: transform + normalize for guaranteed full range"""
        # Stage 1: Apply transformation
        transformed = np.arcsinh(distances - self.median)

        # Stage 2: Force to full [0,1] using percentiles
        p5_trans = np.percentile(transformed, 5)
        p95_trans = np.percentile(transformed, 95)

        range_val = p95_trans - p5_trans
        if range_val > 0:
            scaled = (transformed - p5_trans) / range_val
            return np.clip(scaled, 0, 1)
        else:
            return np.full_like(distances, 0.5)

    def _sqrt_scaling(self, distances):
        """Square root transformation - compresses large values"""
        # Normalize first, then apply sqrt
        range_val = self.p95 - self.p5
        if range_val > 0:
            normalized = np.clip((distances - self.p5) / range_val, 0, 1)
            scaled = np.sqrt(normalized)
            return scaled
        else:
            return np.full_like(distances, 0.5)

    def _quantile_scaling(self, distances):
        """Map distances to their quantile positions"""
        # This gives the empirical CDF value
        scaled = np.zeros_like(distances)
        for i, d in enumerate(distances):
            if d <= self.p5:
                scaled[i] = 0.05
            elif d >= self.p95:
                scaled[i] = 0.95
            else:
                # Linear interpolation between known percentiles
                scaled[i] = 0.05 + 0.9 * (d - self.p5) / (self.p95 - self.p5)
        return scaled

    def _sigmoid_scaling(self, distances):
        """Sigmoid with adaptive parameters"""
        # Use median as center, IQR for scale
        if self.iqr > 0:
            # Sigmoid: 1 / (1 + exp(-(x-c)/s))
            scaled = 1 / (1 + np.exp(-(distances - self.median) / self.iqr))
            return scaled
        else:
            return np.full_like(distances, 0.5)

    def reset(self):
        """Full reset of all statistics.

        Use this when:
        - Starting tracking on a new video/scene
        - The embedding distribution has changed significantly
        - Scene changes detected (camera switch, dramatic lighting change)

        After reset, the scaler will need min_samples frames to become ready again.
        """
        self.min_distance = None
        self.max_distance = None
        self.mean_distance = None
        self.std_distance = None
        self.sample_count = 0
        self.p5 = None
        self.p95 = None
        self.p1 = None
        self.p99 = None
        self.p10 = None
        self.p90 = None
        self.median = None
        self.iqr = None
        self.q25 = None
        self.q75 = None
        self.update_counter = 0
        logger.debug("EmbeddingDistanceScaler: Full reset performed")

    def soft_reset(self, faster_update_rate: float = 0.2):
        """Soft reset - keep statistics but increase learning rate temporarily.

        Use this when:
        - Embedding distribution may be shifting gradually
        - You want to adapt faster without losing all history

        Args:
            faster_update_rate: Temporary update rate (default 0.2, 4x faster than default 0.05)
        """
        original_rate = self.update_rate
        self.update_rate = faster_update_rate

        # Reset sample count to half - makes it learn faster but not from scratch
        self.sample_count = max(self.sample_count // 2, 0)

        logger.debug(
            f"EmbeddingDistanceScaler: Soft reset - update_rate {original_rate:.3f} -> {faster_update_rate:.3f}, "
            f"sample_count halved to {self.sample_count}"
        )

    def restore_update_rate(self, rate: float = None):
        """Restore the update rate after a soft reset.

        Args:
            rate: Rate to restore to. If None, uses 0.05 (default).
        """
        self.update_rate = rate if rate is not None else 0.05
        logger.debug(f"EmbeddingDistanceScaler: Update rate restored to {self.update_rate:.3f}")

    def get_statistics(self) -> dict:
        """Get current scaler statistics"""
        return {
            "method": self.method,
            "sample_count": self.sample_count,
            "min_distance": self.min_distance,
            "max_distance": self.max_distance,
            "mean_distance": self.mean_distance,
            "std_distance": self.std_distance,
            "median": self.median,
            "p1": self.p1,
            "p5": self.p5,
            "p10": self.p10,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
            "q25": self.q25,
            "q75": self.q75,
            "iqr": self.iqr,
            "ready": self.sample_count >= self.min_samples,
            "update_rate": self.update_rate,
        }


# Recommended scaling methods
RECOMMENDED_SCALING_METHODS = [
    "robust_minmax",  # Baseline method
    "min_robustmax",  # Asymmetric scaling
    "arcsinh_percentile",  # Arcsinh + percentile normalization
    "double_transform",  # Two-stage transformation
    "beta",  # Beta CDF transformation
    "quantile",  # Empirical CDF
    "robust_zscore",  # Robust z-score + sigmoid
]
