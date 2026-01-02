# ----------------------------------------------------------------------------------------------------------------------
# CUPY GPU-ACCELERATED EMBEDDINGS FOR SWARMSORT STANDALONE
# ----------------------------------------------------------------------------------------------------------------------
"""
GPU-accelerated embedding extractors for SwarmSort standalone package.
Provides cupytexture and cupytexture_mega embeddings with automatic CPU fallback.
"""

import numpy as np
import cv2
from typing import Optional, List
import numba as nb
from loguru import logger

# Optional CuPy import with graceful fallback
try:
    import cupy as cp
    from cupyx.scipy.ndimage import uniform_filter

    CUPY_AVAILABLE = True
    logger.debug("CuPy detected - GPU acceleration available")
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    logger.info("CuPy not available - falling back to CPU-only mode")


class EmbeddingExtractor:
    """Base class for embedding extractors in standalone package."""

    def __init__(self, use_tensor=False):
        self.use_tensor = use_tensor

    def extract(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Extract single embedding."""
        raise NotImplementedError

    def extract_batch(self, frame: np.ndarray, bboxes: np.ndarray) -> List[np.ndarray]:
        """Extract batch of embeddings."""
        return [self.extract(frame, bbox) for bbox in bboxes]

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimensionality."""
        raise NotImplementedError


@nb.njit(fastmath=True, parallel=True)
def extract_features_batch_jit(patches: np.ndarray, features_out: np.ndarray):
    """
    JIT-optimized batch feature extraction for microorganism patches.
    patches: (N, H, W, 3) - batch of BGR patches
    features_out: (N, feature_dim) - output features
    """
    N, H, W = patches.shape[:3]

    for i in nb.prange(N):
        patch = patches[i]

        # Convert to grayscale manually (BGR to Gray: 0.114*B + 0.587*G + 0.299*R)
        gray = np.zeros((H, W), dtype=np.float32)
        for y in range(H):
            for x in range(W):
                gray[y, x] = (
                    0.114 * patch[y, x, 0] + 0.587 * patch[y, x, 1] + 0.299 * patch[y, x, 2]
                )

        # Basic statistics (4 features)
        features_out[i, 0] = np.mean(gray)
        features_out[i, 1] = np.std(gray)
        features_out[i, 2] = np.min(gray)
        features_out[i, 3] = np.max(gray)

        # Gradient features (4 features)
        grad_x = np.zeros_like(gray)
        grad_y = np.zeros_like(gray)
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                grad_x[y, x] = gray[y, x + 1] - gray[y, x - 1]
                grad_y[y, x] = gray[y + 1, x] - gray[y - 1, x]

        features_out[i, 4] = np.mean(np.abs(grad_x))
        features_out[i, 5] = np.mean(np.abs(grad_y))
        features_out[i, 6] = np.std(grad_x)
        features_out[i, 7] = np.std(grad_y)

        # Central moments and shape features (8 features)
        cy, cx = H // 2, W // 2

        # Weighted center of mass
        total_intensity = 0.0
        weighted_x = 0.0
        weighted_y = 0.0
        for y in range(H):
            for x in range(W):
                w = gray[y, x]
                total_intensity += w
                weighted_x += w * x
                weighted_y += w * y

        if total_intensity > 0:
            cm_x = weighted_x / total_intensity
            cm_y = weighted_y / total_intensity
        else:
            cm_x = cx
            cm_y = cy

        features_out[i, 8] = cm_x - cx  # offset from geometric center
        features_out[i, 9] = cm_y - cy

        # Moments of inertia
        m20 = m02 = m11 = 0.0
        for y in range(H):
            for x in range(W):
                w = gray[y, x]
                dx = x - cm_x
                dy = y - cm_y
                m20 += w * dx * dx
                m02 += w * dy * dy
                m11 += w * dx * dy

        if total_intensity > 0:
            m20 /= total_intensity
            m02 /= total_intensity
            m11 /= total_intensity

        features_out[i, 10] = m20
        features_out[i, 11] = m02
        features_out[i, 12] = m11

        # Eccentricity and orientation
        if m20 + m02 > 0:
            lambda1 = 0.5 * (m20 + m02 + np.sqrt((m20 - m02) ** 2 + 4 * m11**2))
            lambda2 = 0.5 * (m20 + m02 - np.sqrt((m20 - m02) ** 2 + 4 * m11**2))
            eccentricity = np.sqrt(1 - lambda2 / lambda1) if lambda1 > 0 else 0
            orientation = 0.5 * np.arctan2(2 * m11, m20 - m02)
        else:
            eccentricity = 0
            orientation = 0

        features_out[i, 13] = eccentricity
        features_out[i, 14] = np.sin(orientation)  # sin for rotation invariance
        features_out[i, 15] = np.cos(orientation)  # cos for rotation invariance

        # Radial features (8 features) - intensity at different radii
        max_radius = min(H, W) // 2
        for r_idx in range(8):
            radius = (r_idx + 1) * max_radius / 8.0
            radial_sum = 0.0
            radial_count = 0

            for y in range(H):
                for x in range(W):
                    dist = np.sqrt((x - cm_x) ** 2 + (y - cm_y) ** 2)
                    if abs(dist - radius) < 0.5:
                        radial_sum += gray[y, x]
                        radial_count += 1

            if radial_count > 0:
                features_out[i, 16 + r_idx] = radial_sum / radial_count
            else:
                features_out[i, 16 + r_idx] = 0.0

        # HSV color features (8 features)
        hsv_means = np.zeros(3, dtype=np.float32)
        hsv_stds = np.zeros(3, dtype=np.float32)

        # Convert BGR to HSV manually (simplified approximation)
        for y in range(H):
            for x in range(W):
                b, g, r = patch[y, x, 0] / 255.0, patch[y, x, 1] / 255.0, patch[y, x, 2] / 255.0
                max_val = max(r, g, b)
                min_val = min(r, g, b)
                diff = max_val - min_val

                # Hue
                if diff == 0:
                    h = 0
                elif max_val == r:
                    h = (60 * ((g - b) / diff) + 360) % 360
                elif max_val == g:
                    h = (60 * ((b - r) / diff) + 120) % 360
                else:
                    h = (60 * ((r - g) / diff) + 240) % 360

                # Saturation
                s = 0 if max_val == 0 else diff / max_val

                # Value
                v = max_val

                hsv_means[0] += h
                hsv_means[1] += s
                hsv_means[2] += v

        pixel_count = H * W
        hsv_means /= pixel_count

        # Calculate HSV standard deviations
        for y in range(H):
            for x in range(W):
                b, g, r = patch[y, x, 0] / 255.0, patch[y, x, 1] / 255.0, patch[y, x, 2] / 255.0
                max_val = max(r, g, b)
                min_val = min(r, g, b)
                diff = max_val - min_val

                if diff == 0:
                    h = 0
                elif max_val == r:
                    h = (60 * ((g - b) / diff) + 360) % 360
                elif max_val == g:
                    h = (60 * ((b - r) / diff) + 120) % 360
                else:
                    h = (60 * ((r - g) / diff) + 240) % 360

                s = 0 if max_val == 0 else diff / max_val
                v = max_val

                hsv_stds[0] += (h - hsv_means[0]) ** 2
                hsv_stds[1] += (s - hsv_means[1]) ** 2
                hsv_stds[2] += (v - hsv_means[2]) ** 2

        hsv_stds = np.sqrt(hsv_stds / pixel_count)

        features_out[i, 24] = hsv_means[0] / 360.0  # Normalized hue
        features_out[i, 25] = hsv_means[1]
        features_out[i, 26] = hsv_means[2]
        features_out[i, 27] = hsv_stds[0] / 360.0
        features_out[i, 28] = hsv_stds[1]
        features_out[i, 29] = hsv_stds[2]

        # Contrast and texture features (6 features)
        # Local Binary Pattern approximation
        lbp_hist = np.zeros(8, dtype=np.float32)
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                center = gray[y, x]
                code = 0
                if gray[y - 1, x - 1] > center:
                    code |= 1
                if gray[y - 1, x] > center:
                    code |= 2
                if gray[y - 1, x + 1] > center:
                    code |= 4
                if gray[y, x + 1] > center:
                    code |= 8
                if gray[y + 1, x + 1] > center:
                    code |= 16
                if gray[y + 1, x] > center:
                    code |= 32
                if gray[y + 1, x - 1] > center:
                    code |= 64
                if gray[y, x - 1] > center:
                    code |= 128
                lbp_hist[code % 8] += 1

        lbp_sum = np.sum(lbp_hist)
        if lbp_sum > 0:
            lbp_hist /= lbp_sum

        # Use first 6 LBP features
        for j in range(6):
            features_out[i, 30 + j] = lbp_hist[j]


@nb.njit(fastmath=True, parallel=True)
def extract_mega_features_batch_jit(patches: np.ndarray, features_out: np.ndarray):
    """
    JIT-optimized batch feature extraction combining shape, color, and texture.
    - Shape: Moments, eccentricity, orientation
    - Color: HSV statistics
    - Texture: Rotation-invariant LBP, multi-scale wavelet, entropy
    """
    N, H, W = patches.shape[:3]
    FEATURE_DIM = 64

    for i in nb.prange(N):
        patch = patches[i]
        feature_idx = 0

        # ------------------- GRAYSCALE & BASIC STATS (4 features) -------------------
        gray = np.zeros((H, W), dtype=np.float32)
        for y in range(H):
            for x in range(W):
                gray[y, x] = (
                    0.114 * patch[y, x, 0] + 0.587 * patch[y, x, 1] + 0.299 * patch[y, x, 2]
                )

        features_out[i, feature_idx] = np.mean(gray)
        features_out[i, feature_idx + 1] = np.std(gray)
        features_out[i, feature_idx + 2] = np.min(gray)
        features_out[i, feature_idx + 3] = np.max(gray)
        feature_idx += 4

        # ------------------- GRADIENT FEATURES (4 features) -------------------
        grad_x = np.zeros_like(gray)
        grad_y = np.zeros_like(gray)
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                grad_x[y, x] = gray[y, x + 1] - gray[y, x - 1]
                grad_y[y, x] = gray[y + 1, x] - gray[y - 1, x]
        features_out[i, feature_idx] = np.mean(np.abs(grad_x))
        features_out[i, feature_idx + 1] = np.mean(np.abs(grad_y))
        features_out[i, feature_idx + 2] = np.std(grad_x)
        features_out[i, feature_idx + 3] = np.std(grad_y)
        feature_idx += 4

        # ------------------- SHAPE & MOMENTS (10 features) -------------------
        # Weighted Center of Mass
        total_intensity = 0.0
        weighted_x = 0.0
        weighted_y = 0.0
        for y in range(H):
            for x in range(W):
                w = gray[y, x]
                total_intensity += w
                weighted_x += w * x
                weighted_y += w * y

        if total_intensity > 1e-6:
            cm_x = weighted_x / total_intensity
            cm_y = weighted_y / total_intensity
        else:
            cm_x = W / 2
            cm_y = H / 2

        features_out[i, feature_idx] = cm_x - W / 2
        features_out[i, feature_idx + 1] = cm_y - H / 2

        # Central Moments
        m20 = m02 = m11 = 0.0
        for y in range(H):
            for x in range(W):
                w = gray[y, x]
                dx, dy = x - cm_x, y - cm_y
                m20 += w * dx * dx
                m02 += w * dy * dy
                m11 += w * dx * dy
        if total_intensity > 1e-6:
            m20 /= total_intensity
            m02 /= total_intensity
            m11 /= total_intensity
        features_out[i, feature_idx + 2] = m20
        features_out[i, feature_idx + 3] = m02
        features_out[i, feature_idx + 4] = m11

        # Eccentricity and Orientation
        denom = m20 - m02
        # Add small epsilon to avoid division by zero in arctan2
        if abs(denom) < 1e-6:
            denom += 1e-6

        discriminant = np.sqrt(denom**2 + 4 * m11**2)
        lambda1 = 0.5 * (m20 + m02 + discriminant)
        lambda2 = 0.5 * (m20 + m02 - discriminant)
        eccentricity = np.sqrt(1 - lambda2 / lambda1) if lambda1 > 1e-6 else 0
        orientation = 0.5 * np.arctan2(2 * m11, denom)
        features_out[i, feature_idx + 5] = eccentricity
        features_out[i, feature_idx + 6] = np.sin(
            2 * orientation
        )  # Use 2*orientation for rotational stability
        features_out[i, feature_idx + 7] = np.cos(2 * orientation)

        # Hu Moments (2 rotation-invariant moments)
        I1 = m20 + m02
        I2 = (m20 - m02) ** 2 + 4 * m11**2
        features_out[i, feature_idx + 8] = I1
        features_out[i, feature_idx + 9] = I2
        feature_idx += 10

        # ------------------- COLOR FEATURES (6 features) -------------------
        hsv_means = np.zeros(3, dtype=np.float32)
        hsv_stds = np.zeros(3, dtype=np.float32)
        h_vals, s_vals, v_vals = [], [], []

        for y in range(H):
            for x in range(W):
                b, g, r = patch[y, x, 0] / 255.0, patch[y, x, 1] / 255.0, patch[y, x, 2] / 255.0
                max_val, min_val = max(r, g, b), min(r, g, b)
                diff = max_val - min_val
                h, s, v = 0.0, 0.0, max_val
                if diff > 1e-6:
                    if max_val == r:
                        h = (60 * ((g - b) / diff) + 360) % 360
                    elif max_val == g:
                        h = (60 * ((b - r) / diff) + 120) % 360
                    else:
                        h = (60 * ((r - g) / diff) + 240) % 360
                if max_val > 1e-6:
                    s = diff / max_val
                h_vals.append(h)
                s_vals.append(s)
                v_vals.append(v)

        h_rad = np.array([v * np.pi / 180.0 for v in h_vals])
        features_out[i, feature_idx] = np.mean(np.array(s_vals))
        features_out[i, feature_idx + 1] = np.mean(np.array(v_vals))
        features_out[i, feature_idx + 2] = np.std(np.array(s_vals))
        features_out[i, feature_idx + 3] = np.std(np.array(v_vals))
        # Use circular mean for hue
        mean_sin, mean_cos = np.mean(np.sin(h_rad)), np.mean(np.cos(h_rad))
        features_out[i, feature_idx + 4] = np.arctan2(mean_sin, mean_cos)
        # Use circular std for hue
        features_out[i, feature_idx + 5] = np.sqrt(
            -2 * np.log(np.sqrt(mean_sin**2 + mean_cos**2))
        )
        feature_idx += 6

        # ------------------- TEXTURE: LBP (10 features) -------------------
        lbp_hist = np.zeros(36, dtype=np.float32)  # Uniform LBP has 36 patterns for 8 neighbors
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                center = gray[y, x]
                code = 0
                if gray[y - 1, x - 1] > center:
                    code |= 1
                if gray[y - 1, x] > center:
                    code |= 2
                if gray[y - 1, x + 1] > center:
                    code |= 4
                if gray[y, x + 1] > center:
                    code |= 8
                if gray[y + 1, x + 1] > center:
                    code |= 16
                if gray[y + 1, x] > center:
                    code |= 32
                if gray[y + 1, x - 1] > center:
                    code |= 64
                if gray[y, x - 1] > center:
                    code |= 128

                min_code = code
                for rot in range(1, 8):
                    rotated = ((code >> rot) | (code << (8 - rot))) & 0xFF
                    if rotated < min_code:
                        min_code = rotated
                lbp_hist[min_code % 36] += 1

        lbp_sum = np.sum(lbp_hist)
        if lbp_sum > 0:
            lbp_hist /= lbp_sum
        for j in range(10):
            features_out[i, feature_idx + j] = lbp_hist[j]
        feature_idx += 10

        # ------------------- TEXTURE: RADIAL & POLAR (16 features) -------------------
        max_radius = min(W, H) / 2.0
        for r_idx in range(4):  # 4 radii
            radius = (r_idx + 1) * max_radius / 4.0
            radial_values = []
            for angle_step in range(16):
                angle = 2 * np.pi * angle_step / 16.0
                x = cm_x + radius * np.cos(angle)
                y = cm_y + radius * np.sin(angle)
                x1, y1 = int(x), int(y)
                if 0 <= x1 < W - 1 and 0 <= y1 < H - 1:
                    fx, fy = x - x1, y - y1
                    val = (
                        gray[y1, x1] * (1 - fx) * (1 - fy)
                        + gray[y1, x1 + 1] * fx * (1 - fy)
                        + gray[y1 + 1, x1] * (1 - fx) * fy
                        + gray[y1 + 1, x1 + 1] * fx * fy
                    )
                    radial_values.append(val)

            if len(radial_values) > 1:
                radial_arr = np.array(radial_values)
                features_out[i, feature_idx] = np.mean(radial_arr)
                features_out[i, feature_idx + 1] = np.std(radial_arr)
                # DFT 1st harmonic
                n_samples = len(radial_arr)
                real_part, imag_part = 0.0, 0.0
                for n in range(n_samples):
                    angle = 2 * np.pi * n / n_samples
                    real_part += radial_arr[n] * np.cos(angle)
                    imag_part -= radial_arr[n] * np.sin(angle)
                features_out[i, feature_idx + 2] = (
                    np.sqrt(real_part**2 + imag_part**2) / n_samples
                )
            else:
                features_out[i, feature_idx : feature_idx + 3] = 0.0
            feature_idx += 4  # Note: one feature is left empty per radius for now

        # ------------------- TEXTURE: ENTROPY & WAVELET (14 features) -------------------
        # Wavelet-like features
        for scale_exp in range(4):
            block_size = 2 ** (scale_exp + 1)
            std_sum = 0.0
            count = 0
            for y in range(0, H - block_size, block_size):
                for x in range(0, W - block_size, block_size):
                    block = gray[y : y + block_size, x : x + block_size]
                    if block.size > 0:
                        std_sum += np.std(block)
                        count += 1
            if count > 0:
                features_out[i, feature_idx + scale_exp] = std_sum / count
        feature_idx += 4

        # Entropy features
        hist = np.zeros(16, dtype=np.float32)
        for y in range(H):
            for x in range(W):
                bin_idx = min(15, int(gray[y, x] / 16))
                hist[bin_idx] += 1
        if H * W > 0:
            hist /= H * W

        entropy = 0.0
        for p in hist:
            if p > 0:
                entropy -= p * np.log2(p)
        features_out[i, feature_idx] = entropy
        features_out[i, feature_idx + 1] = np.sum(hist**2)  # Uniformity
        feature_idx += 2

        # Fill any remaining features
        while feature_idx < FEATURE_DIM:
            features_out[i, feature_idx] = 0.0
            feature_idx += 1


@nb.njit(fastmath=True)
def cosine_distance_jit(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    JIT-optimized cosine distance computation.

    Cosine distance = 1 - cosine_similarity
    Returns value in [0, 1] where 0 = identical, 1 = opposite.

    This is the standard distance metric for L2-normalized embeddings.
    """
    emb1_f64 = emb1.astype(np.float64)
    emb2_f64 = emb2.astype(np.float64)

    # Dot product
    dot = 0.0
    norm1_sq = 0.0
    norm2_sq = 0.0

    for i in range(len(emb1_f64)):
        dot += emb1_f64[i] * emb2_f64[i]
        norm1_sq += emb1_f64[i] * emb1_f64[i]
        norm2_sq += emb2_f64[i] * emb2_f64[i]

    norm1 = np.sqrt(norm1_sq)
    norm2 = np.sqrt(norm2_sq)

    if norm1 < 1e-8 or norm2 < 1e-8:
        # If either vector is zero, check if both are zero
        if norm1 < 1e-8 and norm2 < 1e-8:
            return 0.0  # Both zero vectors considered identical
        return 1.0  # One zero, one non-zero = maximally different

    cosine_sim = dot / (norm1 * norm2)
    # Clamp to [-1, 1] to handle numerical errors
    cosine_sim = max(-1.0, min(1.0, cosine_sim))

    # Convert to distance in [0, 1]
    # cosine_sim = 1 -> distance = 0 (identical)
    # cosine_sim = -1 -> distance = 1 (opposite)
    distance = (1.0 - cosine_sim) / 2.0
    return distance


@nb.njit(fastmath=True)
def correlation_distance_jit(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    JIT-optimized correlation (Pearson) distance computation.

    Correlation distance = (1 - Pearson_correlation) / 2
    Returns value in [0, 1] where 0 = identical pattern, 1 = opposite pattern.

    More robust to offset differences than cosine, but may discard
    useful mean information from well-normalized embeddings.
    """
    emb1_f64 = emb1.astype(np.float64)
    emb2_f64 = emb2.astype(np.float64)

    mean1 = np.mean(emb1_f64)
    mean2 = np.mean(emb2_f64)

    numerator = np.sum((emb1_f64 - mean1) * (emb2_f64 - mean2))
    var1 = np.sum((emb1_f64 - mean1) ** 2)
    var2 = np.sum((emb2_f64 - mean2) ** 2)

    if var1 == 0.0 or var2 == 0.0:
        # Check if vectors are identical
        is_identical = True
        for i in range(len(emb1_f64)):
            if abs(emb1_f64[i] - emb2_f64[i]) > 1e-10:
                is_identical = False
                break
        return 0.0 if is_identical else 1.0

    correlation = numerator / np.sqrt(var1 * var2)
    distance = (1.0 - correlation) / 2.0
    return max(0.0, min(1.0, distance))


# Default distance metric
DEFAULT_DISTANCE_METRIC = "cosine"


class CupyTextureEmbedding(EmbeddingExtractor):
    """
    Fast GPU-accelerated embedding for microorganism tracking.
    Features: 36 dimensions
    - Basic statistics (4): mean, std, min, max
    - Gradient features (4): gradient magnitudes and variations
    - Shape features (8): center of mass, moments, eccentricity, orientation
    - Radial features (8): intensity at different radii from center
    - Color features (6): HSV statistics
    - Texture features (6): Local Binary Pattern approximation
    """

    def __init__(self, patch_size: int = 32, use_gpu: bool = True):
        super().__init__(use_tensor=False)
        self._dim = 36
        self.patch_size = patch_size
        self.use_gpu = use_gpu and CUPY_AVAILABLE

        if self.use_gpu:
            logger.info("CupyTextureEmbedding initialized with GPU acceleration")
        else:
            logger.info("CupyTextureEmbedding initialized with CPU-only mode")

    def _prepare_patch(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """Extract and resize patch from frame using bbox."""
        x1, y1, x2, y2 = map(int, bbox[:4])
        h_frame, w_frame = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_frame, x2), min(h_frame, y2)

        if x1 >= x2 or y1 >= y2:
            return None

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # Resize to standard patch size for consistency
        patch = cv2.resize(roi, (self.patch_size, self.patch_size), interpolation=cv2.INTER_AREA)
        return patch

    def extract(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Extract features for a single patch."""
        patch = self._prepare_patch(frame, bbox)
        if patch is None:
            return np.zeros(self._dim, dtype=np.float32)

        # Use batch processing even for single patch for consistency
        patches = np.expand_dims(patch, axis=0)
        features = np.zeros((1, self._dim), dtype=np.float32)
        try:
            extract_features_batch_jit(patches, features)
        except Exception as e:
            logger.warning(f"Numba JIT compilation failed: {e}. Using basic CPU fallback for single extraction.")
            # Basic CPU fallback for single patch
            if len(patch.shape) == 3:
                gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
            else:
                gray = patch.astype(np.float32)
            features[0, :min(8, self._dim)] = [
                np.mean(gray), np.std(gray), np.min(gray), np.max(gray),
                np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))),
                np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))),
                np.var(gray), np.max(gray) - np.min(gray)
            ][:min(8, self._dim)]
        return features[0]

    def extract_batch_cpu(self, frame: np.ndarray, bboxes: np.ndarray) -> List[np.ndarray]:
        """CPU batch extraction."""
        if len(bboxes) == 0:
            return []

        # Prepare all patches
        patches = []
        valid_indices = []

        for i, bbox in enumerate(bboxes):
            patch = self._prepare_patch(frame, bbox)
            if patch is not None:
                patches.append(patch)
                valid_indices.append(i)

        if not patches:
            return [np.zeros(self._dim, dtype=np.float32) for _ in bboxes]

        # Convert to batch array
        patches_array = np.array(patches)
        features = np.zeros((len(patches), self._dim), dtype=np.float32)

        # JIT-compiled batch processing with fallback for Python 3.9 compatibility
        try:
            extract_features_batch_jit(patches_array, features)
        except Exception as e:
            logger.warning(f"Numba JIT compilation failed: {e}. Using basic CPU fallback.")
            # Simple CPU fallback - basic texture features
            for i, patch in enumerate(patches):
                if len(patch.shape) == 3:
                    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
                else:
                    gray = patch.astype(np.float32)
                # Fill with basic statistics
                features[i, :min(8, self._dim)] = [
                    np.mean(gray), np.std(gray), np.min(gray), np.max(gray),
                    np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))),
                    np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))),
                    np.var(gray), np.max(gray) - np.min(gray)
                ][:min(8, self._dim)]

        # Map back to original indices
        result = []
        valid_idx = 0
        for i in range(len(bboxes)):
            if i in valid_indices:
                result.append(features[valid_idx])
                valid_idx += 1
            else:
                result.append(np.zeros(self._dim, dtype=np.float32))

        return result

    def extract_batch_gpu(self, frame: np.ndarray, bboxes: np.ndarray) -> List[np.ndarray]:
        """GPU batch extraction using CuPy."""
        if not self.use_gpu:
            logger.warning("GPU extraction requested but CuPy not available, falling back to CPU")
            return self.extract_batch_cpu(frame, bboxes)

        if len(bboxes) == 0:
            return []

        try:
            # Prepare patches on CPU first (ROI extraction is sequential)
            patches = []
            valid_indices = []

            for i, bbox in enumerate(bboxes):
                patch = self._prepare_patch(frame, bbox)
                if patch is not None:
                    patches.append(patch)
                    valid_indices.append(i)

            if not patches:
                return [np.zeros(self._dim, dtype=np.float32) for _ in bboxes]

            # Move to GPU
            patches_array = np.array(patches)
            patches_gpu = cp.asarray(patches_array)

            # GPU feature extraction
            features_gpu = self._extract_features_gpu(patches_gpu)

            # Move back to CPU
            features = cp.asnumpy(features_gpu)

            # Map back to original indices
            result = []
            valid_idx = 0
            for i in range(len(bboxes)):
                if i in valid_indices:
                    result.append(features[valid_idx])
                    valid_idx += 1
                else:
                    result.append(np.zeros(self._dim, dtype=np.float32))

            return result

        except Exception as e:
            logger.error(f"GPU extraction failed: {e}, falling back to CPU")
            return self.extract_batch_cpu(frame, bboxes)

    def _extract_features_gpu(self, patches_gpu) -> "cp.ndarray":
        """GPU-accelerated feature extraction using CuPy."""
        N, H, W, C = patches_gpu.shape
        features = cp.zeros((N, self._dim), dtype=cp.float32)

        # Convert to grayscale
        gray_gpu = cp.sum(patches_gpu * cp.array([0.114, 0.587, 0.299]), axis=3)

        # Basic statistics
        features[:, 0] = cp.mean(gray_gpu, axis=(1, 2))
        features[:, 1] = cp.std(gray_gpu, axis=(1, 2))
        features[:, 2] = cp.min(gray_gpu, axis=(1, 2))
        features[:, 3] = cp.max(gray_gpu, axis=(1, 2))

        # Gradient features
        grad_x = cp.gradient(gray_gpu, axis=2)
        grad_y = cp.gradient(gray_gpu, axis=1)

        features[:, 4] = cp.mean(cp.abs(grad_x), axis=(1, 2))
        features[:, 5] = cp.mean(cp.abs(grad_y), axis=(1, 2))
        features[:, 6] = cp.std(grad_x, axis=(1, 2))
        features[:, 7] = cp.std(grad_y, axis=(1, 2))

        # Shape and color features (simplified for GPU)
        # Center of mass
        y_coords, x_coords = cp.meshgrid(cp.arange(H), cp.arange(W), indexing="ij")
        total_intensity = cp.sum(gray_gpu, axis=(1, 2), keepdims=True)

        mask = total_intensity.squeeze() > 0
        cm_x = cp.zeros(N)
        cm_y = cp.zeros(N)

        if cp.any(mask):
            cm_x[mask] = (
                cp.sum(gray_gpu[mask] * x_coords, axis=(1, 2)) / total_intensity[mask].squeeze()
            )
            cm_y[mask] = (
                cp.sum(gray_gpu[mask] * y_coords, axis=(1, 2)) / total_intensity[mask].squeeze()
            )

        features[:, 8] = cm_x - W // 2
        features[:, 9] = cm_y - H // 2

        # HSV features
        max_vals = cp.max(patches_gpu, axis=3)
        min_vals = cp.min(patches_gpu, axis=3)

        features[:, 24] = cp.mean(max_vals, axis=(1, 2))  # Simplified V channel
        features[:, 25] = cp.mean(max_vals - min_vals, axis=(1, 2))  # Simplified S

        # Fill remaining features with texture approximations
        for i in range(10, 24):
            if i < 16:
                # Moment-based features (simplified)
                features[:, i] = cp.std(gray_gpu, axis=(1, 2)) * (i - 10) / 6
            else:
                # Radial features (simplified)
                radius_idx = i - 16
                features[:, i] = cp.mean(gray_gpu, axis=(1, 2)) * (0.8 + 0.2 * radius_idx / 8)

        # Texture features (simplified)
        for i in range(26, self._dim):
            features[:, i] = cp.std(gray_gpu, axis=(1, 2)) * (i - 26) / (self._dim - 26)

        return features

    def extract_batch(self, frame: np.ndarray, bboxes: np.ndarray) -> List[np.ndarray]:
        """Extract batch of embeddings with GPU/CPU selection."""
        if self.use_gpu:
            return self.extract_batch_gpu(frame, bboxes)
        else:
            return self.extract_batch_cpu(frame, bboxes)

    @property
    def embedding_dim(self) -> int:
        return self._dim


class CupyTextureColorEmbedding(EmbeddingExtractor):
    """
    CuPy-accelerated texture embedding with rich color features.
    
    Combines the robust texture analysis of CupyTextureEmbedding with comprehensive
    color-based features for improved tracking and re-identification performance.
    
    Feature composition (84 total features):
    - Grayscale texture features (32) - from base CupyTextureEmbedding
    - RGB color histograms (24) - 8 bins per channel
    - HSV color histograms (24) - 8 bins per channel  
    - Color moments (4) - mean and std of RGB channels
    
    This embedding is particularly effective for scenarios where both texture
    and color information are important for distinguishing between objects.
    """
    
    def __init__(self, patch_size: int = 32, use_gpu: bool = True):
        self.patch_size = patch_size
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self._dim = 84  # 32 texture + 24 RGB hist + 24 HSV hist + 4 color moments
        
        if self.use_gpu:
            logger.info(f"CupyTextureColorEmbedding initialized with GPU acceleration. Dim={self._dim}")
        else:
            logger.info(f"CupyTextureColorEmbedding initialized with CPU-only mode. Dim={self._dim}")

    def _prepare_patch(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """Extract and resize patch from frame using bbox."""
        x1, y1, x2, y2 = map(int, bbox[:4])
        h_frame, w_frame = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_frame, x2), min(h_frame, y2)

        if x1 >= x2 or y1 >= y2:
            return None

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # Resize to standard patch size for consistency
        patch = cv2.resize(roi, (self.patch_size, self.patch_size), interpolation=cv2.INTER_AREA)
        return patch

    def extract(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Extract features for a single patch."""
        patch = self._prepare_patch(frame, bbox)
        if patch is None:
            return np.zeros(self._dim, dtype=np.float32)

        # Use batch processing even for single patch for consistency
        patches = np.expand_dims(patch, axis=0)
        if self.use_gpu:
            try:
                features = self._extract_features_gpu(patches)
                return features[0]
            except Exception as e:
                logger.error(f"GPU extraction failed: {e}. Falling back to CPU.")
                self.use_gpu = False
        
        features = self._extract_features_cpu(patches)
        return features[0]

    def _extract_features_cpu(self, patches: np.ndarray) -> np.ndarray:
        """CPU-based feature extraction with color features."""
        n_patches = patches.shape[0]
        features = np.zeros((n_patches, self._dim), dtype=np.float32)
        
        for i in range(n_patches):
            patch = patches[i]  # Shape: (H, W, 3)
            H, W = patch.shape[:2]
            
            feature_idx = 0
            
            # === GRAYSCALE TEXTURE FEATURES (32 features) ===
            # Convert to grayscale
            gray = np.zeros((H, W), dtype=np.float32)
            for y in range(H):
                for x in range(W):
                    gray[y, x] = (
                        0.114 * patch[y, x, 0] +  # Blue
                        0.587 * patch[y, x, 1] +  # Green
                        0.299 * patch[y, x, 2]    # Red
                    )
            
            # Basic grayscale stats (4 features)
            features[i, feature_idx] = np.mean(gray)
            features[i, feature_idx + 1] = np.std(gray)
            features[i, feature_idx + 2] = np.min(gray)
            features[i, feature_idx + 3] = np.max(gray)
            feature_idx += 4
            
            # Gradient features (8 features)
            grad_x = np.zeros_like(gray)
            grad_y = np.zeros_like(gray)
            for y in range(1, H - 1):
                for x in range(1, W - 1):
                    grad_x[y, x] = gray[y, x + 1] - gray[y, x - 1]
                    grad_y[y, x] = gray[y + 1, x] - gray[y - 1, x]
            
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            features[i, feature_idx] = np.mean(grad_magnitude)
            features[i, feature_idx + 1] = np.std(grad_magnitude)
            features[i, feature_idx + 2] = np.max(grad_magnitude)
            features[i, feature_idx + 3] = np.mean(np.abs(grad_x))
            features[i, feature_idx + 4] = np.mean(np.abs(grad_y))
            features[i, feature_idx + 5] = np.std(grad_x)
            features[i, feature_idx + 6] = np.std(grad_y)
            features[i, feature_idx + 7] = np.mean(np.arctan2(grad_y, grad_x + 1e-8))
            feature_idx += 8
            
            # Center of mass and moments (8 features)
            total_intensity = np.sum(gray)
            if total_intensity > 0:
                y_coords, x_coords = np.meshgrid(range(H), range(W), indexing='ij')
                cm_x = np.sum(gray * x_coords) / total_intensity
                cm_y = np.sum(gray * y_coords) / total_intensity
                
                dx = x_coords - cm_x
                dy = y_coords - cm_y
                m20 = np.sum(gray * dx * dx) / total_intensity
                m02 = np.sum(gray * dy * dy) / total_intensity
                m11 = np.sum(gray * dx * dy) / total_intensity
                
                features[i, feature_idx] = cm_x / W
                features[i, feature_idx + 1] = cm_y / H
                features[i, feature_idx + 2] = m20
                features[i, feature_idx + 3] = m02
                features[i, feature_idx + 4] = m11
                features[i, feature_idx + 5] = np.sqrt(m20)
                features[i, feature_idx + 6] = np.sqrt(m02)
                features[i, feature_idx + 7] = m20 + m02  # Total moment
            feature_idx += 8
            
            # LBP-like texture features (8 features)
            lbp_hist = np.zeros(8)
            for y in range(1, H - 1):
                for x in range(1, W - 1):
                    center = gray[y, x]
                    pattern = 0
                    if gray[y - 1, x - 1] > center: pattern |= 1
                    if gray[y - 1, x] > center: pattern |= 2
                    if gray[y - 1, x + 1] > center: pattern |= 4
                    if gray[y, x + 1] > center: pattern |= 8
                    if gray[y + 1, x + 1] > center: pattern |= 16
                    if gray[y + 1, x] > center: pattern |= 32
                    if gray[y + 1, x - 1] > center: pattern |= 64
                    if gray[y, x - 1] > center: pattern |= 128
                    
                    # Simplified uniform patterns
                    bin_idx = bin(pattern).count('1') % 8
                    lbp_hist[bin_idx] += 1
            
            # Normalize histogram
            if np.sum(lbp_hist) > 0:
                lbp_hist = lbp_hist / np.sum(lbp_hist)
            features[i, feature_idx:feature_idx + 8] = lbp_hist
            feature_idx += 8
            
            # Edge and corner features (4 features)
            # Simple edge detection
            edge_horizontal = np.abs(np.diff(gray, axis=1)).mean()
            edge_vertical = np.abs(np.diff(gray, axis=0)).mean()
            corner_strength = 0.0
            variance = np.var(gray)
            
            features[i, feature_idx] = edge_horizontal
            features[i, feature_idx + 1] = edge_vertical
            features[i, feature_idx + 2] = corner_strength
            features[i, feature_idx + 3] = variance
            feature_idx += 4
            
            # === COLOR FEATURES ===
            
            # RGB Color Histogram (24 features - 8 bins per channel)
            rgb_hist = np.zeros(24)
            for c in range(3):  # B, G, R channels
                channel = patch[:, :, c]
                hist, _ = np.histogram(channel, bins=8, range=(0, 256))
                if np.sum(hist) > 0:
                    hist = hist.astype(np.float32) / np.sum(hist)
                rgb_hist[c * 8:(c + 1) * 8] = hist
            features[i, feature_idx:feature_idx + 24] = rgb_hist
            feature_idx += 24
            
            # HSV Color Histogram (24 features - 8 bins per channel)
            # Convert BGR to HSV
            hsv_patch = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_BGR2HSV)
            hsv_hist = np.zeros(24)
            
            # H channel (0-179)
            hist_h, _ = np.histogram(hsv_patch[:, :, 0], bins=8, range=(0, 180))
            if np.sum(hist_h) > 0:
                hist_h = hist_h.astype(np.float32) / np.sum(hist_h)
            hsv_hist[0:8] = hist_h
            
            # S channel (0-255)
            hist_s, _ = np.histogram(hsv_patch[:, :, 1], bins=8, range=(0, 256))
            if np.sum(hist_s) > 0:
                hist_s = hist_s.astype(np.float32) / np.sum(hist_s)
            hsv_hist[8:16] = hist_s
            
            # V channel (0-255)
            hist_v, _ = np.histogram(hsv_patch[:, :, 2], bins=8, range=(0, 256))
            if np.sum(hist_v) > 0:
                hist_v = hist_v.astype(np.float32) / np.sum(hist_v)
            hsv_hist[16:24] = hist_v
            
            features[i, feature_idx:feature_idx + 24] = hsv_hist
            feature_idx += 24
            
            # Color Moments (4 features - mean and std for each RGB channel, normalized)
            rgb_means = np.mean(patch.astype(np.float32), axis=(0, 1)) / 255.0
            rgb_stds = np.std(patch.astype(np.float32), axis=(0, 1)) / 255.0
            
            features[i, feature_idx] = np.mean(rgb_means)  # Overall color brightness
            features[i, feature_idx + 1] = np.std(rgb_means)   # Color balance variation
            features[i, feature_idx + 2] = np.mean(rgb_stds)   # Overall color consistency
            features[i, feature_idx + 3] = np.std(rgb_stds)    # Color channel variation
            feature_idx += 4
        
        return features

    def _extract_features_gpu(self, patches: np.ndarray) -> np.ndarray:
        """GPU-accelerated feature extraction with color features."""
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available for GPU computation")

        from cupyx.scipy import ndimage

        n_patches = patches.shape[0]
        patches_gpu = cp.asarray(patches, dtype=cp.float32)
        H, W = patches.shape[1:3]
        features = cp.zeros((n_patches, self._dim), dtype=cp.float32)
        
        feature_idx = 0
        
        # === GRAYSCALE TEXTURE FEATURES (32 features) ===
        # Convert to grayscale
        gray_gpu = cp.sum(patches_gpu * cp.array([0.114, 0.587, 0.299]), axis=3)
        
        # Basic grayscale stats (4 features)
        features[:, feature_idx] = cp.mean(gray_gpu, axis=(1, 2))
        features[:, feature_idx + 1] = cp.std(gray_gpu, axis=(1, 2))
        features[:, feature_idx + 2] = cp.min(gray_gpu, axis=(1, 2))
        features[:, feature_idx + 3] = cp.max(gray_gpu, axis=(1, 2))
        feature_idx += 4
        
        # Gradient features (8 features)
        grad_y, grad_x = cp.gradient(gray_gpu, axis=(1, 2))
        grad_magnitude = cp.sqrt(grad_x**2 + grad_y**2)
        
        features[:, feature_idx] = cp.mean(grad_magnitude, axis=(1, 2))
        features[:, feature_idx + 1] = cp.std(grad_magnitude, axis=(1, 2))
        features[:, feature_idx + 2] = cp.max(grad_magnitude, axis=(1, 2))
        features[:, feature_idx + 3] = cp.mean(cp.abs(grad_x), axis=(1, 2))
        features[:, feature_idx + 4] = cp.mean(cp.abs(grad_y), axis=(1, 2))
        features[:, feature_idx + 5] = cp.std(grad_x, axis=(1, 2))
        features[:, feature_idx + 6] = cp.std(grad_y, axis=(1, 2))
        features[:, feature_idx + 7] = cp.mean(cp.arctan2(grad_y, grad_x + 1e-8), axis=(1, 2))
        feature_idx += 8
        
        # Center of mass and moments (8 features)
        total_intensity = cp.sum(gray_gpu, axis=(1, 2))
        y_coords = cp.arange(H, dtype=cp.float32).reshape(H, 1)
        x_coords = cp.arange(W, dtype=cp.float32).reshape(1, W)
        
        safe_total = cp.maximum(total_intensity, 1e-8)
        cm_y = cp.sum(gray_gpu * y_coords, axis=(1, 2)) / safe_total
        cm_x = cp.sum(gray_gpu * x_coords, axis=(1, 2)) / safe_total
        
        # Compute moments
        dy = y_coords - cm_y.reshape(-1, 1, 1)
        dx = x_coords - cm_x.reshape(-1, 1, 1)
        
        m20 = cp.sum(gray_gpu * dx * dx, axis=(1, 2)) / safe_total
        m02 = cp.sum(gray_gpu * dy * dy, axis=(1, 2)) / safe_total
        m11 = cp.sum(gray_gpu * dx * dy, axis=(1, 2)) / safe_total
        
        features[:, feature_idx] = cm_x / W
        features[:, feature_idx + 1] = cm_y / H
        features[:, feature_idx + 2] = m20
        features[:, feature_idx + 3] = m02
        features[:, feature_idx + 4] = m11
        features[:, feature_idx + 5] = cp.sqrt(cp.maximum(m20, 0))
        features[:, feature_idx + 6] = cp.sqrt(cp.maximum(m02, 0))
        features[:, feature_idx + 7] = m20 + m02
        feature_idx += 8
        
        # LBP-like simplified features (8 features)
        # Use multi-scale standard deviation as texture proxy
        for scale in range(8):
            kernel_size = 2 + scale
            if kernel_size < min(H, W):

                smoothed = ndimage.uniform_filter(gray_gpu, size=kernel_size)
                features[:, feature_idx + scale] = cp.std(smoothed, axis=(1, 2))
            else:
                features[:, feature_idx + scale] = cp.std(gray_gpu, axis=(1, 2))
        feature_idx += 8
        
        # Edge and variance features (4 features)
        edge_h = cp.mean(cp.abs(cp.diff(gray_gpu, axis=2)), axis=(1, 2))
        edge_v = cp.mean(cp.abs(cp.diff(gray_gpu, axis=1)), axis=(1, 2))
        variance = cp.var(gray_gpu, axis=(1, 2))
        
        features[:, feature_idx] = edge_h
        features[:, feature_idx + 1] = edge_v
        features[:, feature_idx + 2] = variance  # Using variance instead of corner strength
        features[:, feature_idx + 3] = variance
        feature_idx += 4
        
        # === COLOR FEATURES ===
        
        # RGB Color Histogram (24 features)
        for c in range(3):  # B, G, R channels
            channel = patches_gpu[:, :, :, c]
            for b in range(8):
                bin_min = b * 32.0  # 256/8 = 32
                bin_max = (b + 1) * 32.0 if b < 7 else 256.0
                mask = (channel >= bin_min) & (channel < bin_max)
                features[:, feature_idx + c * 8 + b] = cp.mean(mask.astype(cp.float32), axis=(1, 2))
        feature_idx += 24
        
        # HSV Color Features (simplified, 24 features)
        # Use RGB-to-HSV approximation for GPU efficiency
        r_norm = patches_gpu[:, :, :, 2] / 255.0
        g_norm = patches_gpu[:, :, :, 1] / 255.0  
        b_norm = patches_gpu[:, :, :, 0] / 255.0
        
        max_rgb = cp.maximum(cp.maximum(r_norm, g_norm), b_norm)
        min_rgb = cp.minimum(cp.minimum(r_norm, g_norm), b_norm)
        delta = max_rgb - min_rgb
        
        # Hue approximation
        hue_approx = cp.where(delta > 1e-8, 
                             cp.where(max_rgb == r_norm, (g_norm - b_norm) / delta,
                                    cp.where(max_rgb == g_norm, 2.0 + (b_norm - r_norm) / delta,
                                           4.0 + (r_norm - g_norm) / delta)), 0.0)
        hue_approx = (hue_approx * 60.0) % 360.0
        
        # Saturation
        saturation = cp.where(max_rgb > 1e-8, delta / max_rgb, 0.0)
        
        # Value (brightness)
        value = max_rgb
        
        # Create histograms for HSV
        for b in range(8):
            # H bins (0-360)
            h_min, h_max = b * 45.0, (b + 1) * 45.0
            features[:, feature_idx + b] = cp.mean((hue_approx >= h_min) & (hue_approx < h_max), axis=(1, 2))
            
            # S bins (0-1)
            s_min, s_max = b * 0.125, (b + 1) * 0.125
            features[:, feature_idx + 8 + b] = cp.mean((saturation >= s_min) & (saturation < s_max), axis=(1, 2))
            
            # V bins (0-1)
            v_min, v_max = b * 0.125, (b + 1) * 0.125
            features[:, feature_idx + 16 + b] = cp.mean((value >= v_min) & (value < v_max), axis=(1, 2))
        
        feature_idx += 24
        
        # Color Moments (4 features)
        rgb_means = cp.mean(patches_gpu, axis=(1, 2)) / 255.0
        rgb_stds = cp.std(patches_gpu, axis=(1, 2)) / 255.0
        
        features[:, feature_idx] = cp.mean(rgb_means, axis=1)
        features[:, feature_idx + 1] = cp.std(rgb_means, axis=1)
        features[:, feature_idx + 2] = cp.mean(rgb_stds, axis=1)
        features[:, feature_idx + 3] = cp.std(rgb_stds, axis=1)
        feature_idx += 4
        
        return cp.asnumpy(features)

    def extract_batch_cpu(self, frame: np.ndarray, bboxes: np.ndarray) -> List[np.ndarray]:
        """CPU batch extraction."""
        if len(bboxes) == 0:
            return []

        patches = []
        valid_indices = []

        for i, bbox in enumerate(bboxes):
            patch = self._prepare_patch(frame, bbox)
            if patch is not None:
                patches.append(patch)
                valid_indices.append(i)

        if not patches:
            return [np.zeros(self._dim, dtype=np.float32) for _ in bboxes]

        patches = np.array(patches)
        features = self._extract_features_cpu(patches)

        result = [np.zeros(self._dim, dtype=np.float32) for _ in bboxes]
        for i, valid_idx in enumerate(valid_indices):
            result[valid_idx] = features[i]
        return result

    def extract_batch_gpu(self, frame: np.ndarray, bboxes: np.ndarray) -> List[np.ndarray]:
        """GPU batch extraction."""
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available for GPU computation")

        if len(bboxes) == 0:
            return []

        patches = []
        valid_indices = []

        for i, bbox in enumerate(bboxes):
            patch = self._prepare_patch(frame, bbox)
            if patch is not None:
                patches.append(patch)
                valid_indices.append(i)

        if not patches:
            return [np.zeros(self._dim, dtype=np.float32) for _ in bboxes]

        patches = np.array(patches)
        features = self._extract_features_gpu(patches)

        result = [np.zeros(self._dim, dtype=np.float32) for _ in bboxes]
        for i, valid_idx in enumerate(valid_indices):
            result[valid_idx] = features[i]
        return result

    def extract_batch(self, frame: np.ndarray, bboxes: np.ndarray) -> List[np.ndarray]:
        """Extract batch of embeddings with GPU/CPU selection."""
        if self.use_gpu:
            try:
                return self.extract_batch_gpu(frame, bboxes)
            except Exception as e:
                logger.error(f"GPU extraction failed: {e}. Falling back to CPU.")
                self.use_gpu = False
                return self.extract_batch_cpu(frame, bboxes)
        return self.extract_batch_cpu(frame, bboxes)

    @property
    def embedding_dim(self) -> int:
        return self._dim


class MegaCupyTextureEmbedding(EmbeddingExtractor):
    """
    Mega Microbe Embedding: Combines the best of shape, color, and texture analysis.
    - Dims 0-7: Basic stats and gradients
    - Dims 8-17: Advanced shape features (moments, eccentricity, Hu)
    - Dims 18-23: HSV color statistics (including circular stats for hue)
    - Dims 24-33: Rotation-invariant Local Binary Patterns (LBP)
    - Dims 34-49: Robust radial & polar features (intensity, variation, DFT)
    - Dims 50-55: Multi-scale wavelet and entropy texture features
    - Dims 56-63: Reserved / Other advanced features
    """

    def __init__(self, patch_size: int = 32, use_gpu: bool = True):
        super().__init__(use_tensor=False)
        self._dim = 64
        self.patch_size = patch_size
        self.use_gpu = use_gpu and CUPY_AVAILABLE

        if self.use_gpu:
            logger.info(
                f"MegaCupyTextureEmbedding loaded. Dim={self._dim}. GPU acceleration ENABLED."
            )
        else:
            logger.info(
                f"MegaCupyTextureEmbedding loaded. Dim={self._dim}. Running on CPU with Numba."
            )

    def _prepare_patch(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = map(int, bbox[:4])
        h_frame, w_frame = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_frame, x2), min(h_frame, y2)
        if x1 >= x2 or y1 >= y2:
            return None
        patch = cv2.resize(
            frame[y1:y2, x1:x2], (self.patch_size, self.patch_size), interpolation=cv2.INTER_AREA
        )
        return patch

    def _extract_features_cpu_fallback(self, patch: np.ndarray) -> np.ndarray:
        """CPU fallback for feature extraction without Numba JIT (Python 3.9 compatibility)."""
        H, W = patch.shape[:2]
        features = np.zeros(self._dim, dtype=np.float32)
        feature_idx = 0
        
        # Convert to grayscale
        if len(patch.shape) == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = patch.astype(np.float32)
        
        # Basic stats (4 features)
        features[feature_idx:feature_idx+4] = [
            np.mean(gray), np.std(gray), np.min(gray), np.max(gray)
        ]
        feature_idx += 4
        
        # Gradients (4 features) 
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        features[feature_idx:feature_idx+4] = [
            np.mean(np.abs(grad_x)), np.std(grad_x),
            np.mean(np.abs(grad_y)), np.std(grad_y)
        ]
        feature_idx += 4
        
        # Simple shape moments (10 features)
        moments = cv2.moments(gray)
        m00 = moments['m00'] + 1e-6
        features[feature_idx:feature_idx+10] = [
            moments['m10']/m00, moments['m01']/m00,  # centroid
            moments['m20']/m00, moments['m11']/m00, moments['m02']/m00,  # second moments
            moments['mu20']/m00, moments['mu11']/m00, moments['mu02']/m00,  # central moments
            0.0, 0.0  # eccentricity and orientation placeholders
        ]
        feature_idx += 10
        
        # Simple HSV color features (6 features)
        if len(patch.shape) == 3:
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            for i in range(3):
                channel = hsv[:,:,i].astype(np.float32)
                features[feature_idx + 2*i] = np.mean(channel)
                features[feature_idx + 2*i + 1] = np.std(channel)
        feature_idx += 6
        
        # Simple texture features - fill remaining with basic texture stats
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        remaining_features = self._dim - feature_idx
        if remaining_features > 0:
            # Fill remaining with Laplacian stats and zeros
            basic_texture = [
                np.mean(np.abs(laplacian)), np.std(laplacian),
                np.var(gray), np.max(gray) - np.min(gray)
            ]
            n_basic = min(len(basic_texture), remaining_features)
            features[feature_idx:feature_idx+n_basic] = basic_texture[:n_basic]
        
        return features

    def extract_batch_cpu(self, frame: np.ndarray, bboxes: np.ndarray) -> List[np.ndarray]:
        if len(bboxes) == 0:
            return []
        patches, valid_indices = [], []
        for i, bbox in enumerate(bboxes):
            patch = self._prepare_patch(frame, bbox)
            if patch is not None:
                patches.append(patch)
                valid_indices.append(i)

        if not patches:
            return [np.zeros(self._dim, dtype=np.float32) for _ in bboxes]

        patches_array = np.array(patches)
        features = np.zeros((len(patches), self._dim), dtype=np.float32)
        
        try:
            extract_mega_features_batch_jit(patches_array, features)
        except Exception as e:
            logger.warning(f"Numba JIT compilation failed: {e}. Falling back to CPU implementation.")
            # Fallback to CPU implementation without JIT for Python 3.9 compatibility
            for i, patch in enumerate(patches):
                features[i] = self._extract_features_cpu_fallback(patch)

        result = [np.zeros(self._dim, dtype=np.float32) for _ in bboxes]
        for i, valid_idx in enumerate(valid_indices):
            result[valid_idx] = features[i]
        return result

    def _extract_features_gpu(self, patches_gpu) -> "cp.ndarray":
        """GPU-accelerated feature extraction with improved CPU-GPU consistency."""
        N, H, W, C = patches_gpu.shape
        features = cp.zeros((N, self._dim), dtype=cp.float32)
        idx = 0

        # GRAYSCALE & BASIC STATS (4)
        gray_gpu = cp.sum(patches_gpu * cp.array([0.114, 0.587, 0.299]), axis=3)
        features[:, idx : idx + 4] = cp.stack(
            [
                cp.mean(gray_gpu, axis=(1, 2)),
                cp.std(gray_gpu, axis=(1, 2)),
                cp.min(gray_gpu, axis=(1, 2)),
                cp.max(gray_gpu, axis=(1, 2)),
            ],
            axis=1,
        )
        idx += 4

        # GRADIENTS (4)
        grad_y, grad_x = cp.gradient(gray_gpu, axis=(1, 2))
        features[:, idx : idx + 4] = cp.stack(
            [
                cp.mean(cp.abs(grad_x), axis=(1, 2)),
                cp.mean(cp.abs(grad_y), axis=(1, 2)),
                cp.std(grad_x, axis=(1, 2)),
                cp.std(grad_y, axis=(1, 2)),
            ],
            axis=1,
        )
        idx += 4

        # SHAPE & MOMENTS (10) - Vectorized implementation
        y_coords, x_coords = cp.meshgrid(
            cp.arange(H, dtype=cp.float32), cp.arange(W, dtype=cp.float32), indexing="ij"
        )
        total_intensity = cp.sum(gray_gpu, axis=(1, 2))

        safe_total = total_intensity + 1e-7
        cm_y = cp.sum(gray_gpu * y_coords, axis=(1, 2)) / safe_total
        cm_x = cp.sum(gray_gpu * x_coords, axis=(1, 2)) / safe_total

        features[:, idx] = cm_x - W / 2
        features[:, idx + 1] = cm_y - H / 2

        dx = x_coords - cm_x[:, None, None]
        dy = y_coords - cm_y[:, None, None]
        m20 = cp.sum(gray_gpu * dx * dx, axis=(1, 2)) / safe_total
        m02 = cp.sum(gray_gpu * dy * dy, axis=(1, 2)) / safe_total
        m11 = cp.sum(gray_gpu * dx * dy, axis=(1, 2)) / safe_total
        features[:, idx + 2 : idx + 5] = cp.stack([m20, m02, m11], axis=1)

        denom = m20 - m02
        discriminant = cp.sqrt(denom**2 + 4 * m11**2)
        lambda1 = 0.5 * (m20 + m02 + discriminant)
        lambda2 = 0.5 * (m20 + m02 - discriminant)
        features[:, idx + 5] = cp.sqrt(1 - lambda2 / (lambda1 + 1e-7))
        orientation = 0.5 * cp.arctan2(2 * m11, denom + 1e-7)
        features[:, idx + 6] = cp.sin(2 * orientation)
        features[:, idx + 7] = cp.cos(2 * orientation)

        features[:, idx + 8] = m20 + m02  # Hu Moment 1
        features[:, idx + 9] = (m20 - m02) ** 2 + 4 * m11**2  # Hu Moment 2
        idx += 10

        # COLOR (6) - Simplified GPU version
        bgr_norm = patches_gpu / 255.0
        r, g, b = bgr_norm[:, :, :, 2], bgr_norm[:, :, :, 1], bgr_norm[:, :, :, 0]
        max_val = cp.max(bgr_norm, axis=3)
        min_val = cp.min(bgr_norm, axis=3)
        s = (max_val - min_val) / (max_val + 1e-7)
        v = max_val
        features[:, idx : idx + 6] = cp.stack(
            [
                cp.mean(s, axis=(1, 2)),
                cp.mean(v, axis=(1, 2)),
                cp.std(s, axis=(1, 2)),
                cp.std(v, axis=(1, 2)),
                cp.mean(r, axis=(1, 2)),
                cp.std(r, axis=(1, 2)),  # Use R channel as proxy for hue stats
            ],
            axis=1,
        )
        idx += 6

        # TEXTURE (approximations) (20)
        # Multi-scale Wavelet approximation
        for scale_exp in range(4):
            size = 2 ** (scale_exp + 1)
            # Convolve with a uniform filter and get the std dev of the result
            filtered = uniform_filter(gray_gpu, size=size)
            features[:, idx + scale_exp] = cp.std(filtered, axis=(1, 2))
        idx += 4

        # Placeholder for remaining texture features
        while idx < self._dim:
            # Use scaled versions of std for variety
            features[:, idx] = features[:, 1] * ((idx % 10 + 1) / 10.0)
            idx += 1

        return features

    def extract(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        return self.extract_batch(frame, np.array([bbox]))[0]

    def extract_batch_gpu(self, frame: np.ndarray, bboxes: np.ndarray) -> List[np.ndarray]:
        if len(bboxes) == 0:
            return []
        patches, valid_indices = [], []
        for i, bbox in enumerate(bboxes):
            patch = self._prepare_patch(frame, bbox)
            if patch is not None:
                patches.append(patch)
                valid_indices.append(i)

        if not patches:
            return [np.zeros(self._dim, dtype=np.float32) for _ in bboxes]

        patches_gpu = cp.asarray(np.array(patches))
        features_gpu = self._extract_features_gpu(patches_gpu)
        features = cp.asnumpy(features_gpu)

        result = [np.zeros(self._dim, dtype=np.float32) for _ in bboxes]
        for i, valid_idx in enumerate(valid_indices):
            result[valid_idx] = features[i]
        return result

    def extract_batch(self, frame: np.ndarray, bboxes: np.ndarray) -> List[np.ndarray]:
        """Extract batch of embeddings with GPU/CPU selection."""
        if self.use_gpu:
            try:
                return self.extract_batch_gpu(frame, bboxes)
            except Exception as e:
                logger.error(f"GPU extraction failed: {e}. Falling back to CPU.")
                self.use_gpu = False  # Prevent further GPU attempts
                return self.extract_batch_cpu(frame, bboxes)
        return self.extract_batch_cpu(frame, bboxes)

    @property
    def embedding_dim(self) -> int:
        return self._dim


def compute_embedding_distance(
    emb1: np.ndarray,
    emb2: np.ndarray,
    metric: str = "cosine"
) -> float:
    """
    Compute distance between two embeddings.

    Args:
        emb1: First embedding vector
        emb2: Second embedding vector
        metric: Distance metric to use
            - "cosine" (default): Cosine distance, standard for L2-normalized embeddings
            - "correlation": Pearson correlation distance, robust to offsets

    Returns:
        Distance in [0, 1] where 0 = identical, 1 = maximally different
    """
    if emb1 is None or emb2 is None:
        return float("inf")
    if emb1.shape != emb2.shape:
        logger.warning(f"Embeddings have different shapes: {emb1.shape} vs {emb2.shape}")
        return float("inf")
    if emb1.size == 0:
        return float("inf")

    try:
        if metric == "cosine":
            return cosine_distance_jit(emb1, emb2)
        elif metric == "correlation":
            return correlation_distance_jit(emb1, emb2)
        else:
            logger.warning(f"Unknown metric '{metric}', using cosine")
            return cosine_distance_jit(emb1, emb2)
    except Exception as e:
        logger.error(f"Error in embedding distance computation: {e}")
        return float("inf")


def compute_embedding_distances_batch(
    emb: np.ndarray,
    embs: List[np.ndarray],
    metric: str = "cosine"
) -> np.ndarray:
    """
    Compute distances from one embedding to multiple embeddings.

    Args:
        emb: Query embedding vector
        embs: List of embedding vectors to compare against
        metric: Distance metric ("cosine" or "correlation")

    Returns:
        Array of distances
    """
    if emb is None or not embs:
        return np.array([])

    distances = np.full(len(embs), float("inf"))

    # Select distance function
    if metric == "cosine":
        dist_fn = cosine_distance_jit
    elif metric == "correlation":
        dist_fn = correlation_distance_jit
    else:
        logger.warning(f"Unknown metric '{metric}', using cosine")
        dist_fn = cosine_distance_jit

    for i, emb2 in enumerate(embs):
        if emb2 is not None and emb.shape == emb2.shape and emb.size > 0:
            try:
                distances[i] = dist_fn(emb, emb2)
            except Exception as e:
                logger.error(f"Error in batch distance computation for item {i}: {e}")

    return distances


class CupyShapeEmbedding(EmbeddingExtractor):
    """
    Shape-focused embedding optimized for microscopy organism tracking.

    Key design principles (from embedding_recommendation.md):
    - NO RESIZE: Processes original patch resolution to preserve fine details
    - SHAPE-FOCUSED: Prioritizes contour/boundary features over texture
    - ROTATION-INVARIANT: Handles organisms at any angle

    Feature composition (96 dimensions):
    - HOG features (36 dims): Histogram of Oriented Gradients for local shape
    - Fourier descriptors (16 dims): Contour shape in frequency domain
    - Radial signature (32 dims): Distance from centroid to boundary
    - Hu moments (7 dims): Classic rotation/scale invariant moments
    - Shape stats (5 dims): Area, perimeter, circularity, aspect ratio, solidity

    This embedding is specifically designed for grayscale microscopy images
    where organisms have similar internal textures but distinct boundary shapes.
    """

    def __init__(self, use_gpu: bool = True, hog_cells: int = 3, hog_bins: int = 9):
        super().__init__(use_tensor=False)
        self._dim = 96  # 36 HOG + 16 Fourier + 32 radial + 7 Hu + 5 shape stats
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.hog_cells = hog_cells  # Grid of cells for HOG
        self.hog_bins = hog_bins    # Orientation bins

        if self.use_gpu:
            logger.info(f"CupyShapeEmbedding initialized with GPU. Dim={self._dim}")
        else:
            logger.info(f"CupyShapeEmbedding initialized (CPU). Dim={self._dim}")

    def _prepare_patch(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """Extract patch WITHOUT resizing to preserve fine morphological details."""
        x1, y1, x2, y2 = map(int, bbox[:4])
        h_frame, w_frame = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_frame, x2), min(h_frame, y2)

        if x1 >= x2 or y1 >= y2:
            return None

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # NO RESIZE - return original patch
        return roi

    def _compute_hog_features(self, gray: np.ndarray) -> np.ndarray:
        """
        Compute HOG (Histogram of Oriented Gradients) features.

        Adaptive cell size based on patch dimensions.
        Returns 36 features (hog_cells x hog_cells x hog_bins normalized).
        """
        H, W = gray.shape
        n_cells = self.hog_cells
        n_bins = self.hog_bins

        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        # Magnitude and orientation
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        orientation = np.arctan2(grad_y, grad_x)  # -pi to pi
        orientation = (orientation + np.pi) * (n_bins / (2 * np.pi))  # 0 to n_bins
        orientation = orientation.astype(np.int32) % n_bins

        # Adaptive cell size
        cell_h = max(1, H // n_cells)
        cell_w = max(1, W // n_cells)

        hog_features = np.zeros((n_cells, n_cells, n_bins), dtype=np.float32)

        for cy in range(n_cells):
            for cx in range(n_cells):
                y_start = cy * cell_h
                y_end = min((cy + 1) * cell_h, H)
                x_start = cx * cell_w
                x_end = min((cx + 1) * cell_w, W)

                cell_mag = magnitude[y_start:y_end, x_start:x_end]
                cell_ori = orientation[y_start:y_end, x_start:x_end]

                # Build histogram weighted by magnitude
                for b in range(n_bins):
                    mask = cell_ori == b
                    hog_features[cy, cx, b] = np.sum(cell_mag[mask])

        # L2 normalize per block (2x2 cells)
        hog_normalized = np.zeros_like(hog_features)
        for by in range(n_cells - 1):
            for bx in range(n_cells - 1):
                block = hog_features[by:by+2, bx:bx+2, :].flatten()
                norm = np.linalg.norm(block) + 1e-6
                block_normalized = block / norm
                # Distribute back (simplified - just use the normalization factor)
                hog_normalized[by:by+2, bx:bx+2, :] /= (norm / (4 * n_bins) + 1e-6)

        # Flatten to 36 features (or pad/truncate)
        flat = hog_features.flatten()
        result = np.zeros(36, dtype=np.float32)
        n_copy = min(len(flat), 36)
        result[:n_copy] = flat[:n_copy]

        # Normalize final vector
        norm = np.linalg.norm(result) + 1e-6
        return result / norm

    def _extract_contour(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Extract the largest contour from grayscale image."""
        # Adaptive thresholding for better contour extraction
        if gray.max() - gray.min() < 10:
            return None

        # Normalize to 0-255
        gray_norm = ((gray - gray.min()) / (gray.max() - gray.min() + 1e-6) * 255).astype(np.uint8)

        # Try Otsu thresholding
        _, binary = cv2.threshold(gray_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            # Try inverted
            _, binary_inv = cv2.threshold(gray_norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return None

        # Return largest contour
        largest = max(contours, key=cv2.contourArea)
        return largest.squeeze()

    def _compute_fourier_descriptors(self, contour: np.ndarray, n_descriptors: int = 16) -> np.ndarray:
        """
        Compute Fourier descriptors of contour for rotation/scale invariant shape.

        Steps:
        1. Sample N points uniformly along contour
        2. Represent as complex numbers z[k] = x[k] + i*y[k]
        3. Compute FFT
        4. Normalize for scale/rotation invariance
        5. Return magnitude of harmonics 2 to n_descriptors+1
        """
        result = np.zeros(n_descriptors, dtype=np.float32)

        if contour is None or len(contour) < 4:
            return result

        # Ensure 2D
        if len(contour.shape) == 1:
            return result
        if contour.shape[1] != 2:
            if len(contour.shape) == 3:
                contour = contour.reshape(-1, 2)
            else:
                return result

        # Sample 64 points uniformly
        n_sample = 64
        n_points = len(contour)
        if n_points < n_sample:
            # Upsample by repeating
            indices = np.linspace(0, n_points - 1, n_sample).astype(int)
        else:
            indices = np.linspace(0, n_points - 1, n_sample).astype(int)

        sampled = contour[indices].astype(np.float64)

        # Represent as complex numbers
        z = sampled[:, 0] + 1j * sampled[:, 1]

        # Compute FFT
        Z = np.fft.fft(z)

        # Normalize by Z[1] for scale invariance (if non-zero)
        if np.abs(Z[1]) > 1e-6:
            Z = Z / Z[1]

        # Take magnitude for rotation invariance
        # Skip DC (Z[0]) and first harmonic (Z[1], now = 1)
        # Use harmonics 2 to n_descriptors+1
        for i in range(n_descriptors):
            idx = i + 2
            if idx < len(Z):
                result[i] = np.abs(Z[idx])

        # Normalize
        norm = np.linalg.norm(result) + 1e-6
        return (result / norm).astype(np.float32)

    def _compute_radial_signature(self, gray: np.ndarray, contour: np.ndarray, n_angles: int = 32) -> np.ndarray:
        """
        Compute radial signature: distance from centroid to contour at N angles.

        This captures global shape variation and is rotation-invariant when
        normalized by mean distance.
        """
        result = np.zeros(n_angles, dtype=np.float32)

        if contour is None or len(contour) < 4:
            return result

        # Ensure 2D contour
        if len(contour.shape) != 2 or contour.shape[1] != 2:
            return result

        # Compute centroid
        M = cv2.moments(gray)
        if M['m00'] < 1e-6:
            cx, cy = gray.shape[1] / 2, gray.shape[0] / 2
        else:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']

        # Cast rays at uniform angles
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

        for i, angle in enumerate(angles):
            # Ray direction
            dx, dy = np.cos(angle), np.sin(angle)

            # Find intersection with contour (closest point along ray direction)
            # Use signed distance along ray
            contour_centered = contour - np.array([cx, cy])

            # Project contour points onto ray
            projections = contour_centered[:, 0] * dx + contour_centered[:, 1] * dy

            # Only consider points in positive ray direction
            positive_mask = projections > 0
            if not np.any(positive_mask):
                continue

            # Find closest point in ray direction
            positive_projections = projections[positive_mask]
            positive_contour = contour_centered[positive_mask]

            # Distance from ray (perpendicular)
            perp_dist = np.abs(positive_contour[:, 0] * (-dy) + positive_contour[:, 1] * dx)

            # Find point closest to ray
            closest_idx = np.argmin(perp_dist)
            result[i] = positive_projections[closest_idx]

        # Normalize by mean distance for scale invariance
        mean_dist = np.mean(result[result > 0]) if np.any(result > 0) else 1.0
        result = result / (mean_dist + 1e-6)

        return result

    def _compute_hu_moments(self, gray: np.ndarray) -> np.ndarray:
        """
        Compute 7 Hu moments (rotation, scale, translation invariant).
        Log-transformed for numerical stability.
        """
        moments = cv2.moments(gray)
        hu = cv2.HuMoments(moments).flatten()

        # Log transform with sign preservation
        hu_log = np.zeros(7, dtype=np.float32)
        for i in range(7):
            if hu[i] != 0:
                hu_log[i] = -np.sign(hu[i]) * np.log10(np.abs(hu[i]) + 1e-10)

        return hu_log

    def _compute_shape_stats(self, gray: np.ndarray, contour: np.ndarray) -> np.ndarray:
        """
        Compute basic shape statistics:
        - Area, Perimeter, Circularity, Aspect ratio, Solidity
        """
        result = np.zeros(5, dtype=np.float32)

        if contour is None or len(contour) < 4:
            return result

        # Ensure proper contour format for OpenCV
        if len(contour.shape) == 2:
            contour_cv = contour.reshape(-1, 1, 2).astype(np.int32)
        else:
            contour_cv = contour.astype(np.int32)

        area = cv2.contourArea(contour_cv)
        perimeter = cv2.arcLength(contour_cv, closed=True)

        # Circularity: 4*pi*area / perimeter^2 (1.0 for perfect circle)
        circularity = (4 * np.pi * area) / (perimeter**2 + 1e-6) if perimeter > 0 else 0

        # Bounding rect aspect ratio
        x, y, w, h = cv2.boundingRect(contour_cv)
        aspect_ratio = float(w) / (h + 1e-6)

        # Solidity: area / convex hull area
        hull = cv2.convexHull(contour_cv)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-6) if hull_area > 0 else 0

        # Normalize
        result[0] = np.sqrt(area) / 100.0  # Normalized area
        result[1] = perimeter / 500.0       # Normalized perimeter
        result[2] = circularity
        result[3] = min(aspect_ratio, 1.0 / (aspect_ratio + 1e-6))  # Make symmetric
        result[4] = solidity

        return result

    def extract(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Extract shape-focused embedding for a single patch."""
        patch = self._prepare_patch(frame, bbox)
        if patch is None:
            return np.zeros(self._dim, dtype=np.float32)

        return self._extract_features_single(patch)

    def _extract_features_single(self, patch: np.ndarray) -> np.ndarray:
        """Extract all shape features from a single patch."""
        features = np.zeros(self._dim, dtype=np.float32)
        idx = 0

        # Convert to grayscale
        if len(patch.shape) == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = patch.astype(np.float32)

        # Extract contour (used by multiple feature types)
        contour = self._extract_contour(gray)

        # 1. HOG features (36 dims)
        hog = self._compute_hog_features(gray)
        features[idx:idx+36] = hog
        idx += 36

        # 2. Fourier descriptors (16 dims)
        fourier = self._compute_fourier_descriptors(contour, n_descriptors=16)
        features[idx:idx+16] = fourier
        idx += 16

        # 3. Radial signature (32 dims)
        radial = self._compute_radial_signature(gray, contour, n_angles=32)
        features[idx:idx+32] = radial
        idx += 32

        # 4. Hu moments (7 dims)
        hu = self._compute_hu_moments(gray)
        features[idx:idx+7] = hu
        idx += 7

        # 5. Shape stats (5 dims)
        shape_stats = self._compute_shape_stats(gray, contour)
        features[idx:idx+5] = shape_stats
        idx += 5

        # Final L2 normalization
        norm = np.linalg.norm(features) + 1e-6
        return features / norm

    def extract_batch(self, frame: np.ndarray, bboxes: np.ndarray) -> List[np.ndarray]:
        """Extract batch of shape embeddings."""
        if len(bboxes) == 0:
            return []

        results = []
        for bbox in bboxes:
            patch = self._prepare_patch(frame, bbox)
            if patch is None:
                results.append(np.zeros(self._dim, dtype=np.float32))
            else:
                results.append(self._extract_features_single(patch))

        return results

    @property
    def embedding_dim(self) -> int:
        return self._dim


# Available embedding extractors
AVAILABLE_EMBEDDINGS = {
    "cupytexture": CupyTextureEmbedding,
    "cupytexture_color": CupyTextureColorEmbedding,
    "cupytexture_mega": MegaCupyTextureEmbedding,
    "cupyshape": CupyShapeEmbedding,
}


def get_embedding_extractor(name: str, **kwargs) -> EmbeddingExtractor:
    """Get an embedding extractor by name."""
    if name not in AVAILABLE_EMBEDDINGS:
        available = list(AVAILABLE_EMBEDDINGS.keys())
        raise ValueError(f"Embedding '{name}' not found. Available: {available}")

    return AVAILABLE_EMBEDDINGS[name](**kwargs)


def list_available_embeddings() -> List[str]:
    """List all available embeddings."""
    return list(AVAILABLE_EMBEDDINGS.keys())


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return CUPY_AVAILABLE
