# ----------------------------------------------------------------------------------------------------------------------
# CUPY GPU-ACCELERATED EMBEDDINGS FOR SWARMSORT STANDALONE
# ----------------------------------------------------------------------------------------------------------------------
"""
GPU-accelerated embedding extractors for SwarmSort standalone package.
Provides cupytexture and mega_cupytexture embeddings with automatic CPU fallback.
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
    logger.info("CuPy detected - GPU acceleration available")
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
                gray[y, x] = 0.114 * patch[y, x, 0] + 0.587 * patch[y, x, 1] + 0.299 * patch[y, x, 2]

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
            lambda1 = 0.5 * (m20 + m02 + np.sqrt((m20 - m02) ** 2 + 4 * m11 ** 2))
            lambda2 = 0.5 * (m20 + m02 - np.sqrt((m20 - m02) ** 2 + 4 * m11 ** 2))
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
                if gray[y - 1, x - 1] > center: code |= 1
                if gray[y - 1, x] > center: code |= 2
                if gray[y - 1, x + 1] > center: code |= 4
                if gray[y, x + 1] > center: code |= 8
                if gray[y + 1, x + 1] > center: code |= 16
                if gray[y + 1, x] > center: code |= 32
                if gray[y + 1, x - 1] > center: code |= 64
                if gray[y, x - 1] > center: code |= 128
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
                gray[y, x] = 0.114 * patch[y, x, 0] + 0.587 * patch[y, x, 1] + 0.299 * patch[y, x, 2]

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

        discriminant = np.sqrt(denom ** 2 + 4 * m11 ** 2)
        lambda1 = 0.5 * (m20 + m02 + discriminant)
        lambda2 = 0.5 * (m20 + m02 - discriminant)
        eccentricity = np.sqrt(1 - lambda2 / lambda1) if lambda1 > 1e-6 else 0
        orientation = 0.5 * np.arctan2(2 * m11, denom)
        features_out[i, feature_idx + 5] = eccentricity
        features_out[i, feature_idx + 6] = np.sin(2 * orientation)  # Use 2*orientation for rotational stability
        features_out[i, feature_idx + 7] = np.cos(2 * orientation)

        # Hu Moments (2 rotation-invariant moments)
        I1 = m20 + m02
        I2 = (m20 - m02) ** 2 + 4 * m11 ** 2
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
                if max_val > 1e-6: s = diff / max_val
                h_vals.append(h);
                s_vals.append(s);
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
        features_out[i, feature_idx + 5] = np.sqrt(-2 * np.log(np.sqrt(mean_sin ** 2 + mean_cos ** 2)))
        feature_idx += 6

        # ------------------- TEXTURE: LBP (10 features) -------------------
        lbp_hist = np.zeros(36, dtype=np.float32)  # Uniform LBP has 36 patterns for 8 neighbors
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                center = gray[y, x]
                code = 0
                if gray[y - 1, x - 1] > center: code |= 1
                if gray[y - 1, x] > center: code |= 2
                if gray[y - 1, x + 1] > center: code |= 4
                if gray[y, x + 1] > center: code |= 8
                if gray[y + 1, x + 1] > center: code |= 16
                if gray[y + 1, x] > center: code |= 32
                if gray[y + 1, x - 1] > center: code |= 64
                if gray[y, x - 1] > center: code |= 128

                min_code = code
                for rot in range(1, 8):
                    rotated = ((code >> rot) | (code << (8 - rot))) & 0xFF
                    if rotated < min_code:
                        min_code = rotated
                lbp_hist[min_code % 36] += 1

        lbp_sum = np.sum(lbp_hist)
        if lbp_sum > 0: lbp_hist /= lbp_sum
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
                    val = (gray[y1, x1] * (1 - fx) * (1 - fy) + gray[y1, x1 + 1] * fx * (1 - fy) +
                           gray[y1 + 1, x1] * (1 - fx) * fy + gray[y1 + 1, x1 + 1] * fx * fy)
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
                features_out[i, feature_idx + 2] = np.sqrt(real_part ** 2 + imag_part ** 2) / n_samples
            else:
                features_out[i, feature_idx:feature_idx + 3] = 0.0
            feature_idx += 4  # Note: one feature is left empty per radius for now

        # ------------------- TEXTURE: ENTROPY & WAVELET (14 features) -------------------
        # Wavelet-like features
        for scale_exp in range(4):
            block_size = 2 ** (scale_exp + 1)
            std_sum = 0.0
            count = 0
            for y in range(0, H - block_size, block_size):
                for x in range(0, W - block_size, block_size):
                    block = gray[y:y + block_size, x:x + block_size]
                    if block.size > 0:
                        std_sum += np.std(block)
                        count += 1
            if count > 0: features_out[i, feature_idx + scale_exp] = std_sum / count
        feature_idx += 4

        # Entropy features
        hist = np.zeros(16, dtype=np.float32)
        for y in range(H):
            for x in range(W):
                bin_idx = min(15, int(gray[y, x] / 16))
                hist[bin_idx] += 1
        if H * W > 0: hist /= (H * W)

        entropy = 0.0
        for p in hist:
            if p > 0: entropy -= p * np.log2(p)
        features_out[i, feature_idx] = entropy
        features_out[i, feature_idx + 1] = np.sum(hist ** 2)  # Uniformity
        feature_idx += 2

        # Fill any remaining features
        while feature_idx < FEATURE_DIM:
            features_out[i, feature_idx] = 0.0
            feature_idx += 1


@nb.njit(fastmath=True)
def correlation_distance_jit(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """JIT-optimized correlation distance computation."""
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
        extract_features_batch_jit(patches, features)
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

        # JIT-compiled batch processing
        extract_features_batch_jit(patches_array, features)

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

    def _extract_features_gpu(self, patches_gpu) -> 'cp.ndarray':
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
        y_coords, x_coords = cp.meshgrid(cp.arange(H), cp.arange(W), indexing='ij')
        total_intensity = cp.sum(gray_gpu, axis=(1, 2), keepdims=True)

        mask = total_intensity.squeeze() > 0
        cm_x = cp.zeros(N)
        cm_y = cp.zeros(N)

        if cp.any(mask):
            cm_x[mask] = cp.sum(gray_gpu[mask] * x_coords, axis=(1, 2)) / total_intensity[mask].squeeze()
            cm_y[mask] = cp.sum(gray_gpu[mask] * y_coords, axis=(1, 2)) / total_intensity[mask].squeeze()

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
            logger.info(f"MegaCupyTextureEmbedding loaded. Dim={self._dim}. GPU acceleration ENABLED.")
        else:
            logger.info(f"MegaCupyTextureEmbedding loaded. Dim={self._dim}. Running on CPU with Numba.")

    def _prepare_patch(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = map(int, bbox[:4])
        h_frame, w_frame = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_frame, x2), min(h_frame, y2)
        if x1 >= x2 or y1 >= y2: return None
        patch = cv2.resize(frame[y1:y2, x1:x2], (self.patch_size, self.patch_size), interpolation=cv2.INTER_AREA)
        return patch

    def extract_batch_cpu(self, frame: np.ndarray, bboxes: np.ndarray) -> List[np.ndarray]:
        if len(bboxes) == 0: return []
        patches, valid_indices = [], []
        for i, bbox in enumerate(bboxes):
            patch = self._prepare_patch(frame, bbox)
            if patch is not None:
                patches.append(patch)
                valid_indices.append(i)

        if not patches: return [np.zeros(self._dim, dtype=np.float32) for _ in bboxes]

        patches_array = np.array(patches)
        features = np.zeros((len(patches), self._dim), dtype=np.float32)
        extract_mega_features_batch_jit(patches_array, features)

        result = [np.zeros(self._dim, dtype=np.float32) for _ in bboxes]
        for i, valid_idx in enumerate(valid_indices):
            result[valid_idx] = features[i]
        return result

    def _extract_features_gpu(self, patches_gpu) -> 'cp.ndarray':
        """GPU-accelerated feature extraction with improved CPU-GPU consistency."""
        N, H, W, C = patches_gpu.shape
        features = cp.zeros((N, self._dim), dtype=cp.float32)
        idx = 0

        # GRAYSCALE & BASIC STATS (4)
        gray_gpu = cp.sum(patches_gpu * cp.array([0.114, 0.587, 0.299]), axis=3)
        features[:, idx:idx + 4] = cp.stack([
            cp.mean(gray_gpu, axis=(1, 2)), cp.std(gray_gpu, axis=(1, 2)),
            cp.min(gray_gpu, axis=(1, 2)), cp.max(gray_gpu, axis=(1, 2))
        ], axis=1)
        idx += 4

        # GRADIENTS (4)
        grad_y, grad_x = cp.gradient(gray_gpu, axis=(1, 2))
        features[:, idx:idx + 4] = cp.stack([
            cp.mean(cp.abs(grad_x), axis=(1, 2)), cp.mean(cp.abs(grad_y), axis=(1, 2)),
            cp.std(grad_x, axis=(1, 2)), cp.std(grad_y, axis=(1, 2))
        ], axis=1)
        idx += 4

        # SHAPE & MOMENTS (10) - Vectorized implementation
        y_coords, x_coords = cp.meshgrid(cp.arange(H, dtype=cp.float32), cp.arange(W, dtype=cp.float32), indexing='ij')
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
        features[:, idx + 2:idx + 5] = cp.stack([m20, m02, m11], axis=1)

        denom = m20 - m02
        discriminant = cp.sqrt(denom ** 2 + 4 * m11 ** 2)
        lambda1 = 0.5 * (m20 + m02 + discriminant)
        lambda2 = 0.5 * (m20 + m02 - discriminant)
        features[:, idx + 5] = cp.sqrt(1 - lambda2 / (lambda1 + 1e-7))
        orientation = 0.5 * cp.arctan2(2 * m11, denom + 1e-7)
        features[:, idx + 6] = cp.sin(2 * orientation)
        features[:, idx + 7] = cp.cos(2 * orientation)

        features[:, idx + 8] = m20 + m02  # Hu Moment 1
        features[:, idx + 9] = (m20 - m02) ** 2 + 4 * m11 ** 2  # Hu Moment 2
        idx += 10

        # COLOR (6) - Simplified GPU version
        bgr_norm = patches_gpu / 255.0
        r, g, b = bgr_norm[:, :, :, 2], bgr_norm[:, :, :, 1], bgr_norm[:, :, :, 0]
        max_val = cp.max(bgr_norm, axis=3)
        min_val = cp.min(bgr_norm, axis=3)
        s = (max_val - min_val) / (max_val + 1e-7)
        v = max_val
        features[:, idx:idx + 6] = cp.stack([
            cp.mean(s, axis=(1, 2)), cp.mean(v, axis=(1, 2)),
            cp.std(s, axis=(1, 2)), cp.std(v, axis=(1, 2)),
            cp.mean(r, axis=(1, 2)), cp.std(r, axis=(1, 2))  # Use R channel as proxy for hue stats
        ], axis=1)
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
        if len(bboxes) == 0: return []
        patches, valid_indices = [], []
        for i, bbox in enumerate(bboxes):
            patch = self._prepare_patch(frame, bbox)
            if patch is not None:
                patches.append(patch)
                valid_indices.append(i)

        if not patches: return [np.zeros(self._dim, dtype=np.float32) for _ in bboxes]

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


def compute_embedding_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute correlation-based distance between embeddings."""
    if emb1 is None or emb2 is None:
        return float('inf')
    if emb1.shape != emb2.shape:
        logger.warning(f"Embeddings have different shapes: {emb1.shape} vs {emb2.shape}")
        return float('inf')
    if emb1.size == 0:
        return float('inf')

    try:
        return correlation_distance_jit(emb1, emb2)
    except Exception as e:
        logger.error(f"Error in embedding distance computation: {e}")
        return float('inf')


def compute_embedding_distances_batch(emb: np.ndarray, embs: List[np.ndarray]) -> np.ndarray:
    """Compute distances from one embedding to multiple embeddings."""
    if emb is None or not embs:
        return np.array([])

    distances = np.full(len(embs), float('inf'))

    for i, emb2 in enumerate(embs):
        if emb2 is not None and emb.shape == emb2.shape and emb.size > 0:
            try:
                distances[i] = correlation_distance_jit(emb, emb2)
            except Exception as e:
                logger.error(f"Error in batch distance computation for item {i}: {e}")

    return distances


# Available embedding extractors
AVAILABLE_EMBEDDINGS = {
    'cupytexture': CupyTextureEmbedding,
    'mega_cupytexture': MegaCupyTextureEmbedding,
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