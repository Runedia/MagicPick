"""
HDRBloom.fx 정확한 구현

HDR Bloom with 8-level multi-scale gaussian blur
Original: MaxG3D
Based on: MagicHDR by luluco250
"""

import numpy as np
from numba import njit, prange

from filters.base_filter import BaseFilter

# Gaussian weights (from HDRShadersFunctions.fxh)
WEIGHTS_5 = np.array(
    [
        0.0613595978134402,
        0.24477019552960988,
        0.38774041331389975,
        0.24477019552960988,
        0.0613595978134402,
    ],
    dtype=np.float32,
)

WEIGHTS_7 = np.array(
    [
        0.03050260371857921,
        0.10546420324961808,
        0.2218866945336653,
        0.28429299699627486,
        0.2218866945336653,
        0.10546420324961808,
        0.03050260371857921,
    ],
    dtype=np.float32,
)

WEIGHTS_11 = np.array(
    [
        0.014642062351313795,
        0.03622922216280118,
        0.0732908252747015,
        0.1212268244846623,
        0.163954439140855,
        0.18131325317133223,
        0.163954439140855,
        0.1212268244846623,
        0.0732908252747015,
        0.03622922216280118,
        0.014642062351313795,
    ],
    dtype=np.float32,
)

WEIGHTS_13 = np.array(
    [
        0.011311335636445246,
        0.02511527845053647,
        0.04823491379898901,
        0.08012955958832953,
        0.11514384884108936,
        0.14312253396755542,
        0.1538850594341098,
        0.14312253396755542,
        0.11514384884108936,
        0.08012955958832953,
        0.04823491379898901,
        0.02511527845053647,
        0.011311335636445246,
    ],
    dtype=np.float32,
)

# Luma coefficients
LUM_COEFF_HDR = np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)


@njit(fastmath=True, cache=True)
def luminance(color):
    """Calculate HDR luminance (BT.2020)"""
    return color[0] * 0.2627 + color[1] * 0.6780 + color[2] * 0.0593


@njit(fastmath=True, cache=True)
def reinhard_inverse(color):
    """Reinhard inverse tonemapping"""
    hdr_max = 10000.0
    return (color * (1.0 + color)) / (1.0 + color / (hdr_max * hdr_max))


@njit(parallel=True, fastmath=True, cache=True)
def adaptive_saturation(img, h, w, sat_amount):
    """
    Adaptive saturation adjustment

    Args:
        img: Image (H, W, 3)
        h, w: Image dimensions
        sat_amount: Saturation multiplier

    Returns:
        Saturated image
    """
    result = np.empty((h, w, 3), dtype=np.float32)

    for y in prange(h):
        for x in range(w):
            r = img[y, x, 0]
            g = img[y, x, 1]
            b = img[y, x, 2]

            lum = 0.2627 * r + 0.6780 * g + 0.0593 * b

            # Calculate color difference from gray
            diff_r = r - lum
            diff_g = g - lum
            diff_b = b - lum

            # Initial saturation
            diff_len = np.sqrt(diff_r * diff_r + diff_g * diff_g + diff_b * diff_b)
            init_sat = diff_len / max(lum, 1e-5)

            # Smooth modulation
            modulation = init_sat / (1.0 + init_sat)  # smoothstep approximation
            factor = 1.0 + (sat_amount - 1.0) * modulation

            # Apply saturation
            sat_r = lum + diff_r * factor
            sat_g = lum + diff_g * factor
            sat_b = lum + diff_b * factor

            # Brightness limiter
            max_orig = max(r, max(g, b))
            max_sat = max(sat_r, max(sat_g, sat_b))

            if max_sat > max_orig and max_sat > 1e-6:
                scale = max_orig / max_sat
                sat_r *= scale
                sat_g *= scale
                sat_b *= scale

            result[y, x, 0] = sat_r
            result[y, x, 1] = sat_g
            result[y, x, 2] = sat_b

    return result


@njit(parallel=True, fastmath=True, cache=True)
def gaussian_blur_h(img, h, w, weights, blur_size, scale):
    """Horizontal gaussian blur"""
    result = np.zeros((h, w, 3), dtype=np.float32)
    kernel_size = len(weights)
    half_kernel = kernel_size // 2

    for y in prange(h):
        for x in range(w):
            r_sum = 0.0
            g_sum = 0.0
            b_sum = 0.0

            for i in range(kernel_size):
                offset = int((i - half_kernel) * blur_size * scale)
                nx = min(max(x + offset, 0), w - 1)

                weight = weights[i]
                r_sum += img[y, nx, 0] * weight
                g_sum += img[y, nx, 1] * weight
                b_sum += img[y, nx, 2] * weight

            result[y, x, 0] = r_sum
            result[y, x, 1] = g_sum
            result[y, x, 2] = b_sum

    return result


@njit(parallel=True, fastmath=True, cache=True)
def gaussian_blur_v(img, h, w, weights, blur_size, scale):
    """Vertical gaussian blur"""
    result = np.zeros((h, w, 3), dtype=np.float32)
    kernel_size = len(weights)
    half_kernel = kernel_size // 2

    for y in prange(h):
        for x in range(w):
            r_sum = 0.0
            g_sum = 0.0
            b_sum = 0.0

            for i in range(kernel_size):
                offset = int((i - half_kernel) * blur_size * scale)
                ny = min(max(y + offset, 0), h - 1)

                weight = weights[i]
                r_sum += img[ny, x, 0] * weight
                g_sum += img[ny, x, 1] * weight
                b_sum += img[ny, x, 2] * weight

            result[y, x, 0] = r_sum
            result[y, x, 1] = g_sum
            result[y, x, 2] = b_sum

    return result


class HDRBloomFilter(BaseFilter):
    """
    HDRBloom - HDR 블룸

    Features:
    - 8-level multi-scale bloom (1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128 resolutions)
    - Gaussian blur with configurable quality (Medium/High/Ultra/Overkill)
    - Adaptive saturation
    - Inverse tonemapping support (Reinhard)
    - Additive/Overlay blending modes
    """

    def __init__(self):
        super().__init__("HDRBloom", "HDR 블룸")

        # Bloom parameters
        self.bloom_amount = 0.05  # 0.0 ~ 1.0
        self.bloom_brightness = 1.0  # 0.001 ~ 10.0
        self.bloom_saturation = 1.0  # 0.0 ~ 10.0
        self.blur_size = 2.0  # 0.5 ~ 4.0

        # Quality (0=Medium/Weights5, 1=High/Weights7, 2=Ultra/Weights11, 3=Overkill/Weights13)
        self.sample_quality = 0  # 0, 1, 2, 3

        # Blending
        self.blending_type = 1  # 0=Additive, 1=Overlay

        # Inverse tonemapping
        self.inv_tonemap = 0  # 0=None, 1=Reinhard

        # Downsample factor
        self.downsample = 4  # Fixed at 4x for performance

    def _get_weights(self):
        """Get gaussian weights based on quality setting"""
        if self.sample_quality == 0:
            return WEIGHTS_5
        elif self.sample_quality == 1:
            return WEIGHTS_7
        elif self.sample_quality == 2:
            return WEIGHTS_11
        else:
            return WEIGHTS_13

    def _downsample(self, img, factor):
        """Simple downsampling by nearest neighbor"""
        return img[::factor, ::factor]

    def _upsample_bilinear(self, img, target_h, target_w):
        """Bilinear upsampling"""
        h, w = img.shape[:2]
        result = np.zeros((target_h, target_w, 3), dtype=np.float32)

        scale_y = h / target_h
        scale_x = w / target_w

        for y in range(target_h):
            for x in range(target_w):
                # Source coordinates (floating point)
                sy = y * scale_y
                sx = x * scale_x

                # Integer part
                sy0 = int(sy)
                sx0 = int(sx)
                sy1 = min(sy0 + 1, h - 1)
                sx1 = min(sx0 + 1, w - 1)

                # Fractional part
                fy = sy - sy0
                fx = sx - sx0

                # Bilinear interpolation
                for c in range(3):
                    v00 = img[sy0, sx0, c]
                    v01 = img[sy0, sx1, c]
                    v10 = img[sy1, sx0, c]
                    v11 = img[sy1, sx1, c]

                    v0 = v00 * (1 - fx) + v01 * fx
                    v1 = v10 * (1 - fx) + v11 * fx
                    result[y, x, c] = v0 * (1 - fy) + v1 * fy

        return result

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Apply HDRBloom filter

        Args:
            image: RGB image (uint8, 0-255)
            **params: Filter parameters
                - bloom_amount: Bloom amount (0.0 ~ 1.0, default 0.05)
                - bloom_brightness: Brightness multiplier (0.001 ~ 10.0, default 1.0)
                - bloom_saturation: Saturation multiplier (0.0 ~ 10.0, default 1.0)
                - blur_size: Gaussian blur size (0.5 ~ 4.0, default 2.0)
                - sample_quality: 0=Medium, 1=High, 2=Ultra, 3=Overkill (default 0)
                - blending_type: 0=Additive, 1=Overlay (default 1)
                - inv_tonemap: 0=None, 1=Reinhard (default 0)

        Returns:
            HDR bloomed image (uint8, 0-255)
        """
        # Update parameters
        self.bloom_amount = params.get("bloom_amount", self.bloom_amount)
        self.bloom_brightness = params.get("bloom_brightness", self.bloom_brightness)
        self.bloom_saturation = params.get("bloom_saturation", self.bloom_saturation)
        self.blur_size = params.get("blur_size", self.blur_size)
        self.sample_quality = params.get("sample_quality", self.sample_quality)
        self.blending_type = params.get("blending_type", self.blending_type)
        self.inv_tonemap = params.get("inv_tonemap", self.inv_tonemap)

        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        original = img_float.copy()
        orig_h, orig_w = img_float.shape[:2]

        # 1. Preprocessing: Inverse tonemap (optional)
        if self.inv_tonemap == 1:
            for y in range(orig_h):
                for x in range(orig_w):
                    img_float[y, x] = reinhard_inverse(img_float[y, x])

        # 2. Apply brightness
        img_float = img_float * self.bloom_brightness

        # 3. Apply adaptive saturation
        img_float = adaptive_saturation(
            img_float, orig_h, orig_w, self.bloom_saturation
        )
        img_float = np.clip(img_float, 0, 65504.0)  # FLT16_MAX

        # 4. Downsample to 1/4 resolution
        downsampled = self._downsample(img_float, self.downsample)
        base_h, base_w = downsampled.shape[:2]

        # Get weights
        weights = self._get_weights()

        # 5. Create 8 bloom levels
        bloom_levels = [downsampled.copy()]

        # Generate downsampled levels (1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128)
        for i in range(7):
            prev_level = bloom_levels[-1]
            bloom_levels.append(self._downsample(prev_level, 2))

        # 6. Apply gaussian blur to each level (horizontal → vertical)
        blurred_levels = []
        for i, level in enumerate(bloom_levels):
            h, w = level.shape[:2]
            scale = 2**i

            # Horizontal blur
            h_blurred = gaussian_blur_h(level, h, w, weights, self.blur_size, scale)

            # Vertical blur
            v_blurred = gaussian_blur_v(h_blurred, h, w, weights, self.blur_size, scale)

            blurred_levels.append(v_blurred)

        # 7. Combine all bloom levels (upsample and add)
        combined_bloom = np.zeros((base_h, base_w, 3), dtype=np.float32)

        for i, bloom in enumerate(blurred_levels):
            if i == 0:
                combined_bloom += bloom
            else:
                # Upsample to base resolution
                upsampled = self._upsample_bilinear(bloom, base_h, base_w)
                combined_bloom += upsampled

        # Average
        combined_bloom /= len(blurred_levels)

        # 8. Upsample final bloom to original resolution
        final_bloom = self._upsample_bilinear(combined_bloom, orig_h, orig_w)

        # 9. Blend with original
        if self.blending_type == 0:  # Additive
            result = original + (final_bloom * self.bloom_amount)
        else:  # Overlay
            blend_factor = np.log10(self.bloom_amount + 1.0)
            result = original + (final_bloom - original) * blend_factor

        # Clamp and convert
        result = np.clip(result, 0, 1)
        return (result * 255).astype(np.uint8)
