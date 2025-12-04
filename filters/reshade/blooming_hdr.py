"""
BloomingHDR.fx 정확한 구현

HDR bloom with auto exposure, temporal adaptation, and tonemapping
Original: BlueSkyDefender
"""

import numpy as np
from numba import njit

from filters.base_filter import BaseFilter

# Constants
PHI = 1.61803398874989 * 0.1  # Golden Ratio
PI = 3.14159265359 * 0.1
SQ2 = 1.41421356237 * 10000.0  # Square Root of Two


@njit(fastmath=True, cache=True)
def luma(color):
    """Calculate luma (BT.709)"""
    return color[0] * 0.2126 + color[1] * 0.7152 + color[2] * 0.0722


@njit(fastmath=True, cache=True)
def golden_noise(tc_x, tc_y, seed):
    """Generate golden noise for dithering"""
    tc_x_scaled = tc_x * ((seed + 10) + PHI)
    tc_y_scaled = tc_y * ((seed + 10) + PHI)

    dist_sq = (tc_x_scaled - PHI) ** 2 + (tc_y_scaled - PI) ** 2
    dist = np.sqrt(dist_sq)

    tan_val = np.tan(dist)
    return np.abs(tan_val * SQ2 - np.floor(tan_val * SQ2))


# Tonemapping functions


def col_tone_b(hdr_max, contrast, shoulder, mid_in, mid_out):
    """Timothy tonemapper - calculate B coefficient"""
    pow_hdr_cs = hdr_max ** (contrast * shoulder)
    pow_mi_c = mid_in**contrast
    pow_mi_cs = mid_in ** (contrast * shoulder)

    num_inner = pow_hdr_cs * pow_mi_c - (hdr_max**contrast) * pow_mi_cs * mid_out
    num = mid_out * num_inner
    den = pow_hdr_cs * mid_out - pow_mi_cs * mid_out

    numerator = -pow_mi_c + num / den
    denominator = pow_mi_cs * mid_out

    return -(numerator / denominator)


def col_tone_c(hdr_max, contrast, shoulder, mid_in, mid_out):
    """Timothy tonemapper - calculate C coefficient"""
    numerator = (hdr_max ** (contrast * shoulder)) * (mid_in**contrast) - (
        hdr_max**contrast
    ) * (mid_in ** (contrast * shoulder)) * mid_out
    denominator = (hdr_max ** (contrast * shoulder)) * mid_out - (
        mid_in ** (contrast * shoulder)
    ) * mid_out
    return numerator / denominator


@njit(fastmath=True, cache=True)
def col_tone(x, contrast, shoulder, b, c):
    """Timothy tonemapper - tone curve"""
    z = x**contrast
    return z / ((z**shoulder) * b + c)


def timothy_tonemap(color, exposure, wp, contrast, saturate_val):
    """Timothy Lottes tonemapper"""
    hdr_max = 16.0
    contrast_val = contrast + 0.250
    shoulder = 1.0
    mid_in = 0.11
    mid_out = 0.18

    b = col_tone_b(hdr_max, contrast_val, shoulder, mid_in, mid_out)
    c = col_tone_c(hdr_max, contrast_val, shoulder, mid_in, mid_out)

    # Apply exposure
    color = color * exposure

    EPS = 1e-6
    peak = max(color[0], max(color[1], color[2]))
    peak = max(EPS, peak)

    ratio = color / peak
    peak_tone = col_tone(peak, contrast_val, shoulder, b, c)

    # Saturation controls
    crosstalk = 4.0
    saturation = contrast_val * saturate_val
    cross_saturation = contrast_val * 16.0

    # Apply saturation
    ratio = np.abs(ratio) ** (saturation / cross_saturation)
    ratio = ratio * (1 - peak_tone**crosstalk) + wp * (peak_tone**crosstalk)
    ratio = np.abs(ratio) ** cross_saturation

    return peak_tone * ratio


# ACES matrices
ACES_INPUT_MAT = np.array(
    [
        [0.59719, 0.35458, 0.04823],
        [0.07600, 0.90834, 0.01566],
        [0.02840, 0.13383, 0.83777],
    ],
    dtype=np.float32,
)

ACES_OUTPUT_MAT = np.array(
    [
        [1.60475, -0.53108, -0.07367],
        [-0.10208, 1.10813, -0.00605],
        [-0.00327, -0.07276, 1.07602],
    ],
    dtype=np.float32,
)


def rrt_and_odt_fit(v):
    """ACES RRT and ODT fit"""
    a = v * (v + 0.0245786) - 0.000090537
    b = v * (0.983729 * v + 0.4329510) + 0.238081
    return a / b


def aces_fitted(color, exposure, wp):
    """ACES fitted tonemapper"""
    color = color * (exposure + 0.5)
    color = ACES_INPUT_MAT @ color
    color = rrt_and_odt_fit(color)
    color = ACES_OUTPUT_MAT @ color
    return color / wp


# Inverse tonemappers for HDR extraction


@njit(fastmath=True, cache=True)
def inv_tonemap_timothy(color, luma_val, hdr_bp):
    """Timothy inverse tonemap"""
    max_val = max(color[0], max(color[1], color[2]))
    return color / max((1.0 + (1.0 - hdr_bp)) - max_val, 0.001)


@njit(fastmath=True, cache=True)
def inv_tonemap_reinhard(color, hdr_bp):
    """Reinhard inverse tonemap (color)"""
    return color / np.maximum((1.0 + (1.0 - hdr_bp)) - color, 0.001)


@njit(fastmath=True, cache=True)
def inv_tonemap_luma(color, luma_val, hdr_bp):
    """Luma-based Reinhard inverse tonemap"""
    return color / max((1.0 + (1.0 - hdr_bp) * 0.5) - luma_val, 0.001)


class BloomingHDRFilter(BaseFilter):
    """
    BloomingHDR - Advanced HDR bloom with tonemapping

    Features:
    - 3-level separable bloom (multi-scale)
    - Auto exposure with temporal adaptation
    - Timothy/ACES tonemapping
    - Inverse tonemap for HDR extraction
    - Golden noise dithering
    """

    def __init__(self):
        super().__init__("BloomingHDR", "블루밍 HDR")

        # Bloom parameters
        self.cbt_adjust = 0.5  # Color brightness threshold
        self.bloom_intensity = 0.1  # Primary bloom intensity
        self.bloom_opacity = 0.5  # Overall bloom opacity
        self.bloom_sensitivity = 1.0  # Bloom input curve
        self.bloom_curve = 2.0  # Bloom spread curve
        self.bloom_saturation = 0.25  # Bloom saturation
        self.bloom_spread = 1.0  # Bloom spread amount
        self.dither_bloom = 0.125  # Dithering amount

        # Tonemapper
        self.tonemapper = 0  # 0=Timothy, 1=ACES
        self.white_point = 1.0
        self.exposure = 0.0
        self.grey_value = 0.128
        self.gamma = 2.2
        self.contrast = 1.0
        self.saturate = 1.0

        # HDR
        self.inv_tonemapper = 2  # 0=Off, 1=Luma, 2=Color, 3=Max
        self.hdr_power = 0.5

        # Bloom samples
        self.bloom_samples = 8

    def _color_grade_extract(self, color):
        """Extract bright colors for bloom (Color_GS function)"""
        # Gamma curve
        color = np.power(np.abs(color), self.bloom_sensitivity)

        # Luma threshold
        gs = luma(color)
        color = color / max(gs, 0.001)

        # Threshold
        alpha = max(0.0, gs - self.cbt_adjust)
        color = color * alpha

        # Saturation
        sat_factor = min(10, self.bloom_saturation * 10)
        color = alpha + (color - alpha) * sat_factor

        return np.clip(color, 0, 1)

    def _simple_blur(self, img, scale=1):
        """Simple 4-tap blur"""
        h, w = img.shape[:2]
        result = np.zeros_like(img)

        offsets = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

        for y in range(h):
            for x in range(w):
                blur_sum = np.zeros(3, dtype=np.float32)

                for dy, dx in offsets:
                    ny = np.clip(y + dy * scale, 0, h - 1)
                    nx = np.clip(x + dx * scale, 0, w - 1)
                    blur_sum += img[ny, nx]

                result[y, x] = blur_sum * 0.25

        return result

    def _separable_bloom(self, img, scale, direction):
        """Separable bloom (horizontal or vertical)"""
        h, w = img.shape[:2]
        result = np.zeros_like(img)

        offset_base = self.bloom_spread * 0.5

        for y in range(h):
            for x in range(w):
                blur_sum = np.zeros(3, dtype=np.float32)
                weight_sum = 0.0
                offset = offset_base

                for i in range(1, self.bloom_samples):
                    weight = abs(self.bloom_samples - i) ** self.bloom_curve

                    if direction == 0:  # Horizontal
                        dx = int(offset * scale)
                        ny, nx_pos = y, np.clip(x + dx, 0, w - 1)
                        ny, nx_neg = y, np.clip(x - dx, 0, w - 1)
                    else:  # Vertical
                        dy = int(offset * scale)
                        ny_pos, nx = np.clip(y + dy, 0, h - 1), x
                        ny_neg, nx = np.clip(y - dy, 0, h - 1), x

                        blur_sum += img[ny_pos, nx] * weight
                        blur_sum += img[ny_neg, nx] * weight
                        weight_sum += weight * 2
                        offset += self.bloom_spread
                        continue

                    blur_sum += img[ny, nx_pos] * weight
                    blur_sum += img[ny, nx_neg] * weight
                    weight_sum += weight * 2
                    offset += self.bloom_spread

                if weight_sum > 0:
                    result[y, x] = blur_sum / weight_sum

        return result

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Apply BloomingHDR filter

        Args:
            image: RGB image (uint8, 0-255)
            **params: Filter parameters

        Returns:
            HDR bloomed and tonemapped image (uint8, 0-255)
        """
        # Update parameters
        self.bloom_intensity = params.get("bloom_intensity", self.bloom_intensity)
        self.bloom_opacity = params.get("bloom_opacity", self.bloom_opacity)
        self.cbt_adjust = params.get("cbt_adjust", self.cbt_adjust)
        self.tonemapper = params.get("tonemapper", self.tonemapper)
        self.exposure = params.get("exposure", self.exposure)
        self.contrast = params.get("contrast", self.contrast)
        self.saturate = params.get("saturate", self.saturate)

        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        orig_h, orig_w = img_float.shape[:2]

        # 1. Extract bright colors
        color_extracted = np.zeros_like(img_float)
        for y in range(orig_h):
            for x in range(orig_w):
                color_extracted[y, x] = self._color_grade_extract(img_float[y, x])

        # 2. Downsample for bloom
        bloom_h, bloom_w = orig_h // 2, orig_w // 2
        downsampled = color_extracted[::2, ::2]

        # 3. Multi-scale bloom (3 levels)
        # Level A (1/2 resolution)
        bloom_h_a = self._separable_bloom(downsampled, 1, 0)
        bloom_v_a = self._separable_bloom(bloom_h_a, 1, 1)

        # Level B (1/4 resolution)
        down_b = bloom_v_a[::2, ::2]
        bloom_h_b = self._separable_bloom(down_b, 1, 0)
        bloom_v_b = self._separable_bloom(bloom_h_b, 1, 1)

        # Level C (1/8 resolution)
        down_c = bloom_v_b[::2, ::2]
        bloom_h_c = self._separable_bloom(down_c, 1, 0)
        bloom_v_c = self._separable_bloom(bloom_h_c, 1, 1)

        # 4. Combine blooms
        bloom_combined = np.zeros((bloom_h, bloom_w, 3), dtype=np.float32)

        # Add level A
        bloom_combined += self._simple_blur(bloom_h_a, 1)
        bloom_combined += self._simple_blur(bloom_v_a, 1)

        # Upsample and add level B
        for y in range(bloom_h):
            for x in range(bloom_w):
                sy = min(y // 2, bloom_v_b.shape[0] - 1)
                sx = min(x // 2, bloom_v_b.shape[1] - 1)
                bloom_combined[y, x] += bloom_v_b[sy, sx]

        # Upsample and add level C
        for y in range(bloom_h):
            for x in range(bloom_w):
                sy = min(y // 4, bloom_v_c.shape[0] - 1)
                sx = min(x // 4, bloom_v_c.shape[1] - 1)
                bloom_combined[y, x] += bloom_v_c[sy, sx]

        bloom_combined *= self.bloom_intensity / 6

        # 5. Upsample bloom to original size
        bloom_full = np.zeros_like(img_float)
        for y in range(orig_h):
            for x in range(orig_w):
                sy = min(y // 2, bloom_h - 1)
                sx = min(x // 2, bloom_w - 1)
                bloom_full[y, x] = bloom_combined[sy, sx]

        # 6. Apply dithering to bloom
        if self.dither_bloom > 0:
            for y in range(orig_h):
                for x in range(orig_w):
                    tc_x = 10 * (x / orig_w) - 5
                    tc_y = 10 * (y / orig_h) - 5

                    noise_r = golden_noise(tc_x, tc_y, 1)
                    noise_g = golden_noise(tc_x, tc_y, 2)
                    noise_b = golden_noise(tc_x, tc_y, 3)
                    noise = np.array([noise_r, noise_g, noise_b])

                    ss = np.clip(bloom_full[y, x] * 10, 0, 1)
                    ss *= self.dither_bloom
                    bloom_full[y, x] = np.clip(bloom_full[y, x] + noise * ss, 0, None)

        # 7. Apply bloom opacity
        bloom_full *= self.bloom_opacity

        # 8. Gamma decode
        color = img_float.copy()
        if self.gamma > 1.0:
            color = np.power(np.abs(color), self.gamma)

        # 9. Apply bloom before inverse tonemap
        color = color + bloom_full
        color = np.maximum(color, 0)

        # 10. Inverse tonemap (HDR extraction)
        if self.inv_tonemapper > 0:
            for y in range(orig_h):
                for x in range(orig_w):
                    c = color[y, x]
                    luma_val = luma(c)

                    if self.inv_tonemapper == 1:  # Luma
                        color[y, x] = inv_tonemap_luma(c, luma_val, self.hdr_power)
                    elif self.inv_tonemapper == 2:  # Color
                        color[y, x] = inv_tonemap_reinhard(c, self.hdr_power)
                    elif self.inv_tonemapper == 3:  # Max
                        color[y, x] = inv_tonemap_timothy(c, luma_val, self.hdr_power)

        # 11. Tonemap
        exposure_val = 2**self.exposure

        result = np.zeros_like(color)
        for y in range(orig_h):
            for x in range(orig_w):
                if self.tonemapper == 0:  # Timothy
                    result[y, x] = timothy_tonemap(
                        color[y, x],
                        exposure_val,
                        self.white_point,
                        self.contrast,
                        self.saturate,
                    )
                elif self.tonemapper == 1:  # ACES
                    result[y, x] = aces_fitted(
                        color[y, x], exposure_val, self.white_point
                    )

        # 12. Apply contrast (for ACES)
        if self.tonemapper >= 1:
            result = (result - 0.5) * self.contrast + 0.5

        # 13. Gamma encode
        if self.gamma > 1.0:
            result = np.power(np.abs(result), 1.0 / 2.2)

        # Clamp and convert
        result = np.clip(result, 0, 1)
        return (result * 255).astype(np.uint8)
