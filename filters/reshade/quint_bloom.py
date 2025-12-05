"""
qUINT_bloom.fx 정확한 구현

7-layer downsample/upsample bloom with adaptive exposure
Original: github.com/martymcmodding
"""

import numpy as np
from numba import njit

from filters.base_filter import BaseFilter


@njit(fastmath=True, cache=True)
def _downsample_pixel(img, y, x, h, w, kernel_type=0):
    """
    단일 픽셀에 대한 다운샘플링 커널

    kernel_type: 0 = low quality, 1 = high quality
    """
    # Source coordinates (2x resolution)
    sy = y * 2
    sx = x * 2

    # Center pixel
    if sy >= h or sx >= w:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    center_r = img[sy, sx, 0]
    center_g = img[sy, sx, 1]
    center_b = img[sy, sx, 2]

    # Small kernel (4 samples at ±2 offset)
    small_r, small_g, small_b = 0.0, 0.0, 0.0
    offsets_small = [(2, 2), (-2, 2), (-2, -2), (2, -2)]

    for dy, dx in offsets_small:
        ny = min(max(sy + dy, 0), h - 1)
        nx = min(max(sx + dx, 0), w - 1)
        small_r += img[ny, nx, 0]
        small_g += img[ny, nx, 1]
        small_b += img[ny, nx, 2]

    if kernel_type == 0:  # Low quality
        result_r = center_r / 5.0 + small_r / 5.0
        result_g = center_g / 5.0 + small_g / 5.0
        result_b = center_b / 5.0 + small_b / 5.0
    else:  # High quality
        # Large kernel 1 (4 samples at ±4 offset)
        large1_r, large1_g, large1_b = 0.0, 0.0, 0.0
        offsets_large = [(4, 4), (-4, 4), (-4, -4), (4, -4)]

        for dy, dx in offsets_large:
            ny = min(max(sy + dy, 0), h - 1)
            nx = min(max(sx + dx, 0), w - 1)
            large1_r += img[ny, nx, 0]
            large1_g += img[ny, nx, 1]
            large1_b += img[ny, nx, 2]

        # Large kernel 2 (4 samples at cross pattern)
        large2_r, large2_g, large2_b = 0.0, 0.0, 0.0
        offsets_cross = [(0, 4), (0, -4), (4, 0), (-4, 0)]

        for dy, dx in offsets_cross:
            ny = min(max(sy + dy, 0), h - 1)
            nx = min(max(sx + dx, 0), w - 1)
            large2_r += img[ny, nx, 0]
            large2_g += img[ny, nx, 1]
            large2_b += img[ny, nx, 2]

        result_r = (
            center_r * 0.5 / 4.0
            + small_r * 0.5 / 4.0
            + large1_r * 0.125 / 4.0
            + large2_r * 0.25 / 4.0
        )
        result_g = (
            center_g * 0.5 / 4.0
            + small_g * 0.5 / 4.0
            + large1_g * 0.125 / 4.0
            + large2_g * 0.25 / 4.0
        )
        result_b = (
            center_b * 0.5 / 4.0
            + small_b * 0.5 / 4.0
            + large1_b * 0.125 / 4.0
            + large2_b * 0.25 / 4.0
        )

    return np.array([result_r, result_g, result_b], dtype=np.float32)


@njit(fastmath=True, cache=True)
def _upsample_pixel(img, y, x, h_src, w_src, kernel_type=0):
    """
    단일 픽셀에 대한 업샘플링 커널

    kernel_type: 0 = low quality, 1 = high quality
    """
    # Map to source coordinates (0.5x resolution)
    sy = y / 2.0
    sx = x / 2.0

    # Integer coordinates
    cy = int(sy)
    cx = int(sx)

    if cy >= h_src or cx >= w_src:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Center
    center_r = img[cy, cx, 0]
    center_g = img[cy, cx, 1]
    center_b = img[cy, cx, 2]

    # Small kernel 1 (4 corners at ±1.5 texels)
    small1_r, small1_g, small1_b = 0.0, 0.0, 0.0
    offset = 1.5

    offsets = [
        (-offset, -offset),
        (offset, -offset),
        (offset, offset),
        (-offset, offset),
    ]

    for dy, dx in offsets:
        ny = int(min(max(cy + dy, 0), h_src - 1))
        nx = int(min(max(cx + dx, 0), w_src - 1))
        small1_r += img[ny, nx, 0]
        small1_g += img[ny, nx, 1]
        small1_b += img[ny, nx, 2]

    if kernel_type == 0:  # Low quality
        result_r = center_r / 5.0 + small1_r / 5.0
        result_g = center_g / 5.0 + small1_g / 5.0
        result_b = center_b / 5.0 + small1_b / 5.0
    else:  # High quality
        # Small kernel 2 (4 cross samples)
        small2_r, small2_g, small2_b = 0.0, 0.0, 0.0

        cross_offsets = [(offset, 0), (-offset, 0), (0, offset), (0, -offset)]

        for dy, dx in cross_offsets:
            ny = int(min(max(cy + dy, 0), h_src - 1))
            nx = int(min(max(cx + dx, 0), w_src - 1))
            small2_r += img[ny, nx, 0]
            small2_g += img[ny, nx, 1]
            small2_b += img[ny, nx, 2]

        result_r = center_r * 4.0 / 16.0 + small1_r * 1.0 / 16.0 + small2_r * 2.0 / 16.0
        result_g = center_g * 4.0 / 16.0 + small1_g * 1.0 / 16.0 + small2_g * 2.0 / 16.0
        result_b = center_b * 4.0 / 16.0 + small1_b * 1.0 / 16.0 + small2_b * 2.0 / 16.0

    return np.array([result_r, result_g, result_b], dtype=np.float32)


class QUINTBloomFilter(BaseFilter):
    """
    qUINT_bloom.fx 정확한 구현

    7-layer bloom with adaptive exposure and tonemap
    """

    def __init__(self):
        super().__init__("qUINT_bloom", "qUINT 블룸")

        # Bloom parameters
        self.intensity = 1.2
        self.curve = 1.5
        self.saturation = 2.0

        # Layer multipliers (7 layers)
        self.layer_mult = [0.05, 0.05, 0.05, 0.1, 0.5, 0.01, 0.01]

        # Adaptive exposure
        self.adapt_strength = 0.5
        self.adapt_exposure = 0.0
        self.adapt_mode = False  # False = adapt whole scene, True = adapt bloom only

        # Tonemap
        self.tonemap_compression = 4.0

        # Quality mode
        self.high_quality = False

        # Internal state (for frame adaptation - not used in single-frame processing)
        self.last_adapt = 0.5

    def _prepass(self, img):
        """Bloom prepass - extract bright areas"""
        # Luma
        luma = np.dot(img, [0.333, 0.333, 0.333])

        # Saturation adjustment
        result = img * self.saturation + luma[:, :, np.newaxis] * (1 - self.saturation)

        # Threshold curve
        factor = (luma**self.curve * self.intensity**3) / (luma + 1e-3)

        result = result * factor[:, :, np.newaxis]

        return result.astype(np.float32), luma.astype(np.float32)

    def _downsample(self, img, kernel_type=0):
        """Downsample image by 2x using custom kernel"""
        h, w = img.shape[:2]
        new_h = max(h // 2, 1)
        new_w = max(w // 2, 1)

        result = np.zeros((new_h, new_w, 3), dtype=np.float32)

        for y in range(new_h):
            for x in range(new_w):
                result[y, x] = _downsample_pixel(img, y, x, h, w, kernel_type)

        return result

    def _upsample(self, img, target_h, target_w, kernel_type=0):
        """Upsample image by 2x using custom kernel"""
        h_src, w_src = img.shape[:2]
        result = np.zeros((target_h, target_w, 3), dtype=np.float32)

        for y in range(target_h):
            for x in range(target_w):
                result[y, x] = _upsample_pixel(img, y, x, h_src, w_src, kernel_type)

        return result

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Apply qUINT Bloom filter

        Args:
            image: RGB image (uint8, 0-255)
            **params:
                intensity (float): Bloom intensity (0-10, default 1.2)
                curve (float): Bloom curve (0-10, default 1.5)
                saturation (float): Bloom saturation (0-5, default 2.0)
                layer_mult (list): 7 layer multipliers
                adapt_strength (float): Adaptation strength (0-1, default 0.5)
                adapt_exposure (float): Exposure bias (-5 to 5, default 0)
                tonemap_compression (float): Tonemap compression (0-10, default 4)
                high_quality (bool): Use high quality kernels (default False)

        Returns:
            Bloom applied image (uint8, 0-255)
        """
        # Update parameters
        self.intensity = params.get("intensity", self.intensity)
        self.curve = params.get("curve", self.curve)
        self.saturation = params.get("saturation", self.saturation)
        self.layer_mult = params.get("layer_mult", self.layer_mult)
        self.adapt_strength = params.get("adapt_strength", self.adapt_strength)
        self.adapt_exposure = params.get("adapt_exposure", self.adapt_exposure)
        self.tonemap_compression = params.get(
            "tonemap_compression", self.tonemap_compression
        )
        self.high_quality = params.get("high_quality", self.high_quality)

        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        orig_h, orig_w = img_float.shape[:2]

        kernel_type = 1 if self.high_quality else 0

        # Prepass - extract bright areas
        bloom_source, luma = self._prepass(img_float)

        # 7-stage downsampling
        layers = [bloom_source]
        current = bloom_source

        for i in range(7):
            current = self._downsample(current, kernel_type)
            layers.append(current)

        # Get adaptation value from smallest layer
        adapt = np.mean(layers[-1]) * 8 + 1e-3

        # 7-stage upsampling with blending
        # Start from smallest layer
        current = layers[7] * self.layer_mult[6]

        # Upsample and blend
        sizes = [
            (max(orig_h // 64, 1), max(orig_w // 64, 1)),  # Layer 6
            (max(orig_h // 32, 1), max(orig_w // 32, 1)),  # Layer 5
            (max(orig_h // 16, 1), max(orig_w // 16, 1)),  # Layer 4
            (max(orig_h // 8, 1), max(orig_w // 8, 1)),  # Layer 3
            (max(orig_h // 4, 1), max(orig_w // 4, 1)),  # Layer 2
            (max(orig_h // 2, 1), max(orig_w // 2, 1)),  # Layer 1
        ]

        for i in range(6):
            target_h, target_w = sizes[i]
            current = self._upsample(current, target_h, target_w, kernel_type)
            # Blend with corresponding layer
            layer_idx = 6 - i
            if layer_idx > 0:
                current = current + layers[layer_idx] * self.layer_mult[layer_idx - 1]

        # Final upsample to original size
        bloom = self._upsample(current, orig_h, orig_w, kernel_type)

        # Normalize by sum of layer weights
        total_weight = sum(self.layer_mult)
        bloom = bloom / max(total_weight, 1e-6)

        # Apply adaptive exposure
        if self.adapt_mode:
            # Adapt bloom only
            bloom *= np.interp(self.adapt_strength, [0, 1], [1, 1.0 / adapt])
            bloom *= 2**self.adapt_exposure
            result = img_float + bloom
        else:
            # Adapt whole scene
            result = img_float + bloom
            result *= np.interp(self.adapt_strength, [0, 1], [1, 1.0 / adapt])
            result *= 2**self.adapt_exposure

        # Tonemap
        result = np.maximum(0, result) ** self.tonemap_compression
        result = result / (1.0 + result)
        result = result ** (1.0 / self.tonemap_compression)

        # Clamp and convert
        result = np.clip(result, 0, 1)
        return (result * 255).astype(np.uint8)
