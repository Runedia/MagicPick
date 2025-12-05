"""
MagicHDR.fx 정확한 구현

HDR Bloom with Tonemapping
Original: FXShaders

Features:
- 6종류 Inverse Tonemapping
- 7-level Multi-scale Bloom
- 6종류 Output Tonemapping
- Input/Output Exposure
- Bloom Brightness/Saturation
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from filters.base_filter import BaseFilter

# Tonemap Type Constants
TONEMAP_REINHARD = 0
TONEMAP_LOTTES = 1
TONEMAP_UNREAL3 = 2
TONEMAP_NARKOWICZ_ACES = 3
TONEMAP_UNCHARTED2_FILMIC = 4
TONEMAP_BAKING_LAB_ACES = 5


class MagicHDRFilter(BaseFilter):
    """
    MagicHDR - HDR 블룸 및 토네맵핑

    Features:
    - Inverse tonemapping (6 types)
    - Multi-scale bloom (7 levels)
    - Output tonemapping (6 types)
    - Exposure control
    """

    def __init__(self):
        super().__init__("MagicHDR", "매직 HDR")

        # Tonemapping parameters
        self.input_exposure = 0.0  # -3.0 ~ 3.0 (f-stops)
        self.exposure = 0.0  # -3.0 ~ 3.0 (f-stops)
        self.inv_tonemap = TONEMAP_REINHARD
        self.tonemap = TONEMAP_BAKING_LAB_ACES

        # Bloom parameters
        self.bloom_amount = 0.3  # 0.0 ~ 1.0
        self.bloom_brightness = 3.0  # 1.0 ~ 5.0
        self.bloom_saturation = 1.0  # 0.0 ~ 2.0
        self.blur_size = 0.5  # 0.01 ~ 1.0
        self.blending_amount = 0.5  # 0.1 ~ 1.0
        self.blending_base = 0.8  # 0.0 ~ 1.0

        # Debug
        self.show_bloom = False

    # ========== Tonemap Functions ==========

    def _tonemap_reinhard_apply(self, color):
        """Reinhard tonemapping"""
        return color / (1.0 + color)

    def _tonemap_reinhard_inverse(self, color):
        """Inverse Reinhard tonemapping"""
        return -(color / np.minimum(color - 1.0, -0.1))

    def _tonemap_lottes_apply(self, color):
        """Lottes tonemapping"""
        max_val = np.maximum(color[:, :, 0], np.maximum(color[:, :, 1], color[:, :, 2]))
        max_val = np.expand_dims(max_val, axis=2)
        return color / (max_val + 1.0)

    def _tonemap_lottes_inverse(self, color):
        """Inverse Lottes tonemapping"""
        max_val = np.maximum(color[:, :, 0], np.maximum(color[:, :, 1], color[:, :, 2]))
        max_val = np.expand_dims(max_val, axis=2)
        return color / np.maximum(1.0 - max_val, 0.1)

    def _tonemap_unreal3_apply(self, color):
        """Unreal3 tonemapping"""
        return color / (color + 0.155) * 1.019

    def _tonemap_unreal3_inverse(self, color):
        """Inverse Unreal3 tonemapping"""
        return (color * -0.155) / (np.maximum(color, 0.01) - 1.019)

    def _tonemap_narkowicz_aces_apply(self, color):
        """Narkowicz ACES tonemapping"""
        A, B, C, D, E = 2.51, 0.03, 2.43, 0.59, 0.14
        return np.clip((color * (A * color + B)) / (color * (C * color + D) + E), 0, 1)

    def _tonemap_narkowicz_aces_inverse(self, color):
        """Inverse Narkowicz ACES tonemapping"""
        A, B, C, D, E = 2.51, 0.03, 2.43, 0.59, 0.14
        return (
            (D * color - B)
            + np.sqrt(
                4.0 * A * E * color
                + B * B
                - 2.0 * B * D * color
                - 4.0 * C * E * color * color
                + D * D * color * color
            )
        ) / (2.0 * (A - C * color))

    def _tonemap_uncharted2_apply(self, color):
        """Uncharted 2 Filmic tonemapping"""
        A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
        return (
            (color * (A * color + C * B) + D * E) / (color * (A * color + B) + D * F)
        ) - E / F

    def _tonemap_uncharted2_inverse(self, color):
        """Inverse Uncharted 2 Filmic tonemapping"""
        A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
        return np.abs(
            (
                (B * C * F - B * E - B * F * color)
                - np.sqrt(
                    np.abs(-B * C * F + B * E + B * F * color) ** 2.0
                    - 4.0 * D * (F * F) * color * (A * E + A * F * color - A * F)
                )
            )
            / (2.0 * A * (E + F * color - F))
        )

    def _tonemap_baking_lab_aces_apply(self, color):
        """Baking Lab ACES tonemapping"""
        A, B, C, D, E = 0.0245786, 0.000090537, 0.983729, 0.4329510, 0.238081
        return np.clip((color * (color + A) - B) / (color * (C * color + D) + E), 0, 1)

    def _tonemap_baking_lab_aces_inverse(self, color):
        """Inverse Baking Lab ACES tonemapping"""
        A, B, C, D, E = 0.0245786, 0.000090537, 0.983729, 0.4329510, 0.238081
        return np.abs(
            (
                (A - D * color)
                - np.sqrt(
                    np.abs(D * color - A) ** 2.0
                    - 4.0 * (C * color - 1.0) * (B + E * color)
                )
            )
            / (2.0 * (C * color - 1.0))
        )

    def _apply_inverse_tonemap(self, color):
        """Apply inverse tonemapping"""
        if self.inv_tonemap == TONEMAP_REINHARD:
            color = self._tonemap_reinhard_inverse(color)
        elif self.inv_tonemap == TONEMAP_LOTTES:
            color = self._tonemap_lottes_inverse(color)
        elif self.inv_tonemap == TONEMAP_UNREAL3:
            color = self._tonemap_unreal3_inverse(color)
        elif self.inv_tonemap == TONEMAP_NARKOWICZ_ACES:
            color = self._tonemap_narkowicz_aces_inverse(color)
        elif self.inv_tonemap == TONEMAP_UNCHARTED2_FILMIC:
            color = self._tonemap_uncharted2_inverse(color)
        elif self.inv_tonemap == TONEMAP_BAKING_LAB_ACES:
            color = self._tonemap_baking_lab_aces_inverse(color)

        # Apply input exposure
        color /= np.exp(self.input_exposure)

        return color

    def _apply_tonemap(self, color):
        """Apply tonemapping"""
        # Apply output exposure
        exposure = np.exp(self.exposure)
        color = color * exposure

        if self.tonemap == TONEMAP_REINHARD:
            color = self._tonemap_reinhard_apply(color)
        elif self.tonemap == TONEMAP_LOTTES:
            color = self._tonemap_lottes_apply(color)
        elif self.tonemap == TONEMAP_UNREAL3:
            color = self._tonemap_unreal3_apply(color)
        elif self.tonemap == TONEMAP_NARKOWICZ_ACES:
            color = self._tonemap_narkowicz_aces_apply(color)
        elif self.tonemap == TONEMAP_UNCHARTED2_FILMIC:
            color = self._tonemap_uncharted2_apply(color)
        elif self.tonemap == TONEMAP_BAKING_LAB_ACES:
            color = self._tonemap_baking_lab_aces_apply(color)

        return color

    def _apply_saturation(self, color, saturation):
        """Apply saturation adjustment"""
        luma = np.sum(color * np.array([0.2126, 0.7152, 0.0722]), axis=2, keepdims=True)
        return luma + (color - luma) * saturation

    def _normal_distribution(self, x, mean, variance):
        """Normal distribution for bloom blending"""
        return np.exp(-((x - mean) ** 2) / (2 * variance**2))

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Apply MagicHDR filter

        Args:
            image: RGB image (uint8, 0-255)
            **params: Filter parameters
                - input_exposure: Input exposure f-stops
                    (-3.0~3.0, default 0.0)
                - exposure: Output exposure f-stops
                    (-3.0~3.0, default 0.0)
                - inv_tonemap: Inverse tonemap type (0~5, default 0)
                - tonemap: Output tonemap type (0~5, default 5)
                - bloom_amount: Bloom amount (0.0~1.0, default 0.3)
                - bloom_brightness: Bloom brightness (1.0~5.0, default 3.0)
                - bloom_saturation: Bloom saturation (0.0~2.0, default 1.0)
                - blur_size: Blur size (0.01~1.0, default 0.5)
                - blending_amount: Blending amount (0.1~1.0, default 0.5)
                - blending_base: Blending base (0.0~1.0, default 0.8)
                - show_bloom: Show bloom only (True/False, default False)

        Returns:
            HDR bloomed image (uint8, 0-255)
        """
        # Update parameters
        self.input_exposure = params.get("input_exposure", self.input_exposure)
        self.exposure = params.get("exposure", self.exposure)
        self.inv_tonemap = params.get("inv_tonemap", self.inv_tonemap)
        self.tonemap = params.get("tonemap", self.tonemap)
        self.bloom_amount = params.get("bloom_amount", self.bloom_amount)
        self.bloom_brightness = params.get("bloom_brightness", self.bloom_brightness)
        self.bloom_saturation = params.get("bloom_saturation", self.bloom_saturation)
        self.blur_size = params.get("blur_size", self.blur_size)
        self.blending_amount = params.get("blending_amount", self.blending_amount)
        self.blending_base = params.get("blending_base", self.blending_base)
        self.show_bloom = params.get("show_bloom", self.show_bloom)

        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        original = img_float.copy()

        # Apply inverse tonemapping
        img_hdr = self._apply_inverse_tonemap(img_float)

        # Apply saturation
        saturation = (
            self.bloom_saturation**2
            if self.bloom_saturation > 1.0
            else self.bloom_saturation
        )
        img_hdr = self._apply_saturation(img_hdr, saturation)
        img_hdr = np.clip(img_hdr, 0, None)

        # Apply brightness
        img_hdr *= np.exp(self.bloom_brightness)

        # Create 7 bloom levels with downsampling
        bloom_levels = []
        current = img_hdr
        for i in range(7):
            # Apply Gaussian blur
            sigma = self.blur_size * (2**i) * 2.0
            blurred = np.zeros_like(current)
            for c in range(3):
                blurred[:, :, c] = gaussian_filter(
                    current[:, :, c], sigma=sigma, mode="reflect"
                )
            bloom_levels.append(blurred)

            # Downsample for next level
            if i < 6:
                h, w = current.shape[:2]
                current = current[::2, ::2]

        # Upsample and blend bloom levels
        h, w = img_hdr.shape[:2]
        for i in range(1, 7):
            # Upsample to original size (simple nearest neighbor)
            level_h, level_w = bloom_levels[i].shape[:2]

            # Create upsampled version
            upsampled = np.zeros((h, w, 3), dtype=np.float32)
            scale_y = level_h / h
            scale_x = level_w / w

            for y in range(h):
                for x in range(w):
                    sy = int(y * scale_y)
                    sx = int(x * scale_x)
                    sy = min(sy, level_h - 1)
                    sx = min(sx, level_w - 1)
                    upsampled[y, x] = bloom_levels[i][sy, sx]

            bloom_levels[i] = upsampled

        # Combine bloom levels using normal distribution weights
        mean = self.blending_base * 7
        variance = self.blending_amount * 7

        bloom = np.zeros_like(img_hdr)
        total_weight = 0
        for i in range(7):
            weight = self._normal_distribution(i + 1, mean, variance)
            bloom += bloom_levels[i] * weight
            total_weight += weight

        bloom /= total_weight

        # Apply inverse tonemapping to original for final blend
        original_hdr = self._apply_inverse_tonemap(original)

        # Blend bloom with original
        if self.show_bloom:
            color = bloom
        else:
            color = original_hdr + (bloom - original_hdr) * np.log10(
                self.bloom_amount + 1.0
            )

        # Apply output tonemapping
        result = self._apply_tonemap(color)

        # Clamp and convert
        result = np.clip(result, 0, 1)
        return (result * 255).astype(np.uint8)
