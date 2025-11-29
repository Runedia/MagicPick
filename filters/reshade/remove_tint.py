import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import dot, length, lerp, normalize, saturate


class RemoveTintFilter(BaseFilter):
    """
    RemoveTint.fx implementation
    Author: Daodan
    Python implementation for Gemini Image Editor
    """

    def __init__(self):
        super().__init__("RemoveTint", "틴트 제거")

        # Default parameters
        self.fUISpeed = 0.1  # Not used for static image processing
        self.bUIUseExcludeColor = False
        self.fUIExcludeColor = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.fUIExcludeColorStrength = 3.0
        self.cUIDebug = 0  # Not implemented (debug modes)
        self.fUIStrength = 1.0

        # Internal state for temporal blending, though not fully used for static images
        self._last_min_rgb = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._last_max_rgb = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    def apply(self, image, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        img_float = image.astype(np.float32) / 255.0

        # OpenCV BGR to RGB for processing
        rgb = img_float[:, :, ::-1]

        # --- Pass 1: Simulate texBackBuffer (downscaled version for min/max calc) ---
        # REMOVE_TINT_MIPLEVEL_EXP2 = 16
        # Downscale the image for faster min/max calculation, as in the shader

        # We need the full resolution for final application, but min/max are taken from downscaled
        # Let's take global min/max for now, as subsampling is tricky without proper mipmapping or complex averaging.
        # The shader explicitly loops through downscaled texture, which in Python means iterating
        # over a subsampled image. Given the constraints and typical use-case, global min/max is acceptable.

        # For simplicity and efficiency in Python, we'll calculate min/max on the full image
        # or a slightly downscaled version without explicitly mimicking the mip-level approach,
        # focusing on the logical min/max derivation.

        # --- Pass 2: MinMaxRGB_PS (Global min/max calculation with exclusion) ---

        # currentMinRGB and currentMaxRGB logic
        current_min_rgb = np.ones(3, dtype=np.float32)
        current_max_rgb = np.zeros(3, dtype=np.float32)

        # Reshape to (N, 3) for easier per-pixel iteration/vectorization
        flat_rgb = rgb.reshape(-1, 3)

        if self.bUIUseExcludeColor:
            # Calculate lerpValue for each pixel
            # diff = saturate(pow(dot(color, fUIExcludeColor), fUIExcludeColorStrength));
            dot_product = dot(flat_rgb, self.fUIExcludeColor)
            diff = saturate(np.power(dot_product, self.fUIExcludeColorStrength))
            lerp_value = 1.0 - diff  # 0 for excluded, 1 for not excluded

            # Filter pixels based on lerp_value (effectively, exclude pixels where lerp_value is low)
            # Threshold chosen to be slightly above 0, as shader uses lerp() where if C is 0, A is chosen.
            # If C is 0, currentMin/MaxRGB remains unchanged.
            threshold = 0.01  # Effectively ignoring very excluded pixels
            eligible_pixels = flat_rgb[lerp_value > threshold]

            if eligible_pixels.size > 0:
                current_min_rgb = np.min(eligible_pixels, axis=0)
                current_max_rgb = np.max(eligible_pixels, axis=0)
            else:  # If all pixels are excluded (e.g., solid fUIExcludeColor image)
                current_min_rgb = np.zeros(3)  # Default to black
                current_max_rgb = np.ones(3)  # Default to white
        else:
            current_min_rgb = np.min(rgb, axis=(0, 1))
            current_max_rgb = np.max(rgb, axis=(0, 1))

        # --- Pass 3 & 4 (Simplified for static image): Store and update min/max ---
        # For static images, frametime is not dynamic. We use the calculated current min/max.
        # The shader's temporal blending and last frame storage are for real-time video.
        # So _last_min_rgb and _last_max_rgb can simply be updated with current frame's values.

        self._last_min_rgb = current_min_rgb
        self._last_max_rgb = current_max_rgb

        # --- Pass 4: Apply_PS (Apply Tint Removal) ---
        MinRGB = self._last_min_rgb
        MaxRGB = self._last_max_rgb

        # Avoid division by zero if MaxRGB == MinRGB
        denom = MaxRGB - MinRGB
        denom = np.where(denom == 0, 1e-6, denom)  # Replace zeros with small epsilon

        # Color Normalize: (color - MinRGB) / (MaxRGB-MinRGB)
        color_normalize = (rgb - MinRGB) / denom
        tint_removed = color_normalize

        # Preserve brightness: tintRemoved = normalize(tintRemoved) * length(color).rrr
        # length(color) is the Euclidean norm for RGB values, not luma.
        len_rgb = length(rgb)  # Returns (H, W, 1) or (H, W) if dot product is used

        # Normalize and then scale by original color's length.
        # Need to handle zero length (black pixels) to avoid NaNs
        norm_tint_removed = np.where(
            length(tint_removed)[..., np.newaxis] > 1e-6, normalize(tint_removed), 0
        )
        tint_removed = (
            norm_tint_removed * len_rgb[..., np.newaxis]
        )  # Reshape len_rgb for broadcasting

        # Don't apply tint removal to excluded colors
        # lerpValue = saturate(pow(dot(color, fUIExcludeColor), fUIExcludeColorStrength))
        # This lerp is inverted compared to MinMaxRGB_PS. If lerpValue is 1.0 (excluded), it lerps to original color.
        dot_product_final = dot(rgb, self.fUIExcludeColor)
        lerp_value_final = saturate(
            np.power(dot_product_final, self.fUIExcludeColorStrength)
        )  # 1 for excluded, 0 for not

        # lerp(tintRemoved, color, lerp_value_final)
        tint_removed = lerp(tint_removed, rgb, lerp_value_final[..., np.newaxis])

        # Blend with original image by fUIStrength
        final_rgb = lerp(rgb, tint_removed, self.fUIStrength)

        final_rgb = saturate(final_rgb)

        # RGB to BGR for output
        result_bgr = final_rgb[:, :, ::-1]

        return (result_bgr * 255).astype(np.uint8)
