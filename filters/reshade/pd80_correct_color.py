import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import lerp, saturate


class PD80_CorrectColor(BaseFilter):
    """
    PD80_01B_RT_Correct_Color.fx implementation
    Author: prod80
    Python implementation for Gemini Image Editor
    """

    def __init__(self):
        super().__init__("PD80CorrectColor", "색상 보정")

        # Default parameters
        self.rt_enable_whitepoint_correction = True
        self.rt_whitepoint_respect_luma = True
        self.rt_whitepoint_method = 0  # 0: By Channel, 1: Find Light Color
        self.rt_wp_str = 1.0
        self.rt_wp_rl_str = 1.0

        self.rt_enable_blackpoint_correction = True
        self.rt_blackpoint_respect_luma = False
        self.rt_blackpoint_method = 1  # 0: By Channel, 1: Find Dark Color
        self.rt_bp_str = 1.0
        self.rt_bp_rl_str = 1.0

        self.rt_enable_midpoint_correction = True
        self.rt_midpoint_respect_luma = True
        self.mid_use_alt_method = True
        self.midCC_scale = 0.5

        self.enable_dither = True
        self.dither_strength = 1.0

        if params:
            for k, v in params.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def apply(self, image, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        img_float = image.astype(np.float32) / 255.0

        # Dither (Simplified noise addition)
        if self.enable_dither and self.dither_strength > 0:
            noise = np.random.normal(
                0, 0.001 * self.dither_strength, img_float.shape
            ).astype(np.float32)
            img_float = saturate(img_float + noise)

        # --- Analyze Image Statistics ---
        # In the shader, this is done via downscaling.
        # Here we can compute statistics on the whole image or a downscaled version for speed.
        # Given Python/NumPy, computing on full image is fine for accuracy, or subsample for speed.
        # Let's use full image for quality, unless it's too slow (not expected for single image edit).

        # Min/Max/Mid calculation logic from PS_MinMax_1

        # Calculate per-channel min/max
        min_c = np.min(img_float, axis=(0, 1))  # minMethod0
        max_c = np.max(img_float, axis=(0, 1))  # maxMethod0

        # Calculate "By Color" min/max (finding the pixel with highest/lowest sum)
        # sum(rgb) + max(r, g, b) logic from shader seems complex to replicate efficiently without loop?
        # Shader: getMin = max(max(r,g),b) + dot(rgb, 1.0)
        # Let's vectorize this score.

        rgb_sum = np.sum(img_float, axis=2)
        rgb_max_val = np.max(img_float, axis=2)
        score_min = rgb_max_val + rgb_sum
        score_max = rgb_sum  # Shader: getMax = dot(rgb, 1.0)

        # Find index of min/max score
        # Note: This is finding the "darkest color" and "lightest color" vector in the image.

        # Flatten for argmin/argmax
        flat_img = img_float.reshape(-1, 3)
        flat_score_min = score_min.flatten()
        flat_score_max = score_max.flatten()

        idx_min = np.argmin(flat_score_min)
        idx_max = np.argmax(flat_score_max)

        min_color = flat_img[idx_min]  # minMethod1
        max_color = flat_img[idx_max]  # maxMethod1

        # Select method
        minValue = min_color if self.rt_blackpoint_method == 1 else min_c
        maxValue = max_color if self.rt_whitepoint_method == 1 else max_c

        # --- Midpoint Calculation ---
        # Shader calculates midValue based on `middle` reference.
        # middle = dot( float2( dot( prevMin, 0.33 ), dot( prevMax, 0.33 ) ), 0.5 )
        # using current min/max instead of prev

        avg_min = np.mean(minValue)
        avg_max = np.mean(maxValue)
        middle = 0.5 * (avg_min + avg_max)

        if not self.mid_use_alt_method:
            middle = 0.5

        # Find color closest to `middle` intensity
        # Shader: getMid = dot( abs( currColor - middle ), 1.0 )
        # Find pixel with min distance to middle gray

        dist_mid = np.sum(np.abs(img_float - middle), axis=2)
        flat_dist_mid = dist_mid.flatten()
        idx_mid = np.argmin(flat_dist_mid)
        midValue = flat_img[idx_mid]

        # --- PS_RemoveTint Logic ---

        # Set min value
        # minValue = lerp(0.0, minValue, rt_bp_str)
        effective_min = lerp(np.zeros(3), minValue, self.rt_bp_str)
        if not self.rt_enable_blackpoint_correction:
            effective_min = np.zeros(3)

        # Set max value
        # maxValue = lerp(1.0, maxValue, rt_wp_str)
        effective_max = lerp(np.ones(3), maxValue, self.rt_wp_str)
        if not self.rt_enable_whitepoint_correction:
            effective_max = np.ones(3)

        # Set mid value
        # midValue = midValue - middle
        # midValue *= midCC_scale
        effective_mid = (midValue - middle) * self.midCC_scale
        if not self.rt_enable_midpoint_correction:
            effective_mid = np.zeros(3)

        # Main color correction
        # color = saturate(color - min) / (max - min)
        denom = effective_max - effective_min
        # Avoid div by zero
        denom = np.maximum(denom, 1e-6)

        # Broadcasting correction
        result = saturate(img_float - effective_min) / denom

        # White Point luma preservation
        # avgMax = dot(maxValue, 0.333) -> using effective_max? Shader uses `maxValue` (local var) which matches `effective_max` logic path?
        # Actually shader: `maxValue` is overwritten by lerp logic earlier. So yes, effective_max.

        avgMax = np.mean(effective_max)
        # color = lerp( color, color * avgMax, rt_whitepoint_respect_luma * rt_wp_rl_str );
        target_wp = result * avgMax
        factor_wp = float(self.rt_whitepoint_respect_luma) * self.rt_wp_rl_str
        result = lerp(result, target_wp, factor_wp)

        # Black Point luma preservation
        avgMin = np.mean(effective_min)
        # color = lerp( color, color * (1.0 - avgMin) + avgMin, ... )
        target_bp = result * (1.0 - avgMin) + avgMin
        factor_bp = float(self.rt_blackpoint_respect_luma) * self.rt_bp_rl_str
        result = lerp(result, target_bp, factor_bp)

        # Mid Point correction
        avgCol = np.mean(result, axis=2, keepdims=True)  # Avg after main correction
        avgMid = np.mean(effective_mid)

        # avgCol = 1.0 - abs( avgCol * 2.0 - 1.0 )
        weight = 1.0 - np.abs(avgCol * 2.0 - 1.0)

        # color = saturate( color - midValue * avgCol + avgMid * avgCol * rt_midpoint_respect_luma )
        mid_correction = effective_mid * weight  # (H,W,3)
        mid_luma_correction = avgMid * weight * float(self.rt_midpoint_respect_luma)

        result = saturate(result - mid_correction + mid_luma_correction)

        return (result * 255).astype(np.uint8)
