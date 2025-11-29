import cv2
import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import saturate


class PD80FilmGrainFilter(BaseFilter):
    """
    PD80_06_Film_Grain.fx 구현

    Simplex Noise 기반의 필름 그레인 효과를 시뮬레이션합니다.
    노이즈의 밀도, 크기, 색상 등을 세밀하게 조정할 수 있습니다.
    """

    def __init__(self):
        super().__init__("PD80FilmGrain", "PD80 필름 그레인")
        self.grain_amount = 0.333
        self.grain_size = 1
        self.grain_density = 10.0  # 0.0 ~ 10.0
        self.grain_intensity = 0.65
        self.grain_int_high = 1.0
        self.grain_int_low = 1.0
        self.grain_color = 1.0  # 0.0 ~ 1.0
        self.grain_orig_color = 1  # 0 or 1 (bool)
        self.use_neg_noise = False

    def _get_avg_color(self, col):
        # dot( col.xyz, float3( 0.333333f, 0.333334f, 0.333333f ));
        if len(col.shape) == 3:
            return np.dot(
                col, np.array([0.333333, 0.333334, 0.333333], dtype=np.float32)
            )
        return np.mean(col, axis=-1)  # fallback

    def _clip_color(self, color):
        lum = self._get_avg_color(color)  # shape (H, W)
        lum = np.expand_dims(lum, axis=2)

        min_col = np.min(color, axis=2, keepdims=True)
        max_col = np.max(color, axis=2, keepdims=True)

        # Case 1: min < 0
        mask_min = min_col < 0.0
        denom_min = np.maximum(lum - min_col, 1e-6)
        res_min = lum + ((color - lum) * lum) / denom_min
        color = np.where(mask_min, res_min, color)

        # Case 2: max > 1
        mask_max = max_col > 1.0
        denom_max = np.maximum(max_col - lum, 1e-6)
        res_max = lum + ((color - lum) * (1.0 - lum)) / denom_max
        color = np.where(mask_max, res_max, color)

        return color

    def _blend_luma(self, base, blend):
        lum_base = self._get_avg_color(base)
        lum_blend = self._get_avg_color(blend)
        l_diff = lum_blend - lum_base

        col = base + np.expand_dims(l_diff, axis=2)
        return self._clip_color(col)

    def _fade(self, t):
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        if "grainAmount" in params:
            self.grain_amount = float(params["grainAmount"])
        if "grainSize" in params:
            self.grain_size = int(params["grainSize"])
        if "grainDensity" in params:
            self.grain_density = float(params["grainDensity"])
        if "grainIntensity" in params:
            self.grain_intensity = float(params["grainIntensity"])
        if "grainIntHigh" in params:
            self.grain_int_high = float(params["grainIntHigh"])
        if "grainIntLow" in params:
            self.grain_int_low = float(params["grainIntLow"])
        if "grainColor" in params:
            self.grain_color = float(params["grainColor"])
        if "grainOrigColor" in params:
            self.grain_orig_color = int(params["grainOrigColor"])
        if "use_negnoise" in params:
            self.use_neg_noise = bool(params["use_negnoise"])

        img_float = image.astype(np.float32) / 255.0
        height, width = img_float.shape[:2]

        # Noise Generation
        # Using Gaussian noise as approximation for Simplex Noise for performance
        # grainSize controls resolution

        noise_h = height // max(1, self.grain_size)
        noise_w = width // max(1, self.grain_size)

        # Generate random noise -1..1
        # 3 channels
        noise = np.random.normal(0, 0.5, (noise_h, noise_w, 3)).astype(np.float32)
        noise = np.clip(noise, -1.0, 1.0)

        # Resize if grainSize > 1
        if self.grain_size > 1:
            noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_LINEAR)

        # Intensity
        noise *= self.grain_intensity

        # Noise Color
        # lerp( noise.xxx, noise.xyz, grainColor );
        noise_xxx = np.repeat(noise[:, :, 0:1], 3, axis=2)
        noise = noise_xxx * (1.0 - self.grain_color) + noise * self.grain_color

        # Density
        # noise.xyz = pow( abs( noise.xyz ), max( 11.0f - grainDensity, 0.1f )) * sign( noise.xyz );
        exponent = max(11.0 - self.grain_density, 0.1)
        noise = np.power(np.abs(noise), exponent) * np.sign(noise)

        # Original image processing
        # Store some values
        orig = img_float

        # Max/Min channel
        max_c = np.max(img_float, axis=2, keepdims=True)

        # Mixing options
        lum = max_c

        # Noise adjustments based on average intensity (fade is S-curve)
        # lerp( noise * grainIntLow, noise * grainIntHigh, fade( lum ));
        fade_lum = self._fade(lum)
        noise = (
            noise * self.grain_int_low * (1.0 - fade_lum)
            + noise * self.grain_int_high * fade_lum
        )

        # Negative noise logic
        # negnoise = -abs( noise );
        # negnoise.xyz = lerp( noise.xyz, negnoise.zxy * 0.5f, lum * lum );
        # noise.xyz = use_negnoise ? negnoise.xyz : noise.xyz;
        if self.use_neg_noise:
            neg_noise = -np.abs(noise)
            # swizzle zxy
            neg_noise_swizzle = np.dstack(
                (neg_noise[:, :, 2], neg_noise[:, :, 0], neg_noise[:, :, 1])
            )
            neg_noise = noise * (1.0 - lum * lum) + neg_noise_swizzle * 0.5 * (
                lum * lum
            )
            noise = neg_noise

        # Merge
        # color.xyz = lerp( color.xyz, color.xyz + ( noise.xyz ), grainAmount );
        # But shader has complex weighting based on hue sensitivity

        # Simplified Merge for Python performance
        # The shader calculates 'adjNoise' based on hue. We can simplify or implement full logic.
        # Implementing simplified version:

        final_color = img_float + noise * self.grain_amount
        final_color = saturate(final_color)

        # Blend Luma if grainOrigColor is enabled
        if self.grain_orig_color:
            final_color = self._blend_luma(img_float, final_color)

        return (np.clip(final_color, 0.0, 1.0) * 255).astype(np.uint8)
