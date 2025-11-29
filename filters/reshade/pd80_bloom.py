import cv2
import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import saturate
from filters.reshade.pd80_chromatic_aberration import PD80ChromaticAberrationFilter


class PD80BloomFilter(BaseFilter):
    """
    PD80_02_Bloom.fx 구현

    고품질 블룸 효과를 제공합니다.
    밝은 영역을 추출하여 블러링하고, 선택적으로 색수차를 적용한 후 원본과 합성합니다.
    """

    def __init__(self):
        super().__init__("PD80Bloom", "PD80 블룸")
        self.bloom_mix = 0.5
        self.bloom_limit = 0.333
        self.blur_sigma = 30.0
        self.exposure = 0.0
        self.bloom_saturation = 0.0
        self.debug_bloom = False

        # CA params (Optional)
        self.enable_ca = False
        self.ca_strength = 0.5
        self.ca_width = 60.0

        # Focus Bloom (Optional)
        self.use_focus_bloom = False
        self.focus_bloom_strength = 0.25
        self.blur_sigma_narrow = 15.0

        # Helper CA filter instance
        self.ca_filter = PD80ChromaticAberrationFilter()

    def _get_luminance(self, img):
        # R, G, B weights
        return np.dot(img, [0.212656, 0.715158, 0.072186])

    def _screen(self, c, b):
        return 1.0 - (1.0 - c) * (1.0 - b)

    def _vib(self, color, saturation):
        # Simple vibrance/saturation adjust
        # ReShade's vib function usually:
        # luma = getLuminance(color)
        # color = lerp(luma, color, 1.0 + saturation)

        if saturation == 0.0:
            return color

        luma = self._get_luminance(color)
        luma = np.expand_dims(luma, axis=2)

        return luma + (color - luma) * (1.0 + saturation)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        if "bloom_mix" in params:
            self.bloom_mix = float(params["bloom_mix"])
        if "bloom_limit" in params:
            self.bloom_limit = float(params["bloom_limit"])
        if "blur_sigma" in params:
            self.blur_sigma = float(params["blur_sigma"])
        if "exposure" in params:
            self.exposure = float(params["exposure"])
        if "bloom_saturation" in params:
            self.bloom_saturation = float(params["bloom_saturation"])
        if "debug_bloom" in params:
            self.debug_bloom = bool(params["debug_bloom"])

        if "enable_ca" in params:
            self.enable_ca = bool(params["enable_ca"])
        if "ca_strength" in params:
            self.ca_strength = float(params["ca_strength"])

        img_float = image.astype(np.float32) / 255.0
        height, width = img_float.shape[:2]

        # 1. Thresholding & Exposure
        luma = self._get_luminance(img_float)
        luma = np.maximum(luma, 0.000001)

        # Bloom Limit
        # Using simplified logic compared to shader's CalcExposedColor which depends on avgLuma
        # We'll use simple thresholding logic: max(0, color - limit)

        # Shader:
        # color.xyz = saturate( color.xyz - luma ) / saturate( 1.0f - luma ); ? No this is different
        # color.xyz = saturate( color.xyz - BloomLimit ) ... in concept

        # Actual shader PS_BloomIn:
        # luma = tex2D(samplerBAvgLuma).x (Average Luma)
        # color = saturate(color - luma) / saturate(1.0 - luma)  <-- Thresholding based on Avg Luma?
        # color = CalcExposedColor(...)

        # Since we don't track AvgLuma over time, we use BloomLimit as threshold
        threshold = self.bloom_limit

        # Soft thresholding
        # extract bright parts
        bright_part = np.maximum(img_float - threshold, 0.0)
        # re-scale? Shader does exposure adjustment
        bright_part *= 2.0**self.exposure

        # 2. Blur (Bloom)
        # Wide Bloom
        # Sigma scaling
        sigma_w = self.blur_sigma * (max(width, height) / 1920.0)
        bloom_wide = cv2.GaussianBlur(
            bright_part, (0, 0), sigmaX=sigma_w, sigmaY=sigma_w
        )

        bloom = bloom_wide

        # Focus Bloom
        if self.use_focus_bloom:
            sigma_n = self.blur_sigma_narrow * (max(width, height) / 1920.0)
            bloom_narrow = cv2.GaussianBlur(
                bright_part, (0, 0), sigmaX=sigma_n, sigmaY=sigma_n
            )
            # Mix
            bloom = (
                bloom_wide * (1.0 - self.focus_bloom_strength)
                + bloom_narrow * self.focus_bloom_strength
            )

        # 3. CA (Optional)
        if self.enable_ca:
            # Reuse PD80ChromaticAberrationFilter logic
            # Convert bloom to uint8 for filter input
            bloom_uint8 = (np.clip(bloom, 0.0, 1.0) * 255).astype(np.uint8)

            # Setup CA params
            ca_params = {
                "ca_strength": self.ca_strength,
                "ca_global_width": self.ca_width,
                "sample_steps": 3,  # Low steps for performance
                "ca_type": 0,  # Radial
            }
            bloom_ca = self.ca_filter.apply(bloom_uint8, **ca_params)
            bloom = bloom_ca.astype(np.float32) / 255.0

        # 4. Vibrance/Saturation
        bloom = self._vib(bloom, self.bloom_saturation)
        bloom = saturate(bloom)

        # 5. Combine
        # bcolor = screen( color.xyz, bloom.xyz );
        # color.xyz = lerp( color.xyz, bcolor.xyz, BloomMix );

        b_color = self._screen(img_float, bloom)
        result = img_float * (1.0 - self.bloom_mix) + b_color * self.bloom_mix

        if self.debug_bloom:
            result = bloom

        return (np.clip(result, 0.0, 1.0) * 255).astype(np.uint8)
