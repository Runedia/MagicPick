import cv2
import numpy as np

from filters.base_filter import BaseFilter


class FilmicAnamorphSharpenFilter(BaseFilter):
    """
    FilmicAnamorphSharpen.fx 구현

    영화 같은 느낌의 아나모픽 샤프닝 효과를 제공합니다.
    루마 기반의 하이패스 필터를 사용하여 디테일을 강조합니다.
    """

    def __init__(self):
        super().__init__("FilmicAnamorphSharpen", "필름 아나모픽 샤프닝")
        self.strength = 60.0
        self.offset = 0.1
        self.clamp_val = 0.65
        self.use_mask = False
        self.preview = False

    def _overlay(self, layer_a, layer_b):
        # Shader Overlay function:
        # MinA = min(LayerA, 0.5); MinB = min(LayerB, 0.5);
        # MaxA = max(LayerA, 0.5); MaxB = max(LayerB, 0.5);
        # return 2f*(MinA*MinB+MaxA+MaxB-MaxA*MaxB)-1.5;

        min_a = np.minimum(layer_a, 0.5)
        min_b = np.minimum(layer_b, 0.5)
        max_a = np.maximum(layer_a, 0.5)
        max_b = np.maximum(layer_b, 0.5)

        return 2.0 * (min_a * min_b + max_a + max_b - max_a * max_b) - 1.5

    def _gamma_to_linear(self, x):
        # Simple sRGB approx
        return np.power(x, 2.2)

    def _gamma_to_display(self, x):
        return np.power(np.maximum(x, 0.0), 1.0 / 2.2)

    def _get_luma(self, rgb):
        # Rec.709
        return np.dot(rgb, [0.2126, 0.7152, 0.0722])

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        if "strength" in params:
            self.strength = float(params["strength"])
        if "offset" in params:
            self.offset = float(params["offset"])
        if "clamp_val" in params:
            self.clamp_val = float(params["clamp_val"])
        if "use_mask" in params:
            self.use_mask = bool(params["use_mask"])
        if "preview" in params:
            self.preview = bool(params["preview"])

        img_float = image.astype(np.float32) / 255.0
        rows, cols = img_float.shape[:2]

        # Linearize
        src_linear = self._gamma_to_linear(img_float)
        src_luma = self._get_luma(src_linear)

        # Shifted samples
        # Offset is in pixels.
        # Using warpAffine for fractional offset
        M_up = np.float32([[1, 0, 0], [0, 1, -self.offset]])
        M_down = np.float32([[1, 0, 0], [0, 1, self.offset]])
        M_left = np.float32([[1, 0, -self.offset], [0, 1, 0]])
        M_right = np.float32([[1, 0, self.offset], [0, 1, 0]])

        # Use BORDER_REFLECT or REPLICATE
        flags = cv2.INTER_LINEAR
        border = cv2.BORDER_REFLECT_101

        # Need to shift RGB then convert to Luma, or shift Luma directly?
        # Shader: Sample RGB -> Linear -> Luma.
        # Shifting Luma directly is equivalent if we assume linearity and independent shifting.
        # Let's shift Luma directly for performance.

        luma_up = cv2.warpAffine(
            src_luma, M_up, (cols, rows), flags=flags, borderMode=border
        )
        luma_down = cv2.warpAffine(
            src_luma, M_down, (cols, rows), flags=flags, borderMode=border
        )
        luma_left = cv2.warpAffine(
            src_luma, M_left, (cols, rows), flags=flags, borderMode=border
        )
        luma_right = cv2.warpAffine(
            src_luma, M_right, (cols, rows), flags=flags, borderMode=border
        )

        high_pass_sum = luma_up + luma_down + luma_left + luma_right

        # HighPassColor = 0.5 - 0.5 * (HighPassColor * 0.25 - SourceLuma)
        high_pass = 0.5 - 0.5 * (high_pass_sum * 0.25 - src_luma)

        # Masking
        mask_val = self.strength
        if self.use_mask:
            # Generate radial mask
            # Mask = 1f - length(UvCoord * 2f - 1f)
            # Mask = Overlay(Mask) * Strength
            y, x = np.indices((rows, cols))
            u = x / cols
            v = y / rows

            dist = np.sqrt((u * 2.0 - 1.0) ** 2 + (v * 2.0 - 1.0) ** 2)
            mask_rad = 1.0 - dist

            # Shader Overlay for mask takes one arg?
            # float Overlay(float LayerAB) { ... return 2f*(MinAB*MinAB+MaxAB+MaxAB-MaxAB*MaxAB)-1.5; }
            # It's actually Overlay(LayerAB, LayerAB) simplified?
            # Let's use the same overlay function with same input.
            mask_overlay = self._overlay(mask_rad, mask_rad)
            mask_val = mask_overlay * self.strength

            # if Mask <= 0 bypass. In python we use mask_val as lerp factor.
            mask_val = np.maximum(mask_val, 0.0)

        # Lerp
        # HighPassColor = lerp(0.5, HighPassColor, Mask)
        high_pass = 0.5 * (1.0 - mask_val) + high_pass * mask_val

        # Clamping
        # HighPassColor = clamp(HighPassColor, 1.0 - Clamp, Clamp)
        high_pass = np.clip(high_pass, 1.0 - self.clamp_val, self.clamp_val)

        # Preview
        if self.preview:
            # return HighPassColor
            res = self._gamma_to_display(high_pass)
            # Expand to 3 channels
            res = np.dstack([res, res, res])
            return (np.clip(res, 0.0, 1.0) * 255).astype(np.uint8)

        # Sharpen
        # Overlay(Source, HighPass)
        # Need to expand HighPass to 3 channels for broadcasting?
        high_pass_3d = np.expand_dims(high_pass, axis=2)

        sharpen_r = self._overlay(src_linear[:, :, 0], high_pass)
        sharpen_g = self._overlay(src_linear[:, :, 1], high_pass)
        sharpen_b = self._overlay(src_linear[:, :, 2], high_pass)

        sharpen = np.dstack([sharpen_r, sharpen_g, sharpen_b])

        result = self._gamma_to_display(sharpen)

        return (np.clip(result, 0.0, 1.0) * 255).astype(np.uint8)
