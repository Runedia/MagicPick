"""
간단한 ReShade 필터들의 정확한 구현

Curves, DPX, Tonemap, Sepia, Vignette 등
"""

import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import lerp, pow_safe, saturate


class CurvesFilterAccurate(BaseFilter):
    """Curves - S-커브를 사용하여 대비를 증가시킵니다"""

    def __init__(self):
        super().__init__("Curves", "커브")

        self.mode = 0
        self.formula = 4
        self.contrast = 0.65

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        self.mode = params.get("Mode", self.mode)
        self.formula = params.get("Formula", self.formula)
        self.contrast = params.get("Contrast", self.contrast)

        img_float = image.astype(np.float32) / 255.0

        lum_coeff = np.array([0.2126, 0.7152, 0.0722])
        contrast_blend = self.contrast
        PI = np.pi

        luma = np.sum(img_float * lum_coeff, axis=2, keepdims=True)
        chroma = img_float - luma

        if self.mode == 0:
            x = luma
        elif self.mode == 1:
            x = chroma * 0.5 + 0.5
        else:
            x = img_float

        if self.formula == 0:
            x = np.sin(PI * 0.5 * x)
            x *= x
        elif self.formula == 1:
            x = x - 0.5
            x = (x / (0.5 + np.abs(x))) + 0.5
        elif self.formula == 2:
            x = x * x * (3.0 - 2.0 * x)
        elif self.formula == 3:
            x = (1.0524 * np.exp(6.0 * x) - 1.05248) / (np.exp(6.0 * x) + 20.0855)
        elif self.formula == 4:
            x = x * (x * (1.5 - x) + 0.5)
            contrast_blend = self.contrast * 2.0
        elif self.formula == 5:
            x = x * x * x * (x * (x * 6.0 - 15.0) + 10.0)

        if self.mode == 0:
            color = luma + chroma
            color = lerp(color, x + chroma, contrast_blend)
        elif self.mode == 1:
            x = x * 2.0 - 1.0
            color = lerp(luma + chroma, luma + x, contrast_blend)
        else:
            color = lerp(img_float, x, contrast_blend)

        return (saturate(color) * 255).astype(np.uint8)


class DPXFilterAccurate(BaseFilter):
    """DPX/Cineon 시네마틱 색상 그레이딩"""

    def __init__(self):
        super().__init__("DPX", "DPX")

        self.rgb_curve = np.array([8.0, 8.0, 8.0], dtype=np.float32)
        self.rgb_c = np.array([0.36, 0.36, 0.34], dtype=np.float32)
        self.contrast = 0.1
        self.saturation = 3.0
        self.colorfulness = 2.5
        self.strength = 0.20

        self.RGB_matrix = np.array(
            [
                [2.6714711726599600, -1.2672360578624100, -0.4109956021722270],
                [-1.0251070293466400, 1.9840911624108900, 0.0439502493584124],
                [0.0610009456429445, -0.2236707508128630, 1.1590210416706100],
            ],
            dtype=np.float32,
        )

        self.XYZ_matrix = np.array(
            [
                [0.5003033835433160, 0.3380975732227390, 0.1645897795458570],
                [0.2579688942747580, 0.6761952591447060, 0.0658358459823868],
                [0.0234517888692628, 0.1126992737203000, 0.8668396731242010],
            ],
            dtype=np.float32,
        )

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        rgb_curve = params.get("RGB_Curve", tuple(self.rgb_curve))
        rgb_c = params.get("RGB_C", tuple(self.rgb_c))
        self.rgb_curve = np.array(rgb_curve, dtype=np.float32)
        self.rgb_c = np.array(rgb_c, dtype=np.float32)
        self.contrast = params.get("Contrast", self.contrast)
        self.saturation = params.get("Saturation", self.saturation)
        self.colorfulness = params.get("Colorfulness", self.colorfulness)
        self.strength = params.get("Strength", self.strength)

        img_float = image.astype(np.float32) / 255.0

        B = img_float * (1.0 - self.contrast) + (0.5 * self.contrast)

        B_temp = 1.0 / (1.0 + np.exp(self.rgb_curve / 2.0))
        B = (
            (1.0 / (1.0 + np.exp(-self.rgb_curve * (B - self.rgb_c))))
            / (-2.0 * B_temp + 1.0)
        ) + (-B_temp / (-2.0 * B_temp + 1.0))

        value = np.maximum(np.maximum(B[:, :, 0:1], B[:, :, 1:2]), B[:, :, 2:3])
        color = B / (value + 1e-8)
        color = pow_safe(color, 1.0 / self.colorfulness)

        c0 = color * value

        c0_transformed = np.dot(c0.reshape(-1, 3), self.XYZ_matrix.T).reshape(c0.shape)

        luma = np.sum(
            c0_transformed * np.array([0.30, 0.59, 0.11]), axis=2, keepdims=True
        )
        c0_transformed = (
            1.0 - self.saturation
        ) * luma + self.saturation * c0_transformed

        result = np.dot(c0_transformed.reshape(-1, 3), self.RGB_matrix.T).reshape(
            c0_transformed.shape
        )

        output = lerp(img_float, result, self.strength)

        return (saturate(output) * 255).astype(np.uint8)


class TonemapFilterAccurate(BaseFilter):
    """Tonemap - 노출, 감마, 채도 조정"""

    def __init__(self):
        super().__init__("Tonemap", "톤맵")

        self.gamma = 1.0
        self.exposure = 0.0
        self.saturation = 0.0
        self.bleach = 0.0
        self.defog = 0.0
        self.fog_color = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        self.gamma = params.get("Gamma", self.gamma)
        self.exposure = params.get("Exposure", self.exposure)
        self.saturation = params.get("Saturation", self.saturation)
        self.bleach = params.get("Bleach", self.bleach)
        self.defog = params.get("Defog", self.defog)
        fog_color = params.get("FogColor", tuple(self.fog_color))
        self.fog_color = np.array(fog_color, dtype=np.float32)

        img_float = image.astype(np.float32) / 255.0

        color = saturate(img_float - self.defog * self.fog_color * 2.55)

        color *= np.power(2.0, self.exposure)

        color = pow_safe(color, self.gamma)

        coef_luma = np.array([0.2126, 0.7152, 0.0722])
        lum = np.sum(coef_luma * color, axis=2, keepdims=True)

        L = saturate(10.0 * (lum - 0.45))
        A2 = self.bleach * color

        result1 = 2.0 * color * lum
        result2 = 1.0 - 2.0 * (1.0 - lum) * (1.0 - color)

        new_color = lerp(result1, result2, L)
        mix_rgb = A2 * new_color
        color += (1.0 - A2) * mix_rgb

        middlegray = np.sum(color, axis=2, keepdims=True) * (1.0 / 3.0)
        diffcolor = color - middlegray
        color = (color + diffcolor * self.saturation) / (
            1 + (diffcolor * self.saturation)
        )

        return (saturate(color) * 255).astype(np.uint8)


class SepiaFilterAccurate(BaseFilter):
    """Sepia - 세피아 톤 효과"""

    def __init__(self):
        super().__init__("Sepia", "세피아")

        self.strength = 0.58
        self.tint = np.array([1.40, 1.10, 0.90], dtype=np.float32)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        self.strength = params.get("Strength", self.strength)
        tint = params.get("Tint", tuple(self.tint))
        self.tint = np.array(tint, dtype=np.float32)

        img_float = image.astype(np.float32) / 255.0

        luma = np.sum(
            img_float * np.array([0.2126, 0.7152, 0.0722]), axis=2, keepdims=True
        )

        sepia = luma * self.tint

        result = lerp(img_float, sepia, self.strength)

        return (saturate(result) * 255).astype(np.uint8)


class VignetteFilterAccurate(BaseFilter):
    """Vignette - 비네팅 효과"""

    def __init__(self):
        super().__init__("Vignette", "비네팅")

        self.vignette_type = 0
        self.ratio = 1.0
        self.radius = 2.0
        self.amount = -1.0
        self.slope = 2
        self.center = np.array([0.5, 0.5], dtype=np.float32)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        self.vignette_type = params.get("Type", self.vignette_type)
        self.ratio = params.get("Ratio", self.ratio)
        self.radius = params.get("Radius", self.radius)
        self.amount = params.get("Amount", self.amount)
        self.slope = params.get("Slope", self.slope)
        center = params.get("Center", tuple(self.center))
        self.center = np.array(center, dtype=np.float32)

        img_float = image.astype(np.float32) / 255.0

        h, w = img_float.shape[:2]

        y_coords = np.linspace(0, 1, h).reshape(h, 1)
        x_coords = np.linspace(0, 1, w).reshape(1, w)

        tc = np.stack(
            [np.repeat(x_coords, h, axis=0), np.repeat(y_coords, w, axis=1)], axis=2
        )

        tc -= self.center
        tc[:, :, 0] *= self.ratio

        v = np.sqrt(np.sum(tc * tc, axis=2, keepdims=True))

        if self.vignette_type == 0:
            v = 1.0 - saturate((v - self.radius) * self.slope)
        else:
            v = saturate(((self.radius - v) * self.slope) + 1.0)

        v = pow_safe(v, abs(self.amount))

        result = img_float * v

        return (saturate(result) * 255).astype(np.uint8)
