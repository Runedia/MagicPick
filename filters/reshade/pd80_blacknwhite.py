import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import hsl_to_rgb, lerp, rgb_to_hsl, saturate


# Helper function from HLSL
def _curve(x, k):
    s = np.sign(x - 0.5)
    o = (1.0 + s) / 2.0
    return o - 0.5 * s * np.power(np.maximum(2.0 * (o - s * x), 0.0), k)


def _process_bw(col, r, y, g, c, b, m, curve_str):
    # col is RGB (float32, 0-1)
    # r, y, g, c, b, m are weights

    hsl = rgb_to_hsl(col)

    # Inverse of luma channel to not apply boosts to intensity on already intense brightness (and blow out easily)
    lum = 1.0 - hsl[:, :, 2]  # hsl.z is lightness

    # Calculate the individual weights per color component in RGB and CMY
    # Sum of all the weights for a given hue is 1.0
    # hsl.x is hue (0-1)
    hue_6 = hsl[:, :, 0] * 6.0

    weight_r = _curve(np.maximum(1.0 - np.abs(hue_6), 0.0), curve_str) + _curve(
        np.maximum(1.0 - np.abs(hue_6 - 6.0), 0.0), curve_str
    )
    weight_y = _curve(np.maximum(1.0 - np.abs(hue_6 - 1.0), 0.0), curve_str)
    weight_g = _curve(np.maximum(1.0 - np.abs(hue_6 - 2.0), 0.0), curve_str)
    weight_c = _curve(np.maximum(1.0 - np.abs(hue_6 - 3.0), 0.0), curve_str)
    weight_b = _curve(np.maximum(1.0 - np.abs(hue_6 - 4.0), 0.0), curve_str)
    weight_m = _curve(np.maximum(1.0 - np.abs(hue_6 - 5.0), 0.0), curve_str)

    # No saturation (greyscale) should not influence B&W image
    sat = hsl[:, :, 1]  # hsl.y is saturation
    ret = hsl[:, :, 2]  # hsl.z is lightness

    # Apply weights
    ret += ret * (weight_r * r) * sat * lum
    ret += ret * (weight_y * y) * sat * lum
    ret += ret * (weight_g * g) * sat * lum
    ret += ret * (weight_c * c) * sat * lum
    ret += ret * (weight_b * b) * sat * lum
    ret += ret * (weight_m * m) * sat * lum

    return saturate(ret)


class PD80_BlacknWhiteFilter(BaseFilter):
    """
    PD80 Black & White 필터

    ReShade PD80_04_BlacknWhite.fx를 기반으로 한 흑백 변환 필터.
    다양한 흑백 변환 모드와 사용자 정의 채널 가중치를 지원합니다.
    """

    def __init__(self):
        super().__init__("PD80_BlacknWhite", "PD80 흑백")
        self.enable_dither = True
        self.dither_strength = 1.0
        self.curve_str = 1.5
        self.show_clip = False  # Not implemented for now

        self.bw_mode = 13  # Custom
        self.redchannel = 0.2
        self.yellowchannel = 0.4
        self.greenchannel = 0.6
        self.cyanchannel = 0.0
        self.bluechannel = -0.6
        self.magentachannel = -0.2

        self.use_tint = False
        self.tinthue = 0.083
        self.tintsat = 0.12

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        self.enable_dither = params.get("enable_dither", self.enable_dither)
        self.dither_strength = params.get("dither_strength", self.dither_strength)
        self.curve_str = params.get("curve_str", self.curve_str)
        self.bw_mode = params.get("bw_mode", self.bw_mode)
        self.redchannel = params.get("redchannel", self.redchannel)
        self.yellowchannel = params.get("yellowchannel", self.yellowchannel)
        self.greenchannel = params.get("greenchannel", self.greenchannel)
        self.cyanchannel = params.get("cyanchannel", self.cyanchannel)
        self.bluechannel = params.get("bluechannel", self.bluechannel)
        self.magentachannel = params.get("magentachannel", self.magentachannel)
        self.use_tint = params.get("use_tint", self.use_tint)
        self.tinthue = params.get("tinthue", self.tinthue)
        self.tintsat = params.get("tintsat", self.tintsat)

        img_float = image.astype(np.float32) / 255.0

        red, yellow, green = 0.0, 0.0, 0.0
        cyan, blue, magenta = 0.0, 0.0, 0.0

        # Based on bw_mode, set weights
        # Only implementing a few for now, custom is default.
        if self.bw_mode == 0:  # Red Filter
            red, yellow, green, cyan, blue, magenta = 0.2, 0.5, -0.2, -0.6, -1.0, -0.2
        elif self.bw_mode == 1:  # Green Filter
            red, yellow, green, cyan, blue, magenta = -0.5, 0.5, 1.2, -0.2, -1.0, -0.5
        elif self.bw_mode == 2:  # Blue Filter
            red, yellow, green, cyan, blue, magenta = -0.2, 0.4, -0.6, 0.5, 1.0, -0.2
        elif self.bw_mode == 6:  # Infrared
            red, yellow, green, cyan, blue, magenta = (
                -1.35,
                2.35,
                1.35,
                -1.35,
                -1.6,
                -1.07,
            )
        elif self.bw_mode == 7:  # Maximum Black
            red, yellow, green, cyan, blue, magenta = -1.0, -1.0, -1.0, -1.0, -1.0, -1.0
        elif self.bw_mode == 8:  # Maximum White
            red, yellow, green, cyan, blue, magenta = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        elif self.bw_mode == 10:  # Neutral Green Filter
            red, yellow, green, cyan, blue, magenta = 0.2, 0.4, 0.6, 0.0, -0.6, -0.2
        elif self.bw_mode == 11:  # Maintain Contrasts
            red, yellow, green, cyan, blue, magenta = -0.3, 1.0, -0.3, -0.6, -1.0, -0.6
        elif self.bw_mode == 12:  # High Contrast
            red, yellow, green, cyan, blue, magenta = -0.3, 2.6, -0.3, -1.2, -0.6, -0.4
        else:  # Custom or Default
            red, yellow, green = self.redchannel, self.yellowchannel, self.greenchannel
            cyan, blue, magenta = (
                self.cyanchannel,
                self.bluechannel,
                self.magentachannel,
            )

        # Process Black & White
        bw_luma = _process_bw(
            img_float, red, yellow, green, cyan, blue, magenta, self.curve_str
        )
        bw_color = np.stack([bw_luma, bw_luma, bw_luma], axis=2)

        # Tinting
        if self.use_tint:
            # The tint in HLSL is lerp(color.xyz, HSLToRGB(float3(tinthue, tintsat, color.x)), use_tint)
            # color.x here is the luma value of the B&W image (all channels are same, so .x is fine)
            tint_hsl = np.stack(
                [
                    np.full_like(bw_luma, self.tinthue),
                    np.full_like(bw_luma, self.tintsat),
                    bw_luma,  # lightness from the B&W image
                ],
                axis=2,
            )
            tint_rgb = hsl_to_rgb(tint_hsl)
            bw_color = lerp(
                bw_color, tint_rgb, self.use_tint
            )  # use_tint (bool) directly as t in lerp

        # Dithering (simple noise for now)
        if self.enable_dither:
            dither_noise = (np.random.rand(*img_float.shape) - 0.5) * (
                self.dither_strength / 255.0
            )
            bw_color = saturate(bw_color + dither_noise)

        # Show clipping (skipped for now as it's a debug feature)

        return (saturate(bw_color) * 255).astype(np.uint8)
