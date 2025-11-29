"""
ReShade LevelsPlus 필터
"""

import numpy as np

from ..base_filter import BaseFilter
from .hlsl_helpers import saturate


class LevelsPlusFilter(BaseFilter):
    def __init__(self):
        super().__init__("LevelsPlus", "ReShade의 LevelsPlus 효과를 적용합니다.")
        self.set_default_params(
            {
                "enable_levels": True,
                "input_black_point": [16.0 / 255.0, 18.0 / 255.0, 20.0 / 255.0],
                "input_white_point": [233.0 / 255.0, 222.0 / 255.0, 211.0 / 255.0],
                "input_gamma": [1.0, 1.0, 1.0],
                "output_black_point": [0.0, 0.0, 0.0],
                "output_white_point": [1.0, 1.0, 1.0],
                "color_range_shift": [0.0, 0.0, 0.0],
                "color_range_shift_switch": 0,
                "enable_aces_old": False,
                "enable_aces_new": False,
                "enable_aces_fitted": False,
                "aces_luminance_percentage": [100, 100, 100],
                "highlight_clipping": False,
            }
        )
        self.aces_input_mat = np.array(
            [
                [0.59719, 0.35458, 0.04823],
                [0.07600, 0.90834, 0.01566],
                [0.02840, 0.13383, 0.83777],
            ]
        )
        self.aces_output_mat = np.array(
            [
                [1.60475, -0.53108, -0.07367],
                [-0.10208, 1.10813, -0.00605],
                [-0.00327, -0.07276, 1.07602],
            ]
        )

    def _aces_film_rec2020_old(self, color, lum_percent):
        slope = 15.8
        toe = 2.12
        shoulder = 1.2
        black_clip = 5.92
        white_clip = 1.9
        color = color * np.array(lum_percent) * 0.005
        numerator = color * (slope * color + toe)
        denominator = color * (shoulder * color + black_clip) + white_clip
        return numerator / denominator

    def _aces_film_rec2020(self, color, lum_percent):
        slope = 0.98
        toe = 0.3
        shoulder = 0.22
        black_clip = 0.0
        white_clip = 0.025
        color = color * np.array(lum_percent) * 0.005
        numerator = color * (slope * color + toe)
        denominator = color * (shoulder * color + black_clip) + white_clip
        return numerator / denominator

    def _rrt_and_odt_fit(self, v):
        a = v * (v + 0.0245786) - 0.000090537
        b = v * (0.983729 * v + 0.4329510) + 0.238081
        return a / b

    def _aces_fitted(self, color):
        color = np.einsum("ij,...j->...i", self.aces_input_mat, color)
        color = self._rrt_and_odt_fit(color)
        color = np.einsum("ij,...j->...i", self.aces_output_mat, color)
        return saturate(color)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        params = self.validate_params(params)
        img_float = image.astype(np.float32) / 255.0

        output_color = img_float.copy()

        if params["enable_levels"]:
            shift = (
                np.array(params["color_range_shift"])
                * params["color_range_shift_switch"]
            )
            in_black = np.array(params["input_black_point"])
            in_white = np.array(params["input_white_point"])
            gamma = np.array(params["input_gamma"])
            out_black = np.array(params["output_black_point"])
            out_white = np.array(params["output_white_point"])

            numerator = (output_color + shift) - in_black
            denominator = in_white - in_black

            # 0으로 나누기 방지
            denominator[denominator == 0] = 1.0

            levels_val = numerator / denominator
            gamma_corrected = np.power(saturate(levels_val), gamma)
            output_color = gamma_corrected * (out_white - out_black) + out_black

        if params["enable_aces_old"]:
            output_color = self._aces_film_rec2020_old(
                output_color, params["aces_luminance_percentage"]
            )

        if params["enable_aces_new"]:
            output_color = self._aces_film_rec2020(
                output_color, params["aces_luminance_percentage"]
            )

        if params["enable_aces_fitted"]:
            output_color = self._aces_fitted(output_color)

        if params["highlight_clipping"]:
            clipped_color = output_color.copy()

            is_whiter = np.any(output_color > 1.0, axis=-1, keepdims=True)
            is_all_whiter = np.all(output_color > 1.0, axis=-1, keepdims=True)
            is_blacker = np.any(output_color < 0.0, axis=-1, keepdims=True)
            is_all_blacker = np.all(output_color < 0.0, axis=-1, keepdims=True)

            clipped_color = np.where(is_whiter, [1.0, 1.0, 0.0], clipped_color)
            clipped_color = np.where(is_all_whiter, [1.0, 0.0, 0.0], clipped_color)
            clipped_color = np.where(is_blacker, [0.0, 1.0, 1.0], clipped_color)
            clipped_color = np.where(is_all_blacker, [0.0, 0.0, 1.0], clipped_color)

            output_color = clipped_color

        return np.clip(output_color * 255.0, 0, 255).astype(np.uint8)
