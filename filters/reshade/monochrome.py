"""
Monochrome 필터

색상을 제거하여 흑백 이미지를 만듭니다.
B/W 카메라 필름을 기반으로 한 다양한 프리셋을 제공합니다.
"""

import numpy as np

from filters.base_filter import BaseFilter

from .hlsl_helpers import lerp, saturate


class MonochromeFilter(BaseFilter):
    """흑백 변환 필터 (다양한 필름 프리셋 지원)"""

    # 프리셋 계수 배열 (RGB 가중치)
    PRESETS = {
        0: [0.21, 0.72, 0.07],  # Custom (사용자 지정값 사용)
        1: [0.21, 0.72, 0.07],  # sRGB monitor
        2: [0.3333333, 0.3333334, 0.3333333],  # Equal weight
        3: [0.18, 0.41, 0.41],  # Agfa 200X
        4: [0.25, 0.39, 0.36],  # Agfapan 25
        5: [0.21, 0.40, 0.39],  # Agfapan 100
        6: [0.20, 0.41, 0.39],  # Agfapan 400
        7: [0.21, 0.42, 0.37],  # Ilford Delta 100
        8: [0.22, 0.42, 0.36],  # Ilford Delta 400
        9: [0.31, 0.36, 0.33],  # Ilford Delta 400 Pro & 3200
        10: [0.28, 0.41, 0.31],  # Ilford FP4
        11: [0.23, 0.37, 0.40],  # Ilford HP5
        12: [0.33, 0.36, 0.31],  # Ilford Pan F
        13: [0.36, 0.31, 0.33],  # Ilford SFX
        14: [0.21, 0.42, 0.37],  # Ilford XP2 Super
        15: [0.24, 0.37, 0.39],  # Kodak Tmax 100
        16: [0.27, 0.36, 0.37],  # Kodak Tmax 400
        17: [0.25, 0.35, 0.40],  # Kodak Tri-X
    }

    PRESET_NAMES = [
        "Custom",
        "Monitor or modern TV",
        "Equal weight",
        "Agfa 200X",
        "Agfapan 25",
        "Agfapan 100",
        "Agfapan 400",
        "Ilford Delta 100",
        "Ilford Delta 400",
        "Ilford Delta 400 Pro & 3200",
        "Ilford FP4",
        "Ilford HP5",
        "Ilford Pan F",
        "Ilford SFX",
        "Ilford XP2 Super",
        "Kodak Tmax 100",
        "Kodak Tmax 400",
        "Kodak Tri-X",
    ]

    def __init__(self):
        super().__init__("Monochrome", "흑백 변환")
        self.preset = 0  # 0 = Custom, 1 = sRGB monitor, etc.
        self.conversion_values = [0.21, 0.72, 0.07]  # Custom RGB coefficients
        self.saturation = 0.0  # 0.0 = full monochrome, 1.0 = full color

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """Monochrome 필터 적용"""
        # 파라미터 업데이트
        self.preset = params.get("preset", self.preset)
        self.saturation = params.get("saturation", self.saturation)

        if "conversion_values" in params:
            self.conversion_values = params["conversion_values"]

        img_float = image.astype(np.float32) / 255.0

        # 프리셋에 따라 계수 선택
        if self.preset == 0:
            # Custom: 사용자 지정 conversion_values 사용
            coefficients = np.array(self.conversion_values, dtype=np.float32)
        else:
            coefficients = np.array(self.PRESETS[self.preset], dtype=np.float32)

        # 흑백 변환: dot product (R*coef[0] + G*coef[1] + B*coef[2])
        grey = np.dot(img_float, coefficients)

        # 3채널로 확장
        grey_rgb = np.stack([grey, grey, grey], axis=2)

        # 채도 조정: 흑백과 원본 색상 사이 보간
        result = lerp(grey_rgb, img_float, self.saturation)

        result = saturate(result)

        return (result * 255).astype(np.uint8)
