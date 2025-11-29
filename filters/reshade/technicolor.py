"""
Technicolor.fx 구현
테크니컬러 영화 효과

원본 셰이더: https://github.com/crosire/reshade-shaders
"""

import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import dot3, lerp, saturate


class TechnicolorFilter(BaseFilter):
    def __init__(self):
        super().__init__("Technicolor", "테크니컬러 2-strip")
        self.strength = 0.4  # 0.0 ~ 1.0
        self.red_negative = (-0.87, -0.88, 0.12)
        self.green_negative = (0.0, 1.24, -0.12)
        self.blue_negative = (-0.03, -0.36, 1.38)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        테크니컬러 효과 적용

        Technicolor 2-strip 프로세스 시뮬레이션:
        - Red 필름과 Cyan 필름을 사용하여 색상 재현
        - 특유의 따뜻하고 강렬한 색감

        Parameters:
        -----------
        strength : float
            효과 강도 (0.0 ~ 1.0)
        """
        self.strength = params.get("strength", self.strength)

        img_float = image.astype(np.float32) / 255.0

        # BGR → RGB 변환
        rgb = img_float[:, :, ::-1].copy()

        # Technicolor 2-strip 네거티브 매트릭스
        # Red 채널
        cyanfilm = saturate(
            dot3(
                rgb,
                np.array(self.red_negative, dtype=np.float32),
            )
        )

        # Green 채널
        redfilm = saturate(
            dot3(
                rgb,
                np.array(self.green_negative, dtype=np.float32),
            )
        )

        # Blue 채널
        bluefilm = saturate(
            dot3(
                rgb,
                np.array(self.blue_negative, dtype=np.float32),
            )
        )

        # 네거티브 필름 결합
        redoutput = np.stack([redfilm, cyanfilm, cyanfilm], axis=2)
        greenoutput = np.stack([redfilm, cyanfilm, redfilm], axis=2)
        blueoutput = np.stack([bluefilm, bluefilm, cyanfilm], axis=2)

        # 최종 Technicolor 색상
        result = redoutput * greenoutput * blueoutput

        # RGB → BGR 변환
        result = result[:, :, ::-1]

        # 원본과 블렌딩
        result = lerp(img_float, result, self.strength)

        result = saturate(result)

        return (result * 255).astype(np.uint8)
