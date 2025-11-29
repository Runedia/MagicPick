"""
ZigZag.fx 구현
지그재그 왜곡 효과

원본 셰이더: https://github.com/crosire/reshade-shaders
"""

import cv2
import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import saturate


class ZigZagFilter(BaseFilter):
    def __init__(self):
        super().__init__("ZigZag", "지그재그 왜곡")
        self.frequency = 10.0  # 1.0 ~ 100.0
        self.amplitude = 5.0  # 0.0 ~ 50.0
        self.center_x = 0.5  # 0.0 ~ 1.0
        self.center_y = 0.5  # 0.0 ~ 1.0  # 0.0 ~ 1.0

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        지그재그 왜곡 적용

        중심으로부터의 거리에 따라 사인파 왜곡 적용

        Parameters:
        -----------
        amplitude : float
            왜곡 진폭 (0.0 ~ 50.0)
        frequency : float
            왜곡 주파수 (0.0 ~ 100.0)
        center_x : float
            중심 X 좌표 (0.0 ~ 1.0)
        center_y : float
            중심 Y 좌표 (0.0 ~ 1.0)
        """
        self.amplitude = params.get("amplitude", self.amplitude)
        self.frequency = params.get("frequency", self.frequency)
        self.center_x = params.get("center_x", self.center_x)
        self.center_y = params.get("center_y", self.center_y)

        h, w = image.shape[:2]
        img_float = image.astype(np.float32) / 255.0

        # 중심점 계산
        cx = self.center_x * w
        cy = self.center_y * h

        # 좌표 그리드 생성
        x = np.arange(w, dtype=np.float32)
        y = np.arange(h, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)

        # 중심으로부터의 상대 좌표
        dx = xx - cx
        dy = yy - cy

        # 극좌표 변환
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)

        # 지그재그 왜곡 적용
        # 반경에 따라 사인파로 각도 조정
        zigzag_angle = np.sin(r * self.frequency * 0.01) * self.amplitude * 0.01

        theta_new = theta + zigzag_angle

        # 극좌표 -> 직교좌표
        new_x = cx + r * np.cos(theta_new)
        new_y = cy + r * np.sin(theta_new)

        # Remap
        result = cv2.remap(
            img_float,
            new_x.astype(np.float32),
            new_y.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        result = saturate(result)

        return (result * 255).astype(np.uint8)
