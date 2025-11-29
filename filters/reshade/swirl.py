"""
Swirl.fx 구현
스월 효과 (소용돌이 왜곡)

원본 셰이더: https://github.com/crosire/reshade-shaders
"""

import cv2
import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import saturate


class SwirlFilter(BaseFilter):
    def __init__(self):
        super().__init__("Swirl", "스월 효과")
        self.angle = 0.5  # -2.0 ~ 2.0 (radians)
        self.radius = 0.5  # 0.0 ~ 1.0
        self.center_x = 0.5  # 0.0 ~ 1.0
        self.center_y = 0.5  # 0.0 ~ 1.0  # 0.0 ~ 1.0

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        스월 효과 적용

        Parameters:
        -----------
        swirl_angle : float
            스월 각도 (-720.0 ~ 720.0 degrees)
        swirl_radius : float
            스월 반경 (0.0 ~ 1.0)
        center_x : float
            중심 X 좌표 (0.0 ~ 1.0)
        center_y : float
            중심 Y 좌표 (0.0 ~ 1.0)
        """
        self.swirl_angle = params.get("swirl_angle", self.swirl_angle)
        self.swirl_radius = params.get("swirl_radius", self.swirl_radius)
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

        # 스월 반경 계산
        max_radius = self.swirl_radius * min(w, h) / 2.0

        # 스월 강도 계산 (중심에서 멀어질수록 약해짐)
        swirl_strength = np.maximum(1.0 - r / max_radius, 0.0)

        # 각도 변환 (degrees to radians)
        angle_rad = np.deg2rad(self.swirl_angle)

        # 스월 적용
        theta_new = theta + angle_rad * swirl_strength

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
