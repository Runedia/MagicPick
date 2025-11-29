"""
ArtisticVignette.fx 구현
예술적 비네팅 효과

원본 셰이더: https://github.com/crosire/reshade-shaders
"""

import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import lerp, pow_safe, saturate


class ArtisticVignetteFilter(BaseFilter):
    def __init__(self):
        super().__init__("ArtisticVignette", "예술적 비네팅 효과")
        self.intensity = 0.5  # 0.0 ~ 1.0
        self.power = 2.0  # 0.5 ~ 8.0
        self.radius = 0.8  # 0.1 ~ 2.0
        self.center_x = 0.5  # 0.0 ~ 1.0
        self.center_y = 0.5  # 0.0 ~ 1.0
        self.color = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # 비네팅 색상  # BGR

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        예술적 비네팅 적용

        Parameters:
        -----------
        intensity : float
            비네팅 강도 (0.0 ~ 1.0)
        radius : float
            비네팅 반경 (0.0 ~ 3.0)
        power : float
            비네팅 곡선 (0.5 ~ 10.0)
        center_x : float
            중심 X 좌표 (0.0 ~ 1.0)
        center_y : float
            중심 Y 좌표 (0.0 ~ 1.0)
        color : np.ndarray
            비네팅 색상 [B, G, R] (0.0 ~ 1.0)
        """
        self.intensity = params.get("intensity", self.intensity)
        self.radius = params.get("radius", self.radius)
        self.power = params.get("power", self.power)
        self.center_x = params.get("center_x", self.center_x)
        self.center_y = params.get("center_y", self.center_y)

        if "color" in params:
            self.color = np.array(params["color"], dtype=np.float32)

        img_float = image.astype(np.float32) / 255.0
        h, w = img_float.shape[:2]

        # 정규화된 좌표 생성 (0 ~ 1)
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x, y)

        # 중심으로부터의 거리 계산
        dx = xx - self.center_x
        dy = yy - self.center_y

        # 종횡비 보정
        aspect_ratio = w / h
        dx = dx * aspect_ratio

        # 거리 계산
        distance = np.sqrt(dx**2 + dy**2)

        # 비네팅 마스크 계산
        # radius로 정규화
        vignette_mask = distance / self.radius

        # power 적용 (곡선)
        vignette_mask = pow_safe(vignette_mask, self.power)

        # 0 ~ 1 범위로 클램핑
        vignette_mask = saturate(vignette_mask)

        # intensity 적용
        vignette_mask = vignette_mask * self.intensity

        # 3채널로 확장
        vignette_mask = np.expand_dims(vignette_mask, axis=2)

        # 비네팅 색상 적용
        result = lerp(img_float, self.color, vignette_mask)

        result = saturate(result)

        return (result * 255).astype(np.uint8)
