"""
LensDistort.fx 구현
렌즈 왜곡 효과 (Barrel/Pincushion)

원본 셰이더: https://github.com/crosire/reshade-shaders
"""

import cv2
import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import saturate


class LensDistortFilter(BaseFilter):
    def __init__(self):
        super().__init__("LensDistort", "렌즈 왜곡")
        self.k1 = 0.0  # -1.0 ~ 1.0 (Barrel < 0, Pincushion > 0)
        self.k2 = 0.0  # -1.0 ~ 1.0
        self.zoom = 1.0  # 0.5 ~ 2.0  # -1.0 ~ 1.0

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        렌즈 왜곡 적용

        Parameters:
        -----------
        distortion : float
            왜곡 강도 (-1.0 ~ 1.0)
            음수: Barrel distortion (배럴형 왜곡)
            양수: Pincushion distortion (핀쿠션형 왜곡)
        cubic_distortion : float
            3차 왜곡 계수 (-1.0 ~ 1.0)
        """
        self.distortion = params.get("distortion", self.distortion)
        self.cubic_distortion = params.get("cubic_distortion", self.cubic_distortion)

        h, w = image.shape[:2]
        img_float = image.astype(np.float32) / 255.0

        # 중심점 설정
        cx, cy = w / 2.0, h / 2.0

        # 정규화된 좌표 생성 (-1 ~ 1)
        x = np.arange(w, dtype=np.float32)
        y = np.arange(h, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)

        # 중심으로부터의 거리 계산 (정규화)
        aspect_ratio = w / h
        norm_x = (xx - cx) / cx
        norm_y = (yy - cy) / cy * aspect_ratio

        # 반경 계산
        r = np.sqrt(norm_x**2 + norm_y**2)

        # 왜곡 계수 적용
        # distortion_factor = 1 + k1*r^2 + k2*r^4
        k1 = self.distortion
        k2 = self.cubic_distortion

        distortion_factor = 1.0 + k1 * r**2 + k2 * r**4

        # 왜곡된 좌표 계산
        distorted_x = norm_x * distortion_factor
        distorted_y = norm_y * distortion_factor / aspect_ratio

        # 다시 픽셀 좌표로 변환
        map_x = (distorted_x * cx + cx).astype(np.float32)
        map_y = (distorted_y * cy + cy).astype(np.float32)

        # Remap
        result = cv2.remap(
            img_float, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )

        result = saturate(result)

        return (result * 255).astype(np.uint8)
