"""
FakeHDR 필터

HDR(High Dynamic Range) 룩을 모방하는 필터입니다.
실제 HDR은 아니지만, 두 개의 다른 블러 반경을 사용하여 대비를 높이고 세부 사항을 강조합니다.
"""

import numpy as np

from filters.base_filter import BaseFilter

from .hlsl_helpers import saturate, shift_image_approx


class FakeHDRFilter(BaseFilter):
    """FakeHDR 필터 구현"""

    def __init__(self):
        super().__init__()
        self.hdr_power = 1.30  # 0.0 ~ 8.0
        self.radius1 = 0.793  # 0.0 ~ 8.0
        self.radius2 = 0.87  # 0.0 ~ 8.0

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        FakeHDR 필터 적용

        Parameters:
            image: 입력 이미지 (H, W, 3)
            HDRPower: 강도 (0.0 ~ 8.0, 기본값 1.30)
            radius1: 첫 번째 샘플링 반경 (0.0 ~ 8.0, 기본값 0.793)
            radius2: 두 번째 샘플링 반경 (0.0 ~ 8.0, 기본값 0.87)
        """
        # 파라미터 업데이트
        self.hdr_power = float(params.get("HDRPower", self.hdr_power))
        self.radius1 = float(params.get("radius1", self.radius1))
        self.radius2 = float(params.get("radius2", self.radius2))

        img_float = image.astype(np.float32) / 255.0

        # 샘플링 오프셋 (Shader 코드 참조)
        # float2(1.5, -1.5), float2(-1.5, -1.5), float2(1.5, 1.5), float2(-1.5, 1.5)
        # float2(0.0, -2.5), float2(0.0, 2.5), float2(-2.5, 0.0), float2(2.5, 0.0)
        offsets_base = [
            (1.5, -1.5),
            (-1.5, -1.5),
            (1.5, 1.5),
            (-1.5, 1.5),
            (0.0, -2.5),
            (0.0, 2.5),
            (-2.5, 0.0),
            (2.5, 0.0),
        ]

        # --- Pass 1: Bloom Sum 1 (Radius 1) ---
        bloom_sum1 = np.zeros_like(img_float)
        for ox, oy in offsets_base:
            # 셰이더: texcoord + float2(...) * radius1 * PixelSize
            # 픽셀 단위 오프셋 = base_offset * radius1
            dx = ox * self.radius1
            dy = oy * self.radius1

            shifted = shift_image_approx(img_float, dx, dy)
            bloom_sum1 += shifted

        bloom_sum1 *= 0.005

        # --- Pass 2: Bloom Sum 2 (Radius 2) ---
        bloom_sum2 = np.zeros_like(img_float)
        for ox, oy in offsets_base:
            dx = ox * self.radius2
            dy = oy * self.radius2

            shifted = shift_image_approx(img_float, dx, dy)
            bloom_sum2 += shifted

        bloom_sum2 *= 0.010

        # --- 합성 ---
        dist = self.radius2 - self.radius1

        # float3 HDR = (color + (bloom_sum2 - bloom_sum1)) * dist;
        hdr = (img_float + (bloom_sum2 - bloom_sum1)) * dist

        # float3 blend = HDR + color;
        blend = hdr + img_float

        # color = pow(abs(blend), abs(HDRPower)) + HDR;
        # 주의: blend가 음수일 경우 abs 처리
        result = np.power(np.abs(blend), np.abs(self.hdr_power)) + hdr

        result = saturate(result)

        return (result * 255).astype(np.uint8)
