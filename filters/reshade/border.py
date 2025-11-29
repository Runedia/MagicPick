"""
Border 필터

이미지 주위에 테두리를 추가합니다.
픽셀 단위 크기 또는 종횡비 기반 크기 설정 가능
"""

import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import saturate


class BorderFilter(BaseFilter):
    """테두리 필터"""

    def __init__(self):
        super().__init__("Border", "테두리 효과")
        self.border_width = [0.0, 0.0]  # (X, Y) 픽셀 크기
        self.border_ratio = 2.35  # 화면 비율 (종횡비)
        self.border_color = [0.0, 0.0, 0.0]  # RGB 색상

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """Border 필터 적용"""
        # 파라미터 업데이트
        if "border_width" in params:
            self.border_width = params["border_width"]
        self.border_ratio = params.get("border_ratio", self.border_ratio)
        if "border_color" in params:
            self.border_color = params["border_color"]

        img_float = image.astype(np.float32) / 255.0
        h, w = img_float.shape[:2]

        # 화면 비율 계산
        aspect_ratio = w / h

        # 테두리 크기 계산
        border_width_x, border_width_y = self.border_width

        # border_width가 설정되지 않은 경우 (0, 0) 또는 특수 조건
        # HLSL: if (border_width.x == -border_width.y) 조건 근사
        use_ratio = (border_width_x == 0.0 and border_width_y == 0.0) or (
            abs(border_width_x + border_width_y) < 1e-6
        )

        if use_ratio:
            # 종횡비 기반 테두리 크기 계산
            if aspect_ratio < self.border_ratio:
                # 세로 테두리 추가
                border_width_x = 0.0
                border_width_y = (h - (w / self.border_ratio)) * 0.5
            else:
                # 가로 테두리 추가
                border_width_x = (w - (h * self.border_ratio)) * 0.5
                border_width_y = 0.0

        # 픽셀 단위 테두리를 정규화된 좌표로 변환
        border_x = border_width_x / w
        border_y = border_width_y / h

        # 텍스처 좌표 생성 (0~1)
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        texcoord_x = x_coords / w
        texcoord_y = y_coords / h

        # 테두리 내부 마스크 계산
        # within_border = saturate((-texcoord * texcoord + texcoord) - (-border * border + border))
        # = saturate(texcoord - texcoord^2 - border + border^2)

        within_x = saturate(
            (-texcoord_x * texcoord_x + texcoord_x) - (-border_x * border_x + border_x)
        )
        within_y = saturate(
            (-texcoord_y * texcoord_y + texcoord_y) - (-border_y * border_y + border_y)
        )

        # all(within_border): 둘 다 양수여야 내부
        within_border = (within_x > 0) & (within_y > 0)

        # 결과 이미지 생성
        result = img_float.copy()

        # BGR 순서로 테두리 색상 적용
        border_color_bgr = np.array(
            [self.border_color[2], self.border_color[1], self.border_color[0]],
            dtype=np.float32,
        )

        # 테두리 영역에 색상 적용
        result[~within_border] = border_color_bgr

        return (result * 255).astype(np.uint8)
