"""
Chromatic Aberration 필터

색상 성분을 이동시켜 이미지를 왜곡합니다.
저렴한 렌즈나 센서에서 나타나는 색수차 효과를 재현합니다.
"""

import cv2
import numpy as np

from filters.base_filter import BaseFilter

from .hlsl_helpers import lerp


class ChromaticAberrationFilter(BaseFilter):
    """색수차 효과 필터"""

    def __init__(self):
        super().__init__("ChromaticAberration", "렌즈 색수차 효과")
        self.shift = [2.5, -0.5]  # (X, Y) 픽셀 이동 거리 (-10 ~ 10)
        self.strength = 0.5  # 효과 강도 (0.0 ~ 1.0)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """Chromatic Aberration 필터 적용"""
        # 파라미터 업데이트
        if "shift" in params:
            self.shift = params["shift"]
        self.strength = params.get("strength", self.strength)

        img_float = image.astype(np.float32) / 255.0
        h, w = img_float.shape[:2]

        # 원본 색상
        color_input = img_float.copy()

        # Shift 값 (픽셀 단위)
        shift_x = self.shift[0]
        shift_y = self.shift[1]

        # 색상 성분별로 이동
        # Red 채널: +Shift 방향으로 이동
        # Green 채널: 원본 유지
        # Blue 채널: -Shift 방향으로 이동

        # 이동 행렬 생성
        # OpenCV warpAffine 사용 (sub-pixel 정밀도 지원)
        def shift_channel(channel, dx, dy):
            """채널을 (dx, dy) 픽셀만큼 이동"""
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(
                channel,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            return shifted

        # BGR 순서 (OpenCV)
        b_channel = img_float[:, :, 0]
        g_channel = img_float[:, :, 1]
        r_channel = img_float[:, :, 2]

        # Red: +Shift
        r_shifted = shift_channel(r_channel, shift_x, shift_y)

        # Green: 원본 유지
        g_shifted = g_channel

        # Blue: -Shift
        b_shifted = shift_channel(b_channel, -shift_x, -shift_y)

        # 결과 합성
        color = np.stack([b_shifted, g_shifted, r_shifted], axis=2)

        # 강도 조정: 원본과 색수차 효과 사이 보간
        result = lerp(color_input, color, self.strength)

        return (result * 255).astype(np.uint8)
