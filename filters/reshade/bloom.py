"""
Bloom 필터

밝은 영역에 글로우 효과를 추가합니다.
기본 블룸 기능만 구현 (렌즈 플레어, 갓레이 등 제외)
"""

import cv2
import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import lerp, saturate


class BloomFilter(BaseFilter):
    """블룸 글로우 필터"""

    # 블렌드 모드
    BLEND_LINEAR_ADD = 0
    BLEND_SCREEN_ADD = 1
    BLEND_SCREEN_LIGHTEN_OPACITY = 2
    BLEND_LIGHTEN = 3

    def __init__(self):
        super().__init__("Bloom", "기본 블룸 효과")
        self.mixmode = 2  # 0~3
        self.threshold = 0.8  # 0.1 ~ 1.0
        self.amount = 0.8  # 0.0 ~ 20.0
        self.saturation = 0.8  # 0.0 ~ 2.0
        self.tint = [0.7, 0.8, 1.0]  # RGB 틴트

    def _extract_bright_areas(self, img_float):
        """임계값보다 밝은 영역 추출"""
        # 각 픽셀의 최대 채널 값
        brightness = np.max(img_float, axis=2)

        # 임계값 적용
        mask = brightness > self.threshold

        bright = img_float.copy()
        bright[~mask] = 0.0

        return bright

    def _apply_bloom_blur(self, bright):
        """블룸 블러 적용 (다중 스케일 가우시안)"""
        # 여러 크기의 블러를 합성하여 부드러운 글로우 생성
        blur1 = cv2.GaussianBlur(bright, (0, 0), sigmaX=5)
        blur2 = cv2.GaussianBlur(bright, (0, 0), sigmaX=15)
        blur3 = cv2.GaussianBlur(bright, (0, 0), sigmaX=25)

        # 합성
        bloom = (blur1 + blur2 + blur3) / 3.0

        return bloom

    def _adjust_bloom_saturation(self, bloom):
        """블룸 채도 조정"""
        # Luma 계산
        luma = np.dot(bloom, [0.299, 0.587, 0.114])
        luma = luma[:, :, np.newaxis]

        # 채도 조정
        bloom = lerp(luma, bloom, self.saturation)

        return bloom

    def _apply_tint(self, bloom):
        """블룸 틴트 적용"""
        tint = np.array(
            [self.tint[2], self.tint[1], self.tint[0]], dtype=np.float32
        )  # BGR
        bloom = bloom * tint
        return bloom

    def _blend(self, original, bloom):
        """블렌드 모드 적용"""
        bloom = bloom * self.amount

        if self.mixmode == self.BLEND_LINEAR_ADD:
            # Linear add
            result = original + bloom
        elif self.mixmode == self.BLEND_SCREEN_ADD:
            # Screen add: 1 - (1-A)*(1-B)
            result = 1.0 - (1.0 - original) * (1.0 - bloom)
        elif self.mixmode == self.BLEND_SCREEN_LIGHTEN_OPACITY:
            # Screen/Lighten/Opacity 혼합
            screen = 1.0 - (1.0 - original) * (1.0 - bloom)
            lighten = np.maximum(original, bloom)
            result = lerp(screen, lighten, 0.5)
        else:  # BLEND_LIGHTEN
            # Lighten: max(A, B)
            result = np.maximum(original, bloom)

        return saturate(result)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """Bloom 필터 적용"""
        # 파라미터 업데이트
        self.mixmode = params.get("mixmode", self.mixmode)
        self.threshold = params.get("threshold", self.threshold)
        self.amount = params.get("amount", self.amount)
        self.saturation = params.get("saturation", self.saturation)

        if "tint" in params:
            self.tint = params["tint"]

        img_float = image.astype(np.float32) / 255.0

        # 1. 밝은 영역 추출
        bright = self._extract_bright_areas(img_float)

        # 2. 블러 적용
        bloom = self._apply_bloom_blur(bright)

        # 3. 채도 조정
        bloom = self._adjust_bloom_saturation(bloom)

        # 4. 틴트 적용
        bloom = self._apply_tint(bloom)

        # 5. 원본과 블렌드
        result = self._blend(img_float, bloom)

        return (result * 255).astype(np.uint8)
