"""
MagicBloom 필터

자연스러운 블룸 효과를 시뮬레이션합니다.
여러 단계의 가우시안 블러를 적용하고 이를 원본 이미지와 혼합하여
밝은 영역에서 빛이 번지는 듯한 효과를 생성합니다.
"""

import cv2
import numpy as np

from filters.base_filter import BaseFilter

from .hlsl_helpers import saturate  # blend_screen 함수는 직접 구현


class MagicBloomFilter(BaseFilter):
    """MagicBloom 필터 구현"""

    def __init__(self):
        super().__init__()
        self.fBloom_Intensity = 1.0  # 블룸 강도 (0.0 ~ 10.0)
        self.fBloom_Threshold = (
            2.0  # 블룸 임계값 (1.0 ~ 10.0, 낮을수록 더 많은 픽셀이 블룸에 기여)
        )
        self.iDebug = 0  # 디버그 모드 (0: 일반, 1: 블룸 텍스처만 표시)

    def _blend_screen(self, a, b):
        """HLSL blend_screen 함수: 스크린 블렌딩 모드"""
        # 1.0 - (1.0 - a) * (1.0 - b)
        return 1.0 - (1.0 - a) * (1.0 - b)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        MagicBloom 필터 적용

        Parameters:
            image: 입력 이미지 (H, W, 3)
            fBloom_Intensity: 블룸 강도
            fBloom_Threshold: 블룸 임계값
            iDebug: 디버그 모드
        """
        # 파라미터 업데이트
        self.fBloom_Intensity = float(
            params.get("fBloom_Intensity", self.fBloom_Intensity)
        )
        self.fBloom_Threshold = float(
            params.get("fBloom_Threshold", self.fBloom_Threshold)
        )
        self.iDebug = int(params.get("iDebug", self.iDebug))

        img_float = image.astype(np.float32) / 255.0
        original_color = img_float.copy()

        # 블룸 소스 추출: 임계값 적용 및 강도 조절
        # col = pow(abs(col), fBloom_Threshold);
        # col *= fBloom_Intensity;
        thresholded_bloom_source = np.power(
            np.abs(original_color), self.fBloom_Threshold
        )
        thresholded_bloom_source *= self.fBloom_Intensity

        # 여러 단계의 블러 적용 (8개의 블러 패스)
        blurred_blooms = []
        iBlurSamples = 4  # shader static const
        base_sigma = float(iBlurSamples) / 2.0  # 2.0

        # MagicBloom.fx의 Blur Pass에서 사용되는 scale 값들 (대략적인 sigma 추정)
        # PS_Blur1: scale = 2.0
        # PS_Blur2: scale = 4.0
        # PS_Blur3: scale = 8.0
        # PS_Blur4: scale = 8.0 (재사용?)
        # PS_Blur5: scale = 16.0
        # PS_Blur6: scale = 32.0
        # PS_Blur7: scale = 64.0
        # PS_Blur8: scale = 128.0
        scales = [2.0, 4.0, 8.0, 8.0, 16.0, 32.0, 64.0, 128.0]

        for scale in scales:
            # cv2.GaussianBlur는 ksize가 (0,0)일 때 sigmaX, sigmaY를 기반으로 ksize를 계산
            # sigma는 blur의 "반경"과 유사. scale 값이 커질수록 더 많이 블러링됨
            current_sigma = base_sigma * scale
            # ksize를 sigma 값에 따라 자동 설정하기 위해 (0,0) 사용
            blurred = cv2.GaussianBlur(thresholded_bloom_source, (0, 0), current_sigma)
            blurred_blooms.append(blurred)

        # 모든 블러된 블룸을 합산 (shader의 PS_Blend 부분)
        # float3 bloom = tex2D(sMagicBloom_1, uv).rgb + ... + tex2D(sMagicBloom_8, uv).rgb;
        # static const float bloom_accum = 1.0 / 8.0; bloom *= bloom_accum;
        bloom_sum = np.zeros_like(img_float)
        for b in blurred_blooms:
            bloom_sum += b
        bloom = bloom_sum * (1.0 / 8.0)  # 평균 내기

        # 최종 합성 (원본 이미지 + 블룸)
        # col = blend_screen(col, bloom);
        combined_color = self._blend_screen(original_color, bloom)

        # 디버그 모드 처리
        # col = iDebug == 1 ? bloom : col;
        result = bloom if self.iDebug == 1 else combined_color

        # 결과 클리핑
        result = saturate(result)

        return (result * 255).astype(np.uint8)
