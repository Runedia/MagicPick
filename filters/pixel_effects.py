"""
픽셀 기반 효과 필터 모듈

모자이크, 블러, 샤프닝, 엠보싱 등 픽셀 레벨 처리 효과를 제공합니다.
"""

import numpy as np
import cv2
from filters.base_filter import BaseFilter


class MosaicFilter(BaseFilter):
    """모자이크 효과 필터"""

    def __init__(self):
        super().__init__("모자이크", "이미지에 모자이크 효과를 적용합니다")
        self.set_default_params({'pixel_size': 10})

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        모자이크 효과 적용

        Args:
            image: 입력 이미지 (NumPy array)
            **params: pixel_size (int, 2-50, 기본값 10)

        Returns:
            모자이크 효과가 적용된 이미지
        """
        params = self.validate_params(params)
        pixel_size = max(2, min(50, params['pixel_size']))

        height, width = image.shape[:2]

        # 픽셀 크기만큼 축소했다가 다시 확대
        temp_height = max(1, height // pixel_size)
        temp_width = max(1, width // pixel_size)

        # 축소
        temp = cv2.resize(image, (temp_width, temp_height), interpolation=cv2.INTER_LINEAR)

        # 확대 (NEAREST로 블록 효과)
        result = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

        return result.astype(np.uint8)


class GaussianBlurFilter(BaseFilter):
    """가우시안 블러 필터"""

    def __init__(self):
        super().__init__("가우시안 블러", "가우시안 분포를 이용한 부드러운 블러 효과를 적용합니다")
        self.set_default_params({'kernel_size': 5})

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        가우시안 블러 효과 적용

        Args:
            image: 입력 이미지 (NumPy array)
            **params: kernel_size (int, 3-25, 홀수만, 기본값 5)

        Returns:
            가우시안 블러가 적용된 이미지
        """
        params = self.validate_params(params)
        kernel_size = max(3, min(25, params['kernel_size']))

        # 홀수로 만들기
        if kernel_size % 2 == 0:
            kernel_size += 1

        result = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        return result.astype(np.uint8)


class AverageBlurFilter(BaseFilter):
    """평균 블러 필터"""

    def __init__(self):
        super().__init__("평균 블러", "평균값을 이용한 블러 효과를 적용합니다")
        self.set_default_params({'kernel_size': 5})

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        평균 블러 효과 적용

        Args:
            image: 입력 이미지 (NumPy array)
            **params: kernel_size (int, 3-25, 기본값 5)

        Returns:
            평균 블러가 적용된 이미지
        """
        params = self.validate_params(params)
        kernel_size = max(3, min(25, params['kernel_size']))

        result = cv2.blur(image, (kernel_size, kernel_size))

        return result.astype(np.uint8)


class MedianBlurFilter(BaseFilter):
    """중앙값 블러 필터"""

    def __init__(self):
        super().__init__("중앙값 블러", "중앙값을 이용한 블러 효과를 적용합니다 (노이즈 제거에 효과적)")
        self.set_default_params({'kernel_size': 5})

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        중앙값 블러 효과 적용

        Args:
            image: 입력 이미지 (NumPy array)
            **params: kernel_size (int, 3-25, 홀수만, 기본값 5)

        Returns:
            중앙값 블러가 적용된 이미지
        """
        params = self.validate_params(params)
        kernel_size = max(3, min(25, params['kernel_size']))

        # 홀수로 만들기
        if kernel_size % 2 == 0:
            kernel_size += 1

        result = cv2.medianBlur(image, kernel_size)

        return result.astype(np.uint8)


class SharpenFilter(BaseFilter):
    """샤프닝 필터"""

    def __init__(self):
        super().__init__("샤프닝", "이미지의 윤곽을 선명하게 만듭니다")
        self.set_default_params({'strength': 1.0})

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        샤프닝 효과 적용

        Args:
            image: 입력 이미지 (NumPy array)
            **params: strength (float, 0.5-3.0, 기본값 1.0)

        Returns:
            샤프닝이 적용된 이미지
        """
        params = self.validate_params(params)
        strength = max(0.5, min(3.0, params['strength']))

        # Unsharp masking 기법
        # 1. 가우시안 블러로 부드러운 이미지 생성
        blurred = cv2.GaussianBlur(image, (0, 0), 3)

        # 2. 원본 - 블러 = 디테일 추출
        # 3. 원본 + (디테일 * strength) = 샤프닝
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)

        # 범위 클리핑
        result = np.clip(sharpened, 0, 255)

        return result.astype(np.uint8)


class EmbossFilter(BaseFilter):
    """엠보싱 필터"""

    def __init__(self):
        super().__init__("엠보싱", "이미지에 입체감을 주는 엠보싱 효과를 적용합니다")
        self.set_default_params({'strength': 1.0})

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        엠보싱 효과 적용

        Args:
            image: 입력 이미지 (NumPy array)
            **params: strength (float, 0.5-3.0, 기본값 1.0)

        Returns:
            엠보싱이 적용된 이미지
        """
        params = self.validate_params(params)
        strength = max(0.5, min(3.0, params['strength']))

        # Emboss 커널
        kernel = np.array([[-1, -1, 0],
                          [-1,  0, 1],
                          [ 0,  1, 1]], dtype=np.float32)

        # strength에 따라 커널 스케일 조정
        kernel = kernel * strength

        # 각 채널에 필터 적용
        result = cv2.filter2D(image, -1, kernel)

        # 중간 회색(128) 추가하여 밝기 조정
        result = np.clip(result + 128, 0, 255)

        return result.astype(np.uint8)
