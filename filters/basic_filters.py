"""
기본 필터 모듈

기본적인 이미지 필터들을 구현합니다.
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from filters.base_filter import BaseFilter


class GrayscaleFilter(BaseFilter):
    """회색조 필터"""

    def __init__(self):
        super().__init__("회색조", "이미지를 흑백으로 변환합니다")

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """회색조 변환 적용"""
        # RGB를 그레이스케일로 변환 (표준 가중치 사용)
        gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        # 3채널로 확장
        result = np.stack([gray, gray, gray], axis=-1)
        return result.astype(np.uint8)


class SepiaFilter(BaseFilter):
    """세피아 필터"""

    def __init__(self):
        super().__init__("세피아", "따뜻한 갈색 톤의 빈티지 효과를 적용합니다")

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """세피아 톤 적용"""
        # 세피아 변환 행렬
        sepia_matrix = np.array(
            [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]]
        )

        # 행렬 곱셈으로 세피아 적용
        result = image @ sepia_matrix.T
        # 값 범위 제한
        result = np.clip(result, 0, 255)
        return result.astype(np.uint8)


class InvertFilter(BaseFilter):
    """반전 필터"""

    def __init__(self):
        super().__init__("반전", "색상을 반전시킵니다")

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """색상 반전 적용"""
        return (255 - image).astype(np.uint8)


class SoftFilter(BaseFilter):
    """부드러운 필터"""

    def __init__(self):
        super().__init__("부드러운", "이미지를 부드럽게 만듭니다")
        self.set_default_params({"radius": 2})

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """부드러운 효과 적용"""
        radius = params.get("radius", 2)

        # NumPy를 PIL로 변환
        pil_image = Image.fromarray(image)
        # 블러 필터 적용
        blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=radius))
        # 다시 NumPy로 변환
        return np.array(blurred)


class SharpFilter(BaseFilter):
    """선명한 필터"""

    def __init__(self):
        super().__init__("선명한", "이미지를 선명하게 만듭니다")
        self.set_default_params({"factor": 2.0})

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """선명 효과 적용"""
        factor = params.get("factor", 2.0)

        # NumPy를 PIL로 변환
        pil_image = Image.fromarray(image)
        # 선명도 향상
        enhancer = ImageEnhance.Sharpness(pil_image)
        sharpened = enhancer.enhance(factor)
        # 다시 NumPy로 변환
        return np.array(sharpened)


class WarmFilter(BaseFilter):
    """따뜻한 필터"""

    def __init__(self):
        super().__init__("따뜻한", "따뜻한 색조를 추가합니다")
        self.set_default_params({"intensity": 30})

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """따뜻한 색조 적용"""
        intensity = params.get("intensity", 30)

        result = image.astype(np.float32)
        # 빨강 채널 증가, 파랑 채널 감소
        result[:, :, 0] = np.clip(result[:, :, 0] + intensity, 0, 255)  # Red
        result[:, :, 2] = np.clip(result[:, :, 2] - intensity * 0.5, 0, 255)  # Blue

        return result.astype(np.uint8)


class CoolFilter(BaseFilter):
    """차가운 필터"""

    def __init__(self):
        super().__init__("차가운", "차가운 색조를 추가합니다")
        self.set_default_params({"intensity": 30})

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """차가운 색조 적용"""
        intensity = params.get("intensity", 30)

        result = image.astype(np.float32)
        # 파랑 채널 증가, 빨강 채널 감소
        result[:, :, 2] = np.clip(result[:, :, 2] + intensity, 0, 255)  # Blue
        result[:, :, 0] = np.clip(result[:, :, 0] - intensity * 0.5, 0, 255)  # Red

        return result.astype(np.uint8)


class VignetteFilter(BaseFilter):
    """비네팅 필터"""

    def __init__(self):
        super().__init__("비네팅", "가장자리를 어둡게 만듭니다")
        self.set_default_params({"intensity": 0.5})

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """비네팅 효과 적용"""
        intensity = params.get("intensity", 0.5)

        height, width = image.shape[:2]

        # 중심에서의 거리 계산
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height / 2, width / 2

        # 정규화된 거리 (0~1)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) / max_dist

        # 비네팅 마스크 생성 (중심은 1, 가장자리는 더 어둡게)
        vignette_mask = 1 - (dist * intensity)
        vignette_mask = np.clip(vignette_mask, 0, 1)

        # 3채널로 확장
        vignette_mask = np.stack([vignette_mask] * 3, axis=-1)

        # 이미지에 마스크 적용
        result = image.astype(np.float32) * vignette_mask
        return np.clip(result, 0, 255).astype(np.uint8)
