"""
예술적 효과 필터 모듈

카툰, 스케치, 유화, 필름 그레인, 빈티지 효과를 제공합니다.
"""

import cv2
import numpy as np

from filters.base_filter import BaseFilter


class CartoonFilter(BaseFilter):
    """카툰 효과 필터"""

    def __init__(self):
        super().__init__("카툰", "이미지를 카툰 스타일로 변환합니다")
        self.set_default_params(
            {
                "bilateral_d": 9,
                "bilateral_sigma_color": 75,
                "bilateral_sigma_space": 75,
                "edge_threshold1": 100,
                "edge_threshold2": 200,
            }
        )

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        카툰 효과 적용

        Args:
            image: 입력 이미지 (RGB)
            bilateral_d: Bilateral filter diameter
            bilateral_sigma_color: Bilateral filter sigma color
            bilateral_sigma_space: Bilateral filter sigma space
            edge_threshold1: Canny edge detection lower threshold
            edge_threshold2: Canny edge detection upper threshold

        Returns:
            카툰 효과가 적용된 이미지
        """
        bilateral_d = params.get("bilateral_d", self._default_params["bilateral_d"])
        bilateral_sigma_color = params.get(
            "bilateral_sigma_color", self._default_params["bilateral_sigma_color"]
        )
        bilateral_sigma_space = params.get(
            "bilateral_sigma_space", self._default_params["bilateral_sigma_space"]
        )
        edge_threshold1 = params.get(
            "edge_threshold1", self._default_params["edge_threshold1"]
        )
        edge_threshold2 = params.get(
            "edge_threshold2", self._default_params["edge_threshold2"]
        )

        # RGB to BGR for OpenCV
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Step 1: 색상 평탄화 (Bilateral filter)
        num_bilateral = 7  # 반복 횟수
        img_color = bgr_image.copy()
        for _ in range(num_bilateral):
            img_color = cv2.bilateralFilter(
                img_color, bilateral_d, bilateral_sigma_color, bilateral_sigma_space
            )

        # Step 2: Edge detection
        img_gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.medianBlur(img_gray, 7)
        img_edge = cv2.adaptiveThreshold(
            img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2
        )

        # Step 3: Edge를 3채널로 변환
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)

        # Step 4: 색상과 Edge 결합
        result = cv2.bitwise_and(img_color, img_edge)

        # BGR to RGB
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        return result


class SketchFilter(BaseFilter):
    """스케치 효과 필터"""

    def __init__(self):
        super().__init__("스케치", "이미지를 연필 스케치 스타일로 변환합니다")
        self.set_default_params(
            {
                "blur_sigma": 50,
                "sketch_type": "pencil",  # 'pencil' or 'charcoal'
            }
        )

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        스케치 효과 적용

        Args:
            image: 입력 이미지 (RGB)
            blur_sigma: Gaussian blur sigma
            sketch_type: 스케치 타입 ('pencil' or 'charcoal')

        Returns:
            스케치 효과가 적용된 이미지
        """
        blur_sigma = params.get("blur_sigma", self._default_params["blur_sigma"])
        sketch_type = params.get("sketch_type", self._default_params["sketch_type"])

        # RGB to BGR
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Grayscale 변환
        img_gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        # 이미지 반전
        img_invert = cv2.bitwise_not(img_gray)

        # Gaussian blur
        img_blur = cv2.GaussianBlur(img_invert, (21, 21), blur_sigma)

        # Dodge blend
        img_blend = self._dodge_blend(img_gray, img_blur)

        if sketch_type == "charcoal":
            # 숯 효과: 대비 강화
            img_blend = cv2.equalizeHist(img_blend)

        # 3채널로 변환 (RGB)
        result = cv2.cvtColor(img_blend, cv2.COLOR_GRAY2RGB)

        return result

    def _dodge_blend(self, front: np.ndarray, back: np.ndarray) -> np.ndarray:
        """
        Dodge blend mode 구현

        Args:
            front: 전경 이미지
            back: 배경 이미지

        Returns:
            블렌드된 이미지
        """
        result = front.astype(np.float32)
        back_inv = 255 - back.astype(np.float32)

        # Avoid division by zero
        back_inv = np.where(back_inv == 0, 0.001, back_inv)

        result = (result * 255) / back_inv
        result = np.clip(result, 0, 255)

        return result.astype(np.uint8)


class OilPaintingFilter(BaseFilter):
    """유화 효과 필터"""

    def __init__(self):
        super().__init__("유화", "이미지를 유화 스타일로 변환합니다")
        self.set_default_params({"size": 7, "dynRatio": 1})

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        유화 효과 적용

        Args:
            image: 입력 이미지 (RGB)
            size: Oil painting size
            dynRatio: Dynamic ratio

        Returns:
            유화 효과가 적용된 이미지
        """
        size = params.get("size", self._default_params["size"])
        dynRatio = params.get("dynRatio", self._default_params["dynRatio"])

        # RGB to BGR
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            # OpenCV의 xphoto 모듈 사용 (설치되어 있는 경우)
            result = cv2.xphoto.oilPainting(bgr_image, size, dynRatio)
        except AttributeError:
            # xphoto 모듈이 없는 경우 대체 구현
            result = self._oil_painting_fallback(bgr_image, size)

        # BGR to RGB
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        return result

    def _oil_painting_fallback(self, image: np.ndarray, size: int) -> np.ndarray:
        """
        유화 효과 대체 구현 (xphoto 없을 때)

        Args:
            image: BGR 이미지
            size: 필터 크기

        Returns:
            유화 효과 이미지
        """
        # Bilateral filter로 유사한 효과 구현
        result = image.copy()
        for _ in range(2):
            result = cv2.bilateralFilter(result, size, 75, 75)

        # 약간의 샤프닝
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) / 1.0
        result = cv2.filter2D(result, -1, kernel)

        return result


class FilmGrainFilter(BaseFilter):
    """필름 그레인 효과 필터"""

    def __init__(self):
        super().__init__("필름 그레인", "이미지에 필름 그레인을 추가합니다")
        self.set_default_params(
            {
                "intensity": 25,
                "grain_type": "gaussian",  # 'gaussian' or 'uniform'
            }
        )

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        필름 그레인 효과 적용

        Args:
            image: 입력 이미지 (RGB)
            intensity: 그레인 강도 (0-100)
            grain_type: 그레인 타입 ('gaussian' or 'uniform')

        Returns:
            필름 그레인이 추가된 이미지
        """
        intensity = params.get("intensity", self._default_params["intensity"])
        grain_type = params.get("grain_type", self._default_params["grain_type"])

        # 이미지를 float로 변환
        img_float = image.astype(np.float32)

        # 노이즈 생성
        if grain_type == "gaussian":
            noise = np.random.normal(0, intensity, image.shape)
        else:  # uniform
            noise = np.random.uniform(-intensity, intensity, image.shape)

        # 노이즈 추가
        result = img_float + noise
        result = np.clip(result, 0, 255)

        return result.astype(np.uint8)


class VintageFilter(BaseFilter):
    """빈티지 효과 필터"""

    def __init__(self):
        super().__init__("빈티지", "이미지에 빈티지 효과를 적용합니다")
        self.set_default_params(
            {
                "sepia_intensity": 0.7,
                "vignette_strength": 0.5,
                "grain_intensity": 15,
                "fade_amount": 0.2,
            }
        )

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        빈티지 효과 적용

        Args:
            image: 입력 이미지 (RGB)
            sepia_intensity: 세피아 강도 (0.0-1.0)
            vignette_strength: 비네팅 강도 (0.0-1.0)
            grain_intensity: 그레인 강도 (0-50)
            fade_amount: 페이드 양 (0.0-1.0)

        Returns:
            빈티지 효과가 적용된 이미지
        """
        sepia_intensity = params.get(
            "sepia_intensity", self._default_params["sepia_intensity"]
        )
        vignette_strength = params.get(
            "vignette_strength", self._default_params["vignette_strength"]
        )
        grain_intensity = params.get(
            "grain_intensity", self._default_params["grain_intensity"]
        )
        fade_amount = params.get("fade_amount", self._default_params["fade_amount"])

        result = image.astype(np.float32)

        # Step 1: 세피아 톤 적용
        result = self._apply_sepia(result, sepia_intensity)

        # Step 2: 비네팅 적용
        result = self._apply_vignette(result, vignette_strength)

        # Step 3: 필름 그레인 추가
        noise = np.random.normal(0, grain_intensity, image.shape)
        result = result + noise

        # Step 4: 페이딩 (약간 밝게)
        result = result + (255 - result) * fade_amount

        result = np.clip(result, 0, 255)
        return result.astype(np.uint8)

    def _apply_sepia(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """
        세피아 톤 적용

        Args:
            image: 입력 이미지 (RGB, float32)
            intensity: 세피아 강도

        Returns:
            세피아가 적용된 이미지
        """
        # 세피아 변환 행렬
        sepia_matrix = np.array(
            [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]]
        )

        # 행렬 곱셈
        sepia_img = np.dot(image, sepia_matrix.T)
        sepia_img = np.clip(sepia_img, 0, 255)

        # 강도에 따라 원본과 블렌드
        result = image * (1 - intensity) + sepia_img * intensity

        return result

    def _apply_vignette(self, image: np.ndarray, strength: float) -> np.ndarray:
        """
        비네팅 효과 적용

        Args:
            image: 입력 이미지 (RGB, float32)
            strength: 비네팅 강도

        Returns:
            비네팅이 적용된 이미지
        """
        rows, cols = image.shape[:2]

        # 중심에서의 거리 계산
        X_resultant_kernel = cv2.getGaussianKernel(cols, cols / 2)
        Y_resultant_kernel = cv2.getGaussianKernel(rows, rows / 2)
        kernel = Y_resultant_kernel * X_resultant_kernel.T
        mask = kernel / kernel.max()

        # 강도 조절
        mask = mask**strength

        # 3채널로 확장
        mask = np.stack([mask] * 3, axis=-1)

        # 이미지에 마스크 적용
        result = image * mask

        return result
