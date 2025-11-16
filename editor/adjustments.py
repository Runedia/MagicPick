"""
이미지 조정 기능 모듈
- 밝기 조절
- 대비 조절
- 채도 조절
- 감마 보정
"""

import numpy as np
from PIL import Image, ImageEnhance


class ImageAdjustments:
    """이미지 색상 및 톤 조정 기능을 제공하는 클래스"""
    
    @staticmethod
    def adjust_brightness(image: np.ndarray, value: int) -> np.ndarray:
        """
        밝기 조절
        
        Args:
            image: 입력 이미지 (NumPy array)
            value: 밝기 조절 값 (-100 ~ +100)
                   음수: 어둡게, 양수: 밝게
            
        Returns:
            밝기가 조정된 이미지
        """
        if not -100 <= value <= 100:
            raise ValueError("밝기 값은 -100에서 100 사이여야 합니다")
        
        # PIL ImageEnhance 사용
        # value를 0.0~2.0 범위로 변환 (0: 검정, 1: 원본, 2: 2배 밝기)
        factor = 1.0 + (value / 100.0)
        
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Brightness(pil_image)
        adjusted = enhancer.enhance(factor)
        
        return np.array(adjusted)
    
    @staticmethod
    def adjust_contrast(image: np.ndarray, value: int) -> np.ndarray:
        """
        대비 조절
        
        Args:
            image: 입력 이미지
            value: 대비 조절 값 (-100 ~ +100)
                   음수: 대비 감소, 양수: 대비 증가
            
        Returns:
            대비가 조정된 이미지
        """
        if not -100 <= value <= 100:
            raise ValueError("대비 값은 -100에서 100 사이여야 합니다")
        
        # value를 0.0~2.0 범위로 변환
        factor = 1.0 + (value / 100.0)
        
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Contrast(pil_image)
        adjusted = enhancer.enhance(factor)
        
        return np.array(adjusted)
    
    @staticmethod
    def adjust_saturation(image: np.ndarray, value: int) -> np.ndarray:
        """
        채도 조절
        
        Args:
            image: 입력 이미지
            value: 채도 조절 값 (0 ~ 200)
                   0: 완전 무채색 (흑백)
                   100: 원본
                   200: 2배 채도
            
        Returns:
            채도가 조정된 이미지
        """
        if not 0 <= value <= 200:
            raise ValueError("채도 값은 0에서 200 사이여야 합니다")
        
        factor = value / 100.0
        
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Color(pil_image)
        adjusted = enhancer.enhance(factor)
        
        return np.array(adjusted)
    
    @staticmethod
    def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
        """
        감마 보정
        
        Args:
            image: 입력 이미지
            gamma: 감마 값
                   < 1.0: 밝게 (중간톤 밝아짐)
                   = 1.0: 변화 없음
                   > 1.0: 어둡게 (중간톤 어두워짐)
            
        Returns:
            감마 보정된 이미지
        """
        if gamma <= 0:
            raise ValueError("감마 값은 0보다 커야 합니다")
        
        # 감마 보정 수식: output = input^(1/gamma)
        # 0-255 범위를 0-1로 정규화 후 감마 적용
        normalized = image.astype(np.float32) / 255.0
        corrected = np.power(normalized, 1.0 / gamma)
        result = (corrected * 255).clip(0, 255).astype(np.uint8)
        
        return result
    
    @staticmethod
    def adjust_sharpness(image: np.ndarray, value: int) -> np.ndarray:
        """
        선명도 조절
        
        Args:
            image: 입력 이미지
            value: 선명도 조절 값 (0 ~ 200)
                   0: 완전 블러
                   100: 원본
                   200: 매우 선명
            
        Returns:
            선명도가 조정된 이미지
        """
        if not 0 <= value <= 200:
            raise ValueError("선명도 값은 0에서 200 사이여야 합니다")
        
        factor = value / 100.0
        
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Sharpness(pil_image)
        adjusted = enhancer.enhance(factor)
        
        return np.array(adjusted)
    
    @staticmethod
    def auto_contrast(image: np.ndarray, cutoff: int = 0) -> np.ndarray:
        """
        자동 대비 조정 (히스토그램 정규화)
        
        Args:
            image: 입력 이미지
            cutoff: 히스토그램 양 끝에서 잘라낼 비율 (0-100)
            
        Returns:
            자동 대비 조정된 이미지
        """
        from PIL import ImageOps
        
        pil_image = Image.fromarray(image)
        adjusted = ImageOps.autocontrast(pil_image, cutoff=cutoff)
        
        return np.array(adjusted)
    
    @staticmethod
    def equalize_histogram(image: np.ndarray) -> np.ndarray:
        """
        히스토그램 평활화 (대비 극대화)
        
        Args:
            image: 입력 이미지
            
        Returns:
            히스토그램 평활화된 이미지
        """
        from PIL import ImageOps
        
        pil_image = Image.fromarray(image)
        adjusted = ImageOps.equalize(pil_image)
        
        return np.array(adjusted)
