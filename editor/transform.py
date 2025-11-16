"""
이미지 변형 기능 모듈
- 크기 조정
- 회전 (90도, 180도, 270도, 임의 각도)
- 좌우/상하 반전
- 자르기
"""

import numpy as np
from PIL import Image


class ImageTransform:
    """이미지 변형 기능을 제공하는 클래스"""
    
    @staticmethod
    def resize(image: np.ndarray, width: int, height: int, maintain_aspect: bool = False) -> np.ndarray:
        """
        이미지 크기 조정
        
        Args:
            image: 입력 이미지 (NumPy array)
            width: 목표 너비
            height: 목표 높이
            maintain_aspect: 종횡비 유지 여부
            
        Returns:
            크기가 조정된 이미지
        """
        pil_image = Image.fromarray(image)
        
        if maintain_aspect:
            pil_image.thumbnail((width, height), Image.Resampling.LANCZOS)
        else:
            pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
        
        return np.array(pil_image)
    
    @staticmethod
    def rotate_90(image: np.ndarray) -> np.ndarray:
        """이미지를 시계방향으로 90도 회전"""
        return np.rot90(image, k=-1)
    
    @staticmethod
    def rotate_180(image: np.ndarray) -> np.ndarray:
        """이미지를 180도 회전"""
        return np.rot90(image, k=2)
    
    @staticmethod
    def rotate_270(image: np.ndarray) -> np.ndarray:
        """이미지를 시계방향으로 270도 회전 (반시계방향 90도)"""
        return np.rot90(image, k=1)
    
    @staticmethod
    def rotate_custom(image: np.ndarray, angle: float, expand: bool = True) -> np.ndarray:
        """
        이미지를 임의 각도로 회전
        
        Args:
            image: 입력 이미지
            angle: 회전 각도 (도 단위, 반시계방향)
            expand: True이면 회전된 이미지 전체를 포함하도록 캔버스 확장
            
        Returns:
            회전된 이미지
        """
        pil_image = Image.fromarray(image)
        rotated = pil_image.rotate(angle, expand=expand, resample=Image.Resampling.BICUBIC)
        return np.array(rotated)
    
    @staticmethod
    def flip_horizontal(image: np.ndarray) -> np.ndarray:
        """이미지를 좌우 반전"""
        return np.fliplr(image)
    
    @staticmethod
    def flip_vertical(image: np.ndarray) -> np.ndarray:
        """이미지를 상하 반전"""
        return np.flipud(image)
    
    @staticmethod
    def crop(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        이미지 자르기
        
        Args:
            image: 입력 이미지
            x: 시작 x 좌표
            y: 시작 y 좌표
            width: 자를 영역의 너비
            height: 자를 영역의 높이
            
        Returns:
            자른 이미지
        """
        if x < 0 or y < 0 or x + width > image.shape[1] or y + height > image.shape[0]:
            raise ValueError("자르기 영역이 이미지 범위를 벗어납니다")
        
        return image[y:y+height, x:x+width].copy()
    
    @staticmethod
    def crop_center(image: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        이미지 중앙을 기준으로 자르기
        
        Args:
            image: 입력 이미지
            width: 자를 영역의 너비
            height: 자를 영역의 높이
            
        Returns:
            중앙에서 자른 이미지
        """
        img_height, img_width = image.shape[:2]
        
        if width > img_width or height > img_height:
            raise ValueError("자르기 크기가 이미지보다 큽니다")
        
        x = (img_width - width) // 2
        y = (img_height - height) // 2
        
        return ImageTransform.crop(image, x, y, width, height)
