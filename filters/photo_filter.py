"""
Photo Filter 모듈

Adobe Photoshop의 Photo Filter 기능을 구현합니다.
"""

import numpy as np
import cv2
from filters.base_filter import BaseFilter


class PhotoFilter(BaseFilter):
    """Photo Filter 필터"""
    
    # 미리 정의된 필터 색상 (RGB 형식)
    PRESET_COLORS = {
        "Warming Filter (85)": (236, 150, 0),  # 주황색
        "Warming Filter (81)": (255, 180, 0),  # 황색
        "Cooling Filter (80)": (0, 128, 255),  # 청색
        "Cooling Filter (82)": (50, 150, 255),  # 연한 청색
        "Underwater": (100, 180, 128),  # 청록색
        "Sepia": (20, 66, 112),  # 세피아
        "Deep Blue": (0, 100, 255),  # 진한 청색
        "Deep Red": (200, 0, 0),  # 진한 적색
        "Deep Yellow": (255, 200, 0),  # 진한 황색
        "Deep Emerald": (0, 200, 100),  # 에메랄드
    }
    
    def __init__(self):
        super().__init__("Photo Filter", "사진에 색조 필터를 적용합니다")
        self.set_default_params({
            'filter_name': "Warming Filter (85)",
            'density': 0.25,  # 0.0 ~ 1.0
            'preserve_luminosity': True
        })
    
    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Photo Filter 적용
        
        Args:
            image: 입력 이미지 (NumPy array, RGB 형식)
            **params: 
                - filter_name: 필터 프리셋 이름 (기본: "Warming Filter (85)")
                - filter_color: 커스텀 색상 (R, G, B) - filter_name보다 우선
                - density: 필터 강도 0.0~1.0 (기본: 0.25)
                - preserve_luminosity: 밝기 유지 여부 (기본: True)
        
        Returns:
            필터가 적용된 이미지 (NumPy array, RGB 형식)
        """
        # 파라미터 추출
        filter_name = params.get('filter_name', self._default_params['filter_name'])
        density = params.get('density', self._default_params['density'])
        preserve_luminosity = params.get('preserve_luminosity', self._default_params['preserve_luminosity'])
        
        # 필터 색상 결정 (커스텀 색상이 있으면 우선 사용)
        if 'filter_color' in params:
            filter_color = params['filter_color']
        else:
            filter_color = self.PRESET_COLORS.get(filter_name, self.PRESET_COLORS["Warming Filter (85)"])
        
        # Photo Filter 알고리즘 적용
        return self._apply_photo_filter(image, filter_color, density, preserve_luminosity)
    
    def _apply_photo_filter(self, image: np.ndarray, filter_color: tuple, 
                           density: float, preserve_luminosity: bool) -> np.ndarray:
        """
        Photo Filter 알고리즘 구현
        
        Args:
            image: 입력 이미지 (RGB)
            filter_color: 필터 색상 (R, G, B)
            density: 필터 강도 (0.0 ~ 1.0)
            preserve_luminosity: 밝기 유지 여부
        
        Returns:
            필터가 적용된 이미지
        """
        # 이미지를 float32로 변환
        img_float = image.astype(np.float32) / 255.0
        
        # 필터 색상을 정규화
        filter_array = np.array(filter_color, dtype=np.float32) / 255.0
        
        if preserve_luminosity:
            # RGB를 BGR로 변환 (OpenCV는 BGR 사용)
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # LAB 색공간으로 변환하여 밝기 유지
            lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
            l_channel = lab_image[:, :, 0].astype(np.float32)
            
            # 필터 적용 (단순 블렌딩)
            filtered = img_float * (1 - density) + filter_array * density
            filtered = np.clip(filtered, 0, 1)
            
            # RGB를 BGR로 변환
            filtered_bgr = cv2.cvtColor((filtered * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # LAB로 변환하여 L 채널 복원
            filtered_lab = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2LAB)
            filtered_lab[:, :, 0] = l_channel
            
            # BGR로 다시 변환
            result_bgr = cv2.cvtColor(filtered_lab, cv2.COLOR_LAB2BGR)
            
            # BGR을 RGB로 변환
            result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            result = result.astype(np.float32) / 255.0
        else:
            # 단순 색상 블렌딩
            result = img_float * (1 - density) + filter_array * density
            result = np.clip(result, 0, 1)
        
        # uint8로 변환
        return (result * 255).astype(np.uint8)
    
    @classmethod
    def get_preset_names(cls):
        """사용 가능한 프리셋 이름 목록 반환"""
        return list(cls.PRESET_COLORS.keys())
    
    @classmethod
    def get_preset_color(cls, preset_name: str):
        """프리셋 이름으로 색상 반환"""
        return cls.PRESET_COLORS.get(preset_name)
