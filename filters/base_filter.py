"""
필터 베이스 클래스 모듈

모든 필터의 기본이 되는 추상 클래스를 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image
from PyQt5.QtCore import QObject, pyqtSignal


class BaseFilter(ABC):
    """
    필터 베이스 추상 클래스

    모든 필터는 이 클래스를 상속받아 구현해야 합니다.
    """

    def __init__(self, name: str, description: str = ""):
        """
        필터 초기화

        Args:
            name: 필터 이름
            description: 필터 설명
        """
        self.name = name
        self.description = description
        self._default_params = {}

    @abstractmethod
    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        이미지에 필터를 적용합니다 (추상 메서드)

        Args:
            image: 입력 이미지 (NumPy array, RGB 형식)
            **params: 필터별 파라미터

        Returns:
            필터가 적용된 이미지 (NumPy array, RGB 형식)
        """
        pass

    def get_default_params(self) -> Dict[str, Any]:
        """
        필터의 기본 파라미터를 반환합니다

        Returns:
            기본 파라미터 딕셔너리
        """
        return self._default_params.copy()

    def set_default_params(self, params: Dict[str, Any]):
        """
        필터의 기본 파라미터를 설정합니다

        Args:
            params: 설정할 기본 파라미터
        """
        self._default_params = params.copy()

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        파라미터 유효성 검증 및 기본값 적용

        Args:
            params: 검증할 파라미터

        Returns:
            검증 및 기본값이 적용된 파라미터
        """
        validated = self._default_params.copy()
        validated.update(params)
        return validated

    def __str__(self):
        return f"{self.name}: {self.description}"


class FilterPipeline(QObject):
    """
    필터 적용 파이프라인

    여러 필터를 순차적으로 적용하고 진행 상황을 관리합니다.
    """

    progress_updated = pyqtSignal(int, str)
    filter_applied = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._filters = []

    def add_filter(
        self, filter_obj: BaseFilter, params: Optional[Dict[str, Any]] = None
    ):
        """
        파이프라인에 필터를 추가합니다

        Args:
            filter_obj: 추가할 필터 객체
            params: 필터 파라미터 (None이면 기본값 사용)
        """
        if params is None:
            params = filter_obj.get_default_params()
        self._filters.append((filter_obj, params))

    def clear(self):
        """파이프라인에서 모든 필터를 제거합니다"""
        self._filters.clear()

    def apply_pipeline(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        파이프라인의 모든 필터를 순차적으로 적용합니다

        Args:
            image: 입력 이미지 (NumPy array)

        Returns:
            필터가 적용된 최종 이미지 (실패 시 None)
        """
        if not self._filters:
            self.error_occurred.emit("파이프라인에 필터가 없습니다")
            return None

        result = image.copy()
        total_filters = len(self._filters)

        for idx, (filter_obj, params) in enumerate(self._filters):
            try:
                progress = int((idx / total_filters) * 100)
                self.progress_updated.emit(progress, f"{filter_obj.name} 적용 중...")

                result = filter_obj.apply(result, **params)
                self.filter_applied.emit(result)

            except Exception as e:
                error_msg = f"{filter_obj.name} 필터 적용 중 오류 발생: {str(e)}"
                self.error_occurred.emit(error_msg)
                return None

        self.progress_updated.emit(100, "완료")
        return result


class FilterManager(QObject):
    """
    필터 관리자

    등록된 모든 필터를 관리하고 필터 적용을 중재합니다.
    """

    filter_started = pyqtSignal(str)
    filter_completed = pyqtSignal(np.ndarray, float)
    filter_failed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._registered_filters = {}

    def register_filter(self, filter_obj: BaseFilter):
        """
        필터를 등록합니다

        Args:
            filter_obj: 등록할 필터 객체
        """
        self._registered_filters[filter_obj.name] = filter_obj

    def unregister_filter(self, filter_name: str):
        """
        필터 등록을 해제합니다

        Args:
            filter_name: 해제할 필터 이름
        """
        if filter_name in self._registered_filters:
            del self._registered_filters[filter_name]

    def get_filter(self, filter_name: str) -> Optional[BaseFilter]:
        """
        등록된 필터를 가져옵니다

        Args:
            filter_name: 필터 이름

        Returns:
            필터 객체 (없으면 None)
        """
        return self._registered_filters.get(filter_name)

    def get_all_filters(self) -> Dict[str, BaseFilter]:
        """
        등록된 모든 필터를 반환합니다

        Returns:
            필터 이름과 객체의 딕셔너리
        """
        return self._registered_filters.copy()

    def apply_filter(
        self, image: np.ndarray, filter_name: str, **params
    ) -> Optional[np.ndarray]:
        """
        이미지에 필터를 적용합니다

        Args:
            image: 입력 이미지 (NumPy array)
            filter_name: 적용할 필터 이름
            **params: 필터 파라미터

        Returns:
            필터가 적용된 이미지 (실패 시 None)
        """
        import time

        filter_obj = self.get_filter(filter_name)
        if filter_obj is None:
            self.filter_failed.emit(f"'{filter_name}' 필터를 찾을 수 없습니다")
            return None

        try:
            self.filter_started.emit(filter_name)
            start_time = time.time()

            validated_params = filter_obj.validate_params(params)

            if params.get("_enable_performance_logging", False):
                validated_params["_enable_performance_logging"] = True

            result = filter_obj.apply(image, **validated_params)

            elapsed_time = time.time() - start_time
            self.filter_completed.emit(result, elapsed_time)

            return result

        except Exception as e:
            error_msg = f"'{filter_name}' 필터 적용 중 오류 발생: {str(e)}"
            self.filter_failed.emit(error_msg)
            return None


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """
    PIL Image를 NumPy array로 변환합니다

    Args:
        image: PIL Image 객체

    Returns:
        NumPy array (RGB 형식)
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """
    NumPy array를 PIL Image로 변환합니다

    Args:
        array: NumPy array (RGB 형식)

    Returns:
        PIL Image 객체
    """
    array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)
