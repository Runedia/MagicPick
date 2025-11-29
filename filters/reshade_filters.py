"""
ReShade 스타일 필터 구현 모듈

ReShade 프리셋의 효과를 구현하는 필터들을 정의합니다.
"""

import importlib
import inspect
from typing import Dict, Type

import numpy as np

from .base_filter import BaseFilter

# 동적 import를 위한 캐시
_FILTER_CLASSES_CACHE: Dict[str, Type[BaseFilter]] = {}


def _discover_filter_classes() -> Dict[str, Type[BaseFilter]]:
    """
    filters.reshade 패키지의 모든 필터 클래스를 자동으로 검색합니다.

    PyInstaller 호환성을 위해 __init__.py의 __all__ 리스트를 사용합니다.

    Returns:
        Dict[효과명, 필터클래스]: 효과 이름과 필터 클래스 매핑
    """
    if _FILTER_CLASSES_CACHE:
        return _FILTER_CLASSES_CACHE

    import filters.reshade as reshade_pkg

    # __all__에 정의된 모듈만 import (PyInstaller 호환)
    for module_name in reshade_pkg.__all__:
        if module_name == "hlsl_helpers":
            continue

        try:
            module = importlib.import_module(f"filters.reshade.{module_name}")

            # 모듈 내 BaseFilter 상속 클래스 찾기
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, BaseFilter)
                    and obj is not BaseFilter
                    and obj.__module__ == module.__name__
                ):
                    # 필터 클래스명에서 효과명 추출
                    # 예: AdaptiveSharpenFilterAccurate -> AdaptiveSharpen
                    effect_name = name.replace("Filter", "").replace("Accurate", "")

                    # 특수 케이스 처리
                    if "PD80_" in name or "PD80" in name:
                        # PD80_BlacknWhiteFilter -> PD80_BlacknWhite
                        effect_name = name.replace("Filter", "")

                    _FILTER_CLASSES_CACHE[effect_name] = obj

        except (ImportError, AttributeError):
            # 일부 모듈은 헬퍼나 미완성일 수 있음
            pass

    return _FILTER_CLASSES_CACHE


def get_filter_class(effect_name: str) -> Type[BaseFilter] | None:
    """
    효과 이름으로 필터 클래스를 가져옵니다.

    Args:
        effect_name: ReShade 효과 이름 (예: "AdaptiveSharpen", "PD80_03_Levels")

    Returns:
        필터 클래스 또는 None
    """
    filters = _discover_filter_classes()
    return filters.get(effect_name)


def list_available_filters() -> list[str]:
    """사용 가능한 모든 필터 효과명 반환"""
    filters = _discover_filter_classes()
    return sorted(filters.keys())


class ReShadeCompositeFilter(BaseFilter):
    """
    ReShade 복합 필터

    여러 ReShade 효과를 조합하여 적용하는 필터
    """

    def __init__(self, preset_name: str, effects: dict):
        """
        Args:
            preset_name: 프리셋 이름
            effects: 효과 파라미터 딕셔너리
        """
        super().__init__(name=preset_name, description=f"ReShade 프리셋: {preset_name}")
        self.effects = effects
        self._setup_effect_filters()

    def _setup_effect_filters(self):
        """효과별 필터 인스턴스 생성 (동적 로딩)"""
        self.effect_filters = {}

        # 모든 필터 클래스 검색
        available_filters = _discover_filter_classes()

        # 요청된 효과에 대해 필터 인스턴스 생성
        for effect_name in self.effects.keys():
            filter_class = get_filter_class(effect_name)

            if filter_class:
                try:
                    self.effect_filters[effect_name] = filter_class()
                except Exception as e:
                    print(f"[ReShade] 필터 생성 실패: {effect_name} - {e}")
            else:
                # 필터 클래스를 찾지 못한 경우 (미구현)
                pass

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        복합 필터 적용 (효과를 순차적으로 적용)

        Args:
            image: 입력 이미지 (NumPy array, RGB)
            **params: 추가 파라미터 (현재 미사용)

        Returns:
            필터 적용된 이미지
        """
        import time

        enable_performance_logging = params.get("_enable_performance_logging", False)

        result = image.copy()

        if enable_performance_logging:
            print(f"\n{'=' * 80}")
            print(f"[ReShade Performance] 프리셋: {self.name}")
            print(
                f"[ReShade Performance] 이미지 크기: {image.shape[1]}x{image.shape[0]}"
            )
            print(f"{'=' * 80}")
            total_start = time.perf_counter()

        effect_order = [
            "LevelIO",
            "WhitepointFixer",
            "Denoise",
            "PD80_BlacknWhite",
            "AdaptiveSharpen",
            "LumaSharpen",
            "FineSharp",
            "Deblur",
            "FilmicAnamorphSharpen",
            "Smart_Sharp",
            "PD80_03_Curved_Levels",
            "PD80_03_Levels",
            "PD80_CorrectContrast",
            "PD80_CorrectColor",
            "PD80_ColorGamut",
            "PD80_ColorSpaceCurves",
            "PD80_SelectiveColor",
            "PD80_04_Selective_Color_v2",
            "PD80_04_Saturation_Limit",
            "RemoveTint",
            "LevelsPlus",
            "Levels",
            "Curves",
            "Tonemap",
            "FakeHDR",
            "PD80_02_Bloom",
            "MagicBloom",
            "FilmicPass",
            "DPX",
            "Vibrance",
            "PD80_04_Technicolor",
            "Clarity",
            "PD80_05_Sharpening",
            "LocalContrastCS",
            "Sepia",
            "FilmGrain",
            "PD80_06_Film_Grain",
            "Vignette",
            "PD80_06_Chromatic_Aberration",
            "FXAA",
            "ASCII",
        ]

        effect_times = []

        for effect_name in effect_order:
            if effect_name in self.effect_filters:
                effect_params = self.effects.get(effect_name, {})

                if enable_performance_logging:
                    effect_start = time.perf_counter()
                    result = self.effect_filters[effect_name].apply(
                        result, **effect_params
                    )
                    effect_end = time.perf_counter()
                    effect_time = (effect_end - effect_start) * 1000
                    effect_times.append((effect_name, effect_time))
                    print(f"  [{effect_name:35s}] {effect_time:8.2f} ms")
                else:
                    result = self.effect_filters[effect_name].apply(
                        result, **effect_params
                    )

        if enable_performance_logging:
            total_end = time.perf_counter()
            total_time = (total_end - total_start) * 1000

            print(f"{'-' * 80}")
            print(f"  {'적용된 효과 수':35s}  {len(effect_times)}")
            print(f"  {'총 실행 시간':35s}  {total_time:8.2f} ms")
            print(
                f"  {'평균 효과당 시간':35s}  {total_time / max(1, len(effect_times)):8.2f} ms"
            )
            print(f"{'=' * 80}\n")

        return result
