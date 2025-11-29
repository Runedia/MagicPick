"""
ReShade INI 파일 파서 모듈

ReShade 프리셋 파일(.ini)을 파싱하여 구현 가능한 효과를 추출합니다.
"""

import configparser
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _get_default_params_for_filter(filter_name: str) -> list[str]:
    """
    필터의 기본 파라미터 목록을 가져옵니다.

    Args:
        filter_name: 필터 이름

    Returns:
        파라미터 이름 리스트
    """
    from filters.reshade_filters import get_filter_class

    filter_class = get_filter_class(filter_name)
    if not filter_class:
        return []

    try:
        # 필터 인스턴스 생성하여 기본 파라미터 확인
        instance = filter_class()

        # __init__ 메서드의 기본값을 추출하거나 인스턴스 변수 확인
        params = []
        for attr_name in dir(instance):
            if not attr_name.startswith("_") and attr_name not in [
                "name",
                "description",
                "apply",
            ]:
                params.append(attr_name)

        return params
    except Exception:
        return []


class ReShadeParser:
    """ReShade INI 파일 파서"""

    # 구현 가능한 효과 목록
    SUPPORTED_EFFECTS = {
        "Levels": ["BlackPoint", "WhitePoint", "HighlightClipping"],
        "AdaptiveSharpen": [
            "curve_height",
            "curveslope",
            "L_overshoot",
            "L_compr_low",
            "L_compr_high",
            "D_overshoot",
            "D_compr_low",
            "D_compr_high",
            "scale_lim",
            "scale_cs",
            "pm_p",
        ],
        "LumaSharpen": [
            "sharp_strength",
            "sharp_clamp",
            "offset_bias",
            "pattern",
            "show_sharpen",
        ],
        "Vibrance": ["Vibrance", "VibranceRGBBalance"],
        "Clarity": [
            "ClarityStrength",
            "ClarityRadius",
            "ClarityOffset",
            "ClarityBlendMode",
            "ClarityDarkIntensity",
            "ClarityLightIntensity",
            "ClarityBlendIfDark",
            "ClarityBlendIfLight",
        ],
        "Curves": ["Contrast", "Formula", "Mode"],
        "Tonemap": [
            "Gamma",
            "Exposure",
            "Saturation",
            "Bleach",
            "Defog",
            "FogColor",
        ],
        "DPX": [
            "Strength",
            "RGB_Curve",
            "RGB_C",
            "Contrast",
            "Saturation",
            "Colorfulness",
        ],
        "FilmGrain": ["Intensity", "Variance", "Mean", "SignalToNoiseRatio"],
        "Sepia": ["Strength", "Tint"],
        "Vignette": ["Type", "Ratio", "Radius", "Amount", "Slope", "Center"],
        "FXAA": ["Subpix", "EdgeThreshold", "EdgeThresholdMin"],
        "LocalContrastCS": ["Strength", "WeightExponent"],
    }

    def __init__(self, ini_path: str):
        """
        Args:
            ini_path: ReShade INI 파일 경로
        """
        self.ini_path = Path(ini_path)
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        self.techniques: List[str] = []
        self.effects: Dict[str, Dict[str, Any]] = {}
        self.unsupported_effects: List[str] = []

    def parse(self) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
        """
        INI 파일을 파싱하여 효과 데이터를 추출합니다.

        Returns:
            Tuple[Dict, List]: (구현된 효과 딕셔너리, 미구현 효과 리스트)
        """
        try:
            # 섹션 헤더가 없는 경우를 처리하기 위해 DEFAULT 섹션으로 감싸기
            with open(self.ini_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 첫 줄이 섹션 헤더가 아닌 경우 [DEFAULT] 추가
            if content and not content.lstrip().startswith("["):
                content = "[DEFAULT]\n" + content

            # 메모리에서 파싱

            self.config.read_string(content)

        except Exception as e:
            raise ValueError(f"INI 파일 읽기 실패: {e}")

        self._parse_techniques()
        self._parse_effects()

        return self.effects, self.unsupported_effects

    def _parse_techniques(self):
        """활성화된 Technique 목록 추출"""
        # DEFAULT 섹션과 루트 레벨 모두 확인
        techniques_str = None

        if self.config.has_option("DEFAULT", "Techniques"):
            techniques_str = self.config.get("DEFAULT", "Techniques")
        elif self.config.has_section("") and self.config.has_option("", "Techniques"):
            techniques_str = self.config.get("", "Techniques")

        if techniques_str:
            self.techniques = [t.split("@")[0] for t in techniques_str.split(",")]

    def _parse_effects(self):
        """각 효과의 파라미터 추출"""
        for section in self.config.sections():
            effect_name = section.replace(".fx", "")

            if effect_name in self.SUPPORTED_EFFECTS:
                params = {}
                supported_params = self.SUPPORTED_EFFECTS[effect_name]

                for param in supported_params:
                    if self.config.has_option(section, param):
                        raw_value = self.config.get(section, param)
                        params[param] = self._parse_value(raw_value)

                if params:
                    self.effects[effect_name] = params
            elif effect_name in self.techniques:
                self.unsupported_effects.append(effect_name)

    def _parse_value(self, value_str: str) -> Any:
        """
        문자열 값을 적절한 타입으로 변환

        Args:
            value_str: 파라미터 값 문자열

        Returns:
            변환된 값 (int, float, tuple, bool)
        """
        value_str = value_str.strip()

        if "," in value_str:
            parts = [float(p.strip()) for p in value_str.split(",")]
            return tuple(parts)

        try:
            if "." in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            return value_str

    def get_preset_name(self) -> str:
        """INI 파일명에서 프리셋 이름 추출"""
        return self.ini_path.stem

    @classmethod
    def get_supported_effects(cls) -> List[str]:
        """구현 가능한 효과 목록 반환 (동적 검색)"""
        from filters.reshade_filters import list_available_filters

        # 동적으로 사용 가능한 필터 + 기존 SUPPORTED_EFFECTS 병합
        dynamic_filters = set(list_available_filters())
        static_filters = set(cls.SUPPORTED_EFFECTS.keys())

        return sorted(dynamic_filters | static_filters)

    @classmethod
    def is_effect_supported(cls, effect_name: str) -> bool:
        """효과가 구현 가능한지 확인 (동적 검색)"""
        from filters.reshade_filters import get_filter_class

        # 동적으로 필터 클래스 확인
        if get_filter_class(effect_name):
            return True

        # 기존 SUPPORTED_EFFECTS에도 확인
        return effect_name in cls.SUPPORTED_EFFECTS
