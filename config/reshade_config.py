"""
ReShade 프리셋 설정 관리 모듈

ReShade 프리셋을 JSON 파일로 저장하고 관리합니다.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from filters.reshade_filters import ReShadeCompositeFilter
from filters.reshade_parser import ReShadeParser
from utils.resource_path import get_user_data_path


class ReShadePresetManager:
    """ReShade 프리셋 관리자"""

    def __init__(self, config_dir: str = "config/reshade_presets"):
        """
        Args:
            config_dir: 프리셋 설정 파일을 저장할 디렉토리
        """
        # exe가 있는 폴더에 사용자 데이터 저장
        self.config_dir = Path(get_user_data_path(config_dir))
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.presets: Dict[str, dict] = {}
        self.load_all_presets()

    def load_all_presets(self):
        """저장된 모든 프리셋을 로드합니다"""
        if not self.config_dir.exists():
            return

        for preset_file in self.config_dir.glob("*.json"):
            try:
                with open(preset_file, "r", encoding="utf-8") as f:
                    preset_data = json.load(f)
                    preset_name = preset_data.get("name", preset_file.stem)
                    self.presets[preset_name] = preset_data
            except Exception as e:
                print(f"프리셋 로드 실패 ({preset_file.name}): {e}")

    def save_preset(
        self,
        preset_name: str,
        effects: dict,
        unsupported_effects: List[str],
        techniques: List[str] = None,
    ) -> bool:
        """
        프리셋을 JSON 파일로 저장합니다

        Args:
            preset_name: 프리셋 이름
            effects: 효과 파라미터 딕셔너리
            unsupported_effects: 미구현 효과 리스트
            techniques: Techniques 적용 순서 (선택사항)

        Returns:
            성공 여부
        """
        try:
            preset_data = {
                "name": preset_name,
                "effects": effects,
                "unsupported_effects": unsupported_effects,
                "techniques": techniques or list(effects.keys()),  # 순서 저장
            }

            safe_filename = "".join(
                c if c.isalnum() or c in (" ", "_", "-") else "_" for c in preset_name
            )
            filepath = self.config_dir / f"{safe_filename}.json"

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(preset_data, f, indent=2, ensure_ascii=False)

            self.presets[preset_name] = preset_data
            return True

        except Exception as e:
            print(f"프리셋 저장 실패: {e}")
            return False

    def load_preset_from_ini(
        self, ini_path: str, custom_name: Optional[str] = None, save: bool = False
    ) -> Tuple[Optional[str], Optional[ReShadeCompositeFilter], List[str]]:
        """
        INI 파일에서 프리셋을 로드하고 필터를 생성합니다

        Args:
            ini_path: ReShade INI 파일 경로
            custom_name: 사용자 지정 이름 (None이면 파일명 사용)
            save: True이면 프리셋을 JSON으로 저장

        Returns:
            Tuple[preset_name, filter, unsupported_effects]
            실패 시 (None, None, [])
        """
        try:
            parser = ReShadeParser(ini_path)
            effects, unsupported_effects = parser.parse()

            if not effects:
                return None, None, unsupported_effects

            preset_name = custom_name or parser.get_preset_name()

            # INI 파일의 Techniques 순서를 함께 전달
            reshade_filter = ReShadeCompositeFilter(
                preset_name, effects, parser.techniques
            )

            # save=True일 때만 저장
            if save:
                self.save_preset(
                    preset_name, effects, unsupported_effects, parser.techniques
                )

            return preset_name, reshade_filter, unsupported_effects

        except Exception as e:
            print(f"INI 파일 로드 실패: {e}")
            return None, None, []

    def get_preset(
        self, preset_name: str
    ) -> Optional[Tuple[dict, ReShadeCompositeFilter]]:
        """
        저장된 프리셋을 가져옵니다

        Args:
            preset_name: 프리셋 이름

        Returns:
            (preset_data, filter) 또는 None
        """
        if preset_name not in self.presets:
            return None

        preset_data = self.presets[preset_name]
        effects = preset_data.get("effects", {})
        techniques = preset_data.get(
            "techniques", list(effects.keys())
        )  # 저장된 순서 사용

        if not effects:
            return None

        reshade_filter = ReShadeCompositeFilter(preset_name, effects, techniques)
        return preset_data, reshade_filter

    def delete_preset(self, preset_name: str) -> bool:
        """
        프리셋을 삭제합니다

        Args:
            preset_name: 삭제할 프리셋 이름

        Returns:
            성공 여부
        """
        if preset_name not in self.presets:
            return False

        try:
            preset_data = self.presets[preset_name]
            safe_filename = "".join(
                c if c.isalnum() or c in (" ", "_", "-") else "_"
                for c in preset_data.get("name", preset_name)
            )
            filepath = self.config_dir / f"{safe_filename}.json"

            if filepath.exists():
                filepath.unlink()

            del self.presets[preset_name]
            return True

        except Exception as e:
            print(f"프리셋 삭제 실패: {e}")
            return False

    def rename_preset(self, old_name: str, new_name: str) -> bool:
        """
        프리셋 이름을 변경합니다

        Args:
            old_name: 기존 이름
            new_name: 새 이름

        Returns:
            성공 여부
        """
        if old_name not in self.presets or new_name in self.presets:
            return False

        try:
            preset_data = self.presets[old_name]

            self.delete_preset(old_name)

            preset_data["name"] = new_name
            effects = preset_data.get("effects", {})
            unsupported = preset_data.get("unsupported_effects", [])
            techniques = preset_data.get("techniques", list(effects.keys()))

            return self.save_preset(new_name, effects, unsupported, techniques)

        except Exception as e:
            print(f"프리셋 이름 변경 실패: {e}")
            return False

    def get_all_preset_names(self) -> List[str]:
        """모든 프리셋 이름을 반환합니다"""
        return list(self.presets.keys())

    def preset_exists(self, preset_name: str) -> bool:
        """프리셋이 존재하는지 확인합니다"""
        return preset_name in self.presets
