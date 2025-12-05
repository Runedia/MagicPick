"""
설정 관리자

QSettings 기반 설정 관리 클래스입니다.
애플리케이션 전역 설정을 저장하고 불러옵니다.
"""

from pathlib import Path
from typing import Any, Dict

from PyQt5.QtCore import QSettings


class SettingsManager:
    """
    애플리케이션 설정 관리자

    QSettings를 사용하여 설정을 Windows 레지스트리 또는 INI 파일에 저장합니다.
    싱글톤 패턴으로 구현되어 애플리케이션 전체에서 동일한 인스턴스를 사용합니다.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # QSettings 초기화 (Windows 레지스트리 사용)
        self._settings = QSettings("MagicPick", "ImageEditor")

        # 기본값 정의
        self._defaults = {
            # 일반 설정
            "general/language": "ko",  # ko, en
            "general/start_with_windows": False,
            # 캡처 설정
            "capture/save_path": str(Path.home() / "Pictures" / "MagicPick"),
            "capture/filename_format": "Screenshot_{datetime}",
            "capture/default_format": "png",  # png, jpg, bmp
            "capture/auto_save": False,
            # 단축키 설정 (VK Code 기반)
            "hotkey/fullscreen": "Ctrl+Shift+F1",
            "hotkey/region": "Ctrl+Shift+F2",
            "hotkey/window": "Ctrl+Shift+F3",
            "hotkey/monitor": "Ctrl+Shift+F4",
            # 캡처 옵션
            "capture_options/sound_enabled": True,
            "capture_options/clipboard_copy": True,
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        설정 값을 가져옵니다.

        Args:
            key: 설정 키 (예: "general/language")
            default: 기본값 (None이면 _defaults에서 찾음)

        Returns:
            설정 값
        """
        if default is None:
            default = self._defaults.get(key)
        return self._settings.value(key, default)

    def get_bool(self, key: str, default: bool = None) -> bool:
        """불리언 타입 설정 값을 가져옵니다."""
        if default is None:
            default = self._defaults.get(key, False)
        value = self._settings.value(key, default)
        # QSettings는 문자열로 저장될 수 있음
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)

    def set(self, key: str, value: Any) -> None:
        """
        설정 값을 저장합니다.

        Args:
            key: 설정 키
            value: 저장할 값
        """
        self._settings.setValue(key, value)
        self._settings.sync()

    def remove(self, key: str) -> None:
        """설정 키를 삭제합니다."""
        self._settings.remove(key)
        self._settings.sync()

    def reset_to_defaults(self) -> None:
        """모든 설정을 기본값으로 초기화합니다."""
        for key, value in self._defaults.items():
            self._settings.setValue(key, value)
        self._settings.sync()

    def get_all(self) -> Dict[str, Any]:
        """현재 모든 설정을 딕셔너리로 반환합니다."""
        result = {}
        for key in self._defaults.keys():
            result[key] = self.get(key)
        return result

    def get_save_path(self) -> Path:
        """캡처 저장 경로를 반환합니다. 없으면 생성합니다."""
        path = Path(self.get("capture/save_path"))
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path

    def get_capture_filename(self) -> str:
        """
        캡처 파일 이름을 생성합니다.

        설정된 형식에 따라 파일 이름을 생성합니다.
        {datetime} → YYYYMMDD_HHMMSS
        {date} → YYYYMMDD
        {time} → HHMMSS

        Returns:
            생성된 파일 이름 (확장자 포함)
        """
        from datetime import datetime

        format_str = self.get("capture/filename_format")
        extension = self.get("capture/default_format")

        now = datetime.now()
        filename = format_str.replace("{datetime}", now.strftime("%Y%m%d_%H%M%S"))
        filename = filename.replace("{date}", now.strftime("%Y%m%d"))
        filename = filename.replace("{time}", now.strftime("%H%M%S"))

        return f"{filename}.{extension}"

    def get_capture_full_path(self) -> Path:
        """완전한 캡처 파일 경로를 반환합니다."""
        return self.get_save_path() / self.get_capture_filename()


# 편의를 위한 전역 인스턴스
settings = SettingsManager()
