"""
전역 단축키 관리자

keyboard 라이브러리를 사용하여 시스템 전역 단축키를 등록하고 관리합니다.
Qt 시그널을 통해 스레드 안전하게 단축키 이벤트를 전달합니다.
"""

import keyboard
from PyQt5.QtCore import QObject, pyqtSignal


class GlobalHotkeyManager(QObject):
    """
    전역 키보드 단축키 관리자.

    keyboard 라이브러리를 사용하여 화면 캡처 단축키를 등록합니다.
    백그라운드 스레드에서 실행되는 콜백을 Qt 시그널로 변환하여
    메인 스레드에서 안전하게 처리할 수 있도록 합니다.

    Signals:
        fullscreen_pressed: Ctrl+Shift+F 단축키 감지
        region_pressed: Ctrl+Shift+R 단축키 감지
        window_pressed: Ctrl+Shift+W 단축키 감지
        monitor_pressed: Ctrl+Alt+F 단축키 감지
    """

    fullscreen_pressed = pyqtSignal()
    region_pressed = pyqtSignal()
    window_pressed = pyqtSignal()
    monitor_pressed = pyqtSignal()

    def __init__(self):
        """GlobalHotkeyManager 초기화"""
        super().__init__()
        self._registered_hotkeys = []
        self._capturing = False  # 디바운스 플래그

    def register_hotkeys(self):
        """
        모든 전역 단축키를 등록합니다.

        등록되는 단축키:
        - Ctrl+Shift+F: 전체 화면 캡처
        - Ctrl+Shift+R: 영역 지정 캡처
        - Ctrl+Shift+W: 윈도우 캡처
        - Ctrl+Alt+F: 모니터 캡처
        """
        hotkeys = {
            "ctrl+shift+f": self._emit_fullscreen,
            "ctrl+shift+r": self._emit_region,
            "ctrl+shift+w": self._emit_window,
            "ctrl+alt+f": self._emit_monitor,
        }

        for combo, callback in hotkeys.items():
            keyboard.add_hotkey(combo, callback, suppress=False)
            self._registered_hotkeys.append(combo)

    def unregister_hotkeys(self):
        """애플리케이션 종료 시 모든 단축키 등록을 해제합니다."""
        for combo in self._registered_hotkeys:
            keyboard.remove_hotkey(combo)
        self._registered_hotkeys.clear()

    def _emit_fullscreen(self):
        """
        전체 화면 캡처 단축키 감지 시 시그널 발생.

        디바운스를 적용하여 중복 캡처를 방지합니다.
        """
        if not self._capturing:
            self._capturing = True
            self.fullscreen_pressed.emit()

    def _emit_region(self):
        """
        영역 지정 캡처 단축키 감지 시 시그널 발생.

        디바운스를 적용하여 중복 캡처를 방지합니다.
        """
        if not self._capturing:
            self._capturing = True
            self.region_pressed.emit()

    def _emit_window(self):
        """
        윈도우 캡처 단축키 감지 시 시그널 발생.

        디바운스를 적용하여 중복 캡처를 방지합니다.
        """
        if not self._capturing:
            self._capturing = True
            self.window_pressed.emit()

    def _emit_monitor(self):
        """
        모니터 캡처 단축키 감지 시 시그널 발생.

        디바운스를 적용하여 중복 캡처를 방지합니다.
        """
        if not self._capturing:
            self._capturing = True
            self.monitor_pressed.emit()

    def reset_capture_state(self):
        """
        캡처 완료 또는 실패 후 디바운스 플래그를 리셋합니다.

        이 메서드는 캡처가 완료되거나 실패한 후 TrayService에서 호출됩니다.
        """
        self._capturing = False
