"""
전역 단축키 관리

pynput의 win32_event_filter를 사용하여 전역 단축키를 감지하고 이벤트를 차단(suppress)합니다.
Windows 환경에서만 동작합니다.
"""

import ctypes
import platform

from pynput import keyboard
from PyQt5.QtCore import QObject, pyqtSignal

# Windows Virtual Key Codes
VK_F1 = 0x70
VK_F2 = 0x71
VK_F3 = 0x72
VK_F4 = 0x73
VK_CONTROL = 0x11
VK_SHIFT = 0x10
WM_KEYDOWN = 0x0100
WM_SYSKEYDOWN = 0x0104
WM_KEYUP = 0x0101
WM_SYSKEYUP = 0x0105


class GlobalHotkeyManager(QObject):
    """
    전역 단축키 관리자 (pynput 기반)

    단축키 목록:
    - Ctrl+Shift+F1: 전체 화면 캡처
    - Ctrl+Shift+F2: 영역 지정 캡처
    - Ctrl+Shift+F3: 윈도우 캡처
    - Ctrl+Shift+F4: 모니터 캡처
    """

    # 단축키 트리거 시그널
    fullscreen_pressed = pyqtSignal()
    region_pressed = pyqtSignal()
    window_pressed = pyqtSignal()
    monitor_pressed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.listener = None
        self._capturing = False  # 디바운스 플래그

        # VK Code -> Action 매핑
        self.hotkey_map = {
            VK_F1: "fullscreen",
            VK_F2: "region",
            VK_F3: "window",
            VK_F4: "monitor",
        }

        # 이전에 눌린 키 추적 (KeyUp 억제용 - 선택사항이나 안정성을 위해)
        self._suppressed_keys = set()

    def start(self):
        """단축키 리스너 시작"""
        if self.listener is not None:
            return

        if platform.system() == "Windows":
            self.listener = keyboard.Listener(
                win32_event_filter=self._win32_event_filter
            )
        else:
            # Windows가 아닌 경우 일반 리스너 (Suppression 미지원)
            self.listener = keyboard.Listener(on_press=self._on_press_fallback)
            print("Warning: Event suppression is only supported on Windows.")

        self.listener.start()

    def stop(self):
        """단축키 리스너 중지"""
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
        self._suppressed_keys.clear()

    def _win32_event_filter(self, msg, data):
        """
        Windows 이벤트 필터.
        False를 반환하면 이벤트가 시스템의 다른 곳으로 전달되지 않습니다.
        """
        if msg in (WM_KEYDOWN, WM_SYSKEYDOWN):
            if data.vkCode in self.hotkey_map:
                # Ctrl과 Shift 상태 확인
                ctrl_down = (ctypes.windll.user32.GetKeyState(VK_CONTROL) & 0x8000) != 0
                shift_down = (ctypes.windll.user32.GetKeyState(VK_SHIFT) & 0x8000) != 0

                if ctrl_down and shift_down:
                    action = self.hotkey_map[data.vkCode]
                    self._trigger_action(action)
                    self._suppressed_keys.add(data.vkCode)
                    return False  # 이벤트 차단

        elif msg in (WM_KEYUP, WM_SYSKEYUP):
            if data.vkCode in self._suppressed_keys:
                self._suppressed_keys.remove(data.vkCode)
                return False  # KeyUp 이벤트도 차단하여 깔끔하게 처리

        return True  # 이벤트 통과

    def _on_press_fallback(self, key):
        """Non-Windows용 폴백 (기능 제한적)"""
        # 구현 생략 (Windows 타겟 프로젝트)
        pass

    def _trigger_action(self, action):
        """액션 실행 및 시그널 발생"""
        if self._capturing:
            return

        self._capturing = True

        if action == "fullscreen":
            self.fullscreen_pressed.emit()
        elif action == "region":
            self.region_pressed.emit()
        elif action == "window":
            self.window_pressed.emit()
        elif action == "monitor":
            self.monitor_pressed.emit()

    def reset_capture_state(self):
        """캡처 상태 리셋 (외부에서 호출)"""
        self._capturing = False

    def register_hotkeys(self):
        """호환성을 위한 메서드 (start 호출)"""
        self.start()

    def unregister_hotkeys(self):
        """호환성을 위한 메서드 (stop 호출)"""
        self.stop()
