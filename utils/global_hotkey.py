"""
전역 단축키 관리

pynput을 사용하여 전역 단축키를 등록하고 관리합니다.
관리자 권한 없이도 대부분의 애플리케이션에서 동작합니다.
"""

from pynput import keyboard
from PyQt5.QtCore import QObject, pyqtSignal


class GlobalHotkeyManager(QObject):
    """전역 단축키 관리자"""

    # 단축키 트리거 시그널
    fullscreen_capture_triggered = pyqtSignal()
    region_capture_triggered = pyqtSignal()
    window_capture_triggered = pyqtSignal()
    monitor_capture_triggered = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.listener = None
        self.current_keys = set()

        # 기본 단축키 설정
        self.hotkeys = {
            "fullscreen": {keyboard.Key.ctrl_l, keyboard.Key.shift, keyboard.KeyCode.from_char("1")},
            "region": {keyboard.Key.ctrl_l, keyboard.Key.shift, keyboard.KeyCode.from_char("2")},
            "window": {keyboard.Key.ctrl_l, keyboard.Key.shift, keyboard.KeyCode.from_char("3")},
            "monitor": {keyboard.Key.ctrl_l, keyboard.Key.shift, keyboard.KeyCode.from_char("4")},
        }

    def start(self):
        """단축키 리스너 시작"""
        if self.listener is not None:
            return

        self.listener = keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        )
        self.listener.start()

    def stop(self):
        """단축키 리스너 중지"""
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
        self.current_keys.clear()

    def on_press(self, key):
        """키 눌림 이벤트"""
        try:
            # 현재 눌린 키 추가
            self.current_keys.add(key)

            # 각 단축키 조합 확인
            if self.current_keys == self.hotkeys["fullscreen"]:
                self.fullscreen_capture_triggered.emit()
                self.current_keys.clear()
            elif self.current_keys == self.hotkeys["region"]:
                self.region_capture_triggered.emit()
                self.current_keys.clear()
            elif self.current_keys == self.hotkeys["window"]:
                self.window_capture_triggered.emit()
                self.current_keys.clear()
            elif self.current_keys == self.hotkeys["monitor"]:
                self.monitor_capture_triggered.emit()
                self.current_keys.clear()

        except Exception as e:
            print(f"단축키 처리 오류: {e}")

    def on_release(self, key):
        """키 떼기 이벤트"""
        try:
            # 현재 눌린 키에서 제거
            if key in self.current_keys:
                self.current_keys.remove(key)
        except Exception:
            pass

    def set_hotkey(self, action, keys):
        """
        단축키 설정 변경

        Args:
            action: "fullscreen", "region", "window", "monitor"
            keys: set of pynput keys (예: {Key.ctrl_l, Key.shift, KeyCode.from_char('1')})
        """
        if action in self.hotkeys:
            self.hotkeys[action] = keys

    def get_hotkey_text(self, action):
        """
        단축키를 텍스트로 반환

        Args:
            action: "fullscreen", "region", "window", "monitor"

        Returns:
            단축키 텍스트 (예: "Ctrl+Shift+1")
        """
        if action not in self.hotkeys:
            return ""

        keys = self.hotkeys[action]
        parts = []

        for key in keys:
            if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                parts.append("Ctrl")
            elif key == keyboard.Key.shift or key == keyboard.Key.shift_r:
                parts.append("Shift")
            elif key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
                parts.append("Alt")
            elif hasattr(key, "char") and key.char:
                parts.append(key.char.upper())
            else:
                parts.append(str(key).replace("Key.", ""))

        return "+".join(sorted(parts))
