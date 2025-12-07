"""
전역 단축키 관리

pywin32의 SetWindowsHookEx를 사용하여 Low-Level Keyboard Hook을 직접 구현합니다.
이를 통해 전역 단축키를 감지하고 이벤트를 확실하게 차단(suppress)합니다.
Windows 환경에서만 동작합니다.
"""

import ctypes
import platform
import threading
from ctypes import wintypes

from PyQt5.QtCore import QObject, pyqtSignal

# Windows Virtual Key Codes
VK_F1 = 0x70
VK_F2 = 0x71
VK_F3 = 0x72
VK_F4 = 0x73
VK_CONTROL = 0x11
VK_SHIFT = 0x10

# Windows Messages
WM_KEYDOWN = 0x0100
WM_SYSKEYDOWN = 0x0104
WM_KEYUP = 0x0101
WM_SYSKEYUP = 0x0105

# Hook Constants
WH_KEYBOARD_LL = 13
HC_ACTION = 0

# GetMessage constants
PM_REMOVE = 0x0001


class KBDLLHOOKSTRUCT(ctypes.Structure):
    """Low-Level Keyboard Input Event Structure"""

    _fields_ = [
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]


class GlobalHotkeyManager(QObject):
    """
    전역 단축키 관리자 (pywin32 Low-Level Hook 기반)

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
        self._capturing = False  # 디바운스 플래그
        self._suspended = False  # 일시 중지 플래그 (설정창 등이 열려있을 때)
        self._hook_id = None
        self._hook_thread = None
        self._hook_callback = None  # Keep reference to prevent garbage collection

        # VK Code -> Action 매핑 (동적으로 변경 가능)
        self.hotkey_map = {}

        # 설정에서 단축키 로드
        self._load_hotkeys_from_settings()

        # Windows API 함수 정의
        if platform.system() == "Windows":
            self.user32 = ctypes.windll.user32
            self.kernel32 = ctypes.windll.kernel32

            # CallNextHookEx의 argtypes와 restype 명시적 정의
            self.user32.CallNextHookEx.argtypes = [
                wintypes.HHOOK,  # hhk
                ctypes.c_int,  # nCode
                wintypes.WPARAM,  # wParam
                wintypes.LPARAM,  # lParam
            ]
            self.user32.CallNextHookEx.restype = wintypes.LPARAM  # LRESULT

            # SetWindowsHookExW의 restype 정의
            self.user32.SetWindowsHookExW.restype = wintypes.HHOOK

    def _load_hotkeys_from_settings(self):
        """설정 파일에서 단축키를 로드하여 hotkey_map을 갱신합니다."""
        from config.settings import settings

        # 각 단축키 타입에 대해 설정에서 로드
        hotkey_types = ["fullscreen", "region", "window", "monitor"]
        self.hotkey_map.clear()

        for hotkey_type in hotkey_types:
            key_sequence = settings.get(f"hotkey/{hotkey_type}")
            if key_sequence:
                vk_code = self._parse_key_sequence(key_sequence)
                if vk_code is not None:
                    self.hotkey_map[vk_code] = hotkey_type

    def _parse_key_sequence(self, key_sequence: str):
        """
        QKeySequence 문자열을 Windows VK Code로 변환합니다.

        Args:
            key_sequence: PyQt5 형식의 단축키 문자열 (예: "Ctrl+Shift+F1")

        Returns:
            int: VK Code, 파싱 실패 시 None
        """
        # 마지막 키 추출 (Ctrl+Shift+F1 -> F1)
        if not key_sequence:
            return None

        parts = key_sequence.split("+")
        if not parts:
            return None

        main_key = parts[-1].strip().upper()

        # F1~F12 매핑
        f_keys = {
            "F1": 0x70,
            "F2": 0x71,
            "F3": 0x72,
            "F4": 0x73,
            "F5": 0x74,
            "F6": 0x75,
            "F7": 0x76,
            "F8": 0x77,
            "F9": 0x78,
            "F10": 0x79,
            "F11": 0x7A,
            "F12": 0x7B,
        }

        if main_key in f_keys:
            return f_keys[main_key]

        # 숫자 0~9 매핑
        if main_key.isdigit() and len(main_key) == 1:
            return 0x30 + int(main_key)  # VK_0 = 0x30, VK_9 = 0x39

        # 알파벳 A~Z 매핑
        if main_key.isalpha() and len(main_key) == 1:
            return ord(main_key)  # VK_A = 0x41, VK_Z = 0x5A

        # 특수 키 매핑
        special_keys = {
            "-": 0xBD,  # VK_OEM_MINUS
            "=": 0xBB,  # VK_OEM_PLUS
            "[": 0xDB,  # VK_OEM_4
            "]": 0xDD,  # VK_OEM_6
            ";": 0xBA,  # VK_OEM_1
            "'": 0xDE,  # VK_OEM_7
            "`": 0xC0,  # VK_OEM_3
            ",": 0xBC,  # VK_OEM_COMMA
            ".": 0xBE,  # VK_OEM_PERIOD
            "/": 0xBF,  # VK_OEM_2
            "\\": 0xDC,  # VK_OEM_5
        }

        if main_key in special_keys:
            return special_keys[main_key]

        # 파싱 실패
        return None

    def update_hotkey(self, hotkey_type: str, new_key_sequence: str):
        """
        단축키를 업데이트합니다.

        Args:
            hotkey_type: 단축키 타입 ("fullscreen", "region", "window", "monitor")
            new_key_sequence: 새 키 시퀀스 (예: "Ctrl+Shift+F5")
        """
        # 기존 매핑에서 해당 타입 제거
        keys_to_remove = [k for k, v in self.hotkey_map.items() if v == hotkey_type]
        for key in keys_to_remove:
            del self.hotkey_map[key]

        # 새 매핑 추가
        vk_code = self._parse_key_sequence(new_key_sequence)
        if vk_code is not None:
            self.hotkey_map[vk_code] = hotkey_type

    def start(self):
        """단축키 리스너 시작"""
        if self._hook_id is not None:
            return

        if platform.system() != "Windows":
            print("Warning: Global hotkey manager only works on Windows.")
            return

        # Hook을 별도 스레드에서 실행 (메시지 루프 필요)
        self._hook_thread = threading.Thread(target=self._run_hook, daemon=True)
        self._hook_thread.start()

    def stop(self):
        """단축키 리스너 중지"""
        if self._hook_id is not None and platform.system() == "Windows":
            self.user32.UnhookWindowsHookEx(self._hook_id)
            self._hook_id = None
            self._hook_callback = None

    def _run_hook(self):
        """Hook 메시지 루프 실행"""
        # Hook 프로시저 콜백 타입 정의
        # Low-Level Hook의 반환 타입은 LRESULT (c_long)
        HOOKPROC = ctypes.WINFUNCTYPE(
            wintypes.LPARAM,  # LRESULT
            ctypes.c_int,  # nCode
            wintypes.WPARAM,  # wParam
            wintypes.LPARAM,  # lParam
        )

        # Hook 프로시저 생성 및 참조 유지
        self._hook_callback = HOOKPROC(self._keyboard_hook_proc)

        # Hook 설치 (Low-Level Hook은 hMod를 NULL로 전달)
        self._hook_id = self.user32.SetWindowsHookExW(
            WH_KEYBOARD_LL,
            self._hook_callback,
            None,  # hMod는 NULL (Low-Level Hook의 경우)
            0,  # dwThreadId는 0 (모든 스레드)
        )

        if not self._hook_id:
            error_code = self.kernel32.GetLastError()
            print(f"Failed to install keyboard hook. Error code: {error_code}")
            return

        # 메시지 루프 실행
        msg = wintypes.MSG()
        while self.user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
            self.user32.TranslateMessage(ctypes.byref(msg))
            self.user32.DispatchMessageW(ctypes.byref(msg))

    def _keyboard_hook_proc(self, nCode, wParam, lParam):
        """
        Low-Level Keyboard Hook 프로시저

        반환값:
        - 1: 이벤트 차단 (다른 애플리케이션으로 전달 안됨)
        - CallNextHookEx: 이벤트 통과
        """
        if nCode == HC_ACTION:
            # lParam을 KBDLLHOOKSTRUCT 포인터로 캐스팅
            kb_struct = ctypes.cast(lParam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
            vk_code = kb_struct.vkCode

            if wParam in (WM_KEYDOWN, WM_SYSKEYDOWN):
                if vk_code in self.hotkey_map:
                    # Ctrl과 Shift 상태 확인
                    ctrl_down = (self.user32.GetKeyState(VK_CONTROL) & 0x8000) != 0
                    shift_down = (self.user32.GetKeyState(VK_SHIFT) & 0x8000) != 0

                    if ctrl_down and shift_down:
                        action = self.hotkey_map[vk_code]
                        self._trigger_action(action)
                        return 1  # 이벤트 차단 (중요!)

        # 이벤트 통과
        return self.user32.CallNextHookEx(self._hook_id, nCode, wParam, lParam)

    def _trigger_action(self, action):
        """액션 실행 및 시그널 발생"""
        # 일시 중지 상태이면 무시 (설정창이 열려있을 때)
        if self._suspended:
            return

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

    def suspend(self):
        """단축키 일시 중지 (설정창 등이 열려있을 때 사용)"""
        self._suspended = True

    def resume(self):
        """단축키 재개"""
        self._suspended = False

    def is_suspended(self):
        """일시 중지 상태 확인"""
        return self._suspended

    def reset_capture_state(self):
        """캡처 상태 리셋 (외부에서 호출)"""
        self._capturing = False

    def register_hotkeys(self):
        """호환성을 위한 메서드 (start 호출)"""
        self.start()

    def unregister_hotkeys(self):
        """호환성을 위한 메서드 (stop 호출)"""
        self.stop()
