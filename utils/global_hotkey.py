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
        self._hook_id = None
        self._hook_thread = None
        self._hook_callback = None  # Keep reference to prevent garbage collection

        # VK Code -> Action 매핑
        self.hotkey_map = {
            VK_F1: "fullscreen",
            VK_F2: "region",
            VK_F3: "window",
            VK_F4: "monitor",
        }

        # Windows API 함수 정의
        if platform.system() == "Windows":
            self.user32 = ctypes.windll.user32
            self.kernel32 = ctypes.windll.kernel32
            
            # CallNextHookEx의 argtypes와 restype 명시적 정의
            self.user32.CallNextHookEx.argtypes = [
                wintypes.HHOOK,   # hhk
                ctypes.c_int,     # nCode
                wintypes.WPARAM,  # wParam
                wintypes.LPARAM   # lParam
            ]
            self.user32.CallNextHookEx.restype = wintypes.LPARAM  # LRESULT
            
            # SetWindowsHookExW의 restype 정의
            self.user32.SetWindowsHookExW.restype = wintypes.HHOOK

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
            ctypes.c_int,     # nCode
            wintypes.WPARAM,  # wParam
            wintypes.LPARAM   # lParam
        )

        # Hook 프로시저 생성 및 참조 유지
        self._hook_callback = HOOKPROC(self._keyboard_hook_proc)

        # Hook 설치 (Low-Level Hook은 hMod를 NULL로 전달)
        self._hook_id = self.user32.SetWindowsHookExW(
            WH_KEYBOARD_LL,
            self._hook_callback,
            None,  # hMod는 NULL (Low-Level Hook의 경우)
            0      # dwThreadId는 0 (모든 스레드)
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
