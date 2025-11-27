"""
윈도우 캡처 모듈

현재 활성화된 윈도우를 캡처합니다.
"""

import mss
from PIL import Image
import win32gui
import win32ui
import win32con
from ctypes import windll


def capture_window():
    """
    활성 윈도우 캡처
    
    현재 활성화된 윈도우의 클라이언트 영역만 정확히 캡처합니다.
    
    Returns:
        PIL.Image: 캡처된 이미지 (실패 시 None)
    """
    try:
        # 활성 윈도우 핸들 가져오기
        hwnd = win32gui.GetForegroundWindow()
        
        if not hwnd:
            print("활성 윈도우를 찾을 수 없습니다.")
            return None
        
        # 윈도우 클라이언트 영역 좌표 가져오기
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        width = right - left
        height = bottom - top
        
        # 유효성 검사
        if width <= 0 or height <= 0:
            print("유효하지 않은 윈도우 크기입니다.")
            return None
        
        # 클라이언트 영역을 스크린 좌표로 변환
        left_top = win32gui.ClientToScreen(hwnd, (left, top))
        right_bottom = win32gui.ClientToScreen(hwnd, (right, bottom))
        
        # mss로 해당 영역 캡처
        with mss.mss() as sct:
            monitor = {
                'top': left_top[1],
                'left': left_top[0],
                'width': right_bottom[0] - left_top[0],
                'height': right_bottom[1] - left_top[1]
            }
            screenshot = sct.grab(monitor)
            
            # PIL Image로 변환
            img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
            
            return img
            
    except Exception as e:
        print(f"윈도우 캡처 실패: {str(e)}")
        return None


def get_window_list():
    """
    모든 윈도우 목록 가져오기
    
    Returns:
        list: 윈도우 타이틀 리스트
    """
    window_titles = []
    
    def enum_window_callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                window_titles.append(title)
    
    try:
        win32gui.EnumWindows(enum_window_callback, None)
        return window_titles
    except Exception as e:
        print(f"윈도우 목록 가져오기 실패: {str(e)}")
        return []


def capture_window_by_title(title):
    """
    특정 타이틀의 윈도우 캡처
    
    Args:
        title: 윈도우 타이틀
    
    Returns:
        PIL.Image: 캡처된 이미지 (실패 시 None)
    """
    try:
        # 타이틀로 윈도우 핸들 찾기
        hwnd = win32gui.FindWindow(None, title)
        
        if not hwnd:
            print(f"'{title}' 윈도우를 찾을 수 없습니다.")
            return None
        
        # 윈도우 클라이언트 영역 좌표 가져오기
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        width = right - left
        height = bottom - top
        
        # 유효성 검사
        if width <= 0 or height <= 0:
            print("유효하지 않은 윈도우 크기입니다.")
            return None
        
        # 클라이언트 영역을 스크린 좌표로 변환
        left_top = win32gui.ClientToScreen(hwnd, (left, top))
        right_bottom = win32gui.ClientToScreen(hwnd, (right, bottom))
        
        # mss로 해당 영역 캡처
        with mss.mss() as sct:
            monitor = {
                'top': left_top[1],
                'left': left_top[0],
                'width': right_bottom[0] - left_top[0],
                'height': right_bottom[1] - left_top[1]
            }
            screenshot = sct.grab(monitor)
            
            # PIL Image로 변환
            img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
            
            return img
            
    except Exception as e:
        print(f"윈도우 캡처 실패: {str(e)}")
        return None
