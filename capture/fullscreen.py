"""
전체 화면 캡처 모듈

mss 라이브러리를 사용하여 모든 모니터의 전체 화면을 캡처합니다.
"""

import mss
from PIL import Image


def capture_fullscreen():
    """
    전체 화면 캡처 (모든 모니터)

    Returns:
        PIL.Image: 캡처된 이미지
    """
    try:
        with mss.mss() as sct:
            # 모든 모니터를 포함하는 영역 (monitors[0])
            monitor = sct.monitors[0]

            # 스크린샷 캡처
            screenshot = sct.grab(monitor)

            # PIL Image로 변환
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

            return img

    except Exception as e:
        print(f"전체 화면 캡처 실패: {str(e)}")
        return None
