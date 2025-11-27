"""
모니터 캡처 모듈

특정 모니터를 선택하여 캡처합니다.
"""

import mss
from PIL import Image


def get_monitor_list():
    """
    모니터 목록 가져오기

    Returns:
        list: 모니터 정보 리스트 [(index, info_dict), ...]
    """
    try:
        with mss.mss() as sct:
            monitors = sct.monitors[1:]  # 0번은 전체 화면이므로 제외
            monitor_list = []

            for i, monitor in enumerate(monitors, start=1):
                info = {
                    "index": i,
                    "left": monitor["left"],
                    "top": monitor["top"],
                    "width": monitor["width"],
                    "height": monitor["height"],
                }
                monitor_list.append(info)

            return monitor_list

    except Exception as e:
        print(f"모니터 목록 가져오기 실패: {str(e)}")
        return []


def capture_monitor(monitor_index=1):
    """
    특정 모니터 캡처

    Args:
        monitor_index: 모니터 인덱스 (1부터 시작)

    Returns:
        PIL.Image: 캡처된 이미지 (실패 시 None)
    """
    try:
        with mss.mss() as sct:
            # 모니터 개수 확인
            total_monitors = len(sct.monitors) - 1  # 0번 제외

            if monitor_index < 1 or monitor_index > total_monitors:
                print(f"유효하지 않은 모니터 인덱스: {monitor_index}")
                return None

            # 모니터 캡처
            monitor = sct.monitors[monitor_index]
            screenshot = sct.grab(monitor)

            # PIL Image로 변환
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

            return img

    except Exception as e:
        print(f"모니터 캡처 실패: {str(e)}")
        return None


def get_primary_monitor_index():
    """
    주 모니터 인덱스 가져오기

    Returns:
        int: 주 모니터 인덱스 (일반적으로 1)
    """
    # mss에서는 주 모니터가 일반적으로 1번
    return 1
