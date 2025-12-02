"""
화면 캡처 메인 모듈

다양한 화면 캡처 방식을 통합 관리하는 메인 모듈입니다.
"""

import numpy as np
from PyQt5.QtCore import QObject, QTimer, pyqtSignal


class ScreenCapture(QObject):
    """화면 캡처 통합 관리 클래스"""

    capture_completed = pyqtSignal(np.ndarray)  # 캡처 완료 시그널
    capture_failed = pyqtSignal(str)  # 캡처 실패 시그널

    def __init__(self, parent=None):
        super().__init__(parent)
        self.delay_seconds = 0

    def set_delay(self, seconds):
        """
        캡처 지연 시간 설정

        Args:
            seconds: 지연 시간 (초)
        """
        self.delay_seconds = seconds

    def execute_capture(self, capture_func, *args, external_delay=None, **kwargs):
        """
        캡처 실행 (지연 시간 적용)

        Args:
            capture_func: 캡처 함수
            *args: 캡처 함수에 전달할 위치 인자
            external_delay: 외부 지연 시간 오버라이드 (초)
                           None이면 self.delay_seconds 사용 (toolbar 설정)
            **kwargs: 캡처 함수에 전달할 키워드 인자
        """
        # 메인 윈도우 숨기기 (이미 숨겨져 있어도 안전)
        main_window = self.parent()
        if main_window and main_window.isVisible():
            main_window.hide()

        # 지연 시간 결정
        if external_delay is not None:
            actual_delay = external_delay  # TrayService에서 전달
        else:
            actual_delay = self.delay_seconds  # toolbar 설정 사용

        # 최소 100ms 대기 (숨기기 애니메이션)
        actual_delay = max(actual_delay, 0.1)

        # 타이머 사용하여 지연 후 캡처
        QTimer.singleShot(
            int(actual_delay * 1000),
            lambda: self._do_capture(capture_func, *args, **kwargs),
        )

    def _do_capture(self, capture_func, *args, **kwargs):
        """
        실제 캡처 수행

        Args:
            capture_func: 캡처 함수
            *args, **kwargs: 캡처 함수에 전달할 인자
        """
        main_window = self.parent()

        try:
            # 콜백 기반 캡처 함수인지 확인 (region.capture_region)
            if capture_func.__name__ == "capture_region":
                # 콜백 함수를 전달하여 비동기 처리
                def handle_result(pil_image):
                    if pil_image is None:
                        self.capture_failed.emit("캡처가 취소되었습니다.")
                        if main_window:
                            main_window.show()
                        return

                    # PIL Image를 NumPy 배열로 변환
                    image_array = np.array(pil_image)

                    # RGB로 변환 (필요시)
                    if len(image_array.shape) == 2:
                        # 그레이스케일 -> RGB
                        image_array = np.stack([image_array] * 3, axis=-1)
                    elif image_array.shape[2] == 4:
                        # RGBA -> RGB
                        image_array = image_array[:, :, :3]

                    # 캡처 완료 시그널 발생
                    self.capture_completed.emit(image_array)

                    # 메인 윈도우 다시 표시
                    if main_window:
                        main_window.show()

                # RegionSelector 위젯 참조를 저장 (가비지 컬렉션 방지)
                self._region_selector = capture_func(callback=handle_result)
            else:
                # 동기 방식 캡처 함수 (fullscreen, window, monitor)
                pil_image = capture_func(*args, **kwargs)

                if pil_image is None:
                    self.capture_failed.emit("캡처가 취소되었습니다.")
                    if main_window:
                        main_window.show()
                    return

                # PIL Image를 NumPy 배열로 변환
                image_array = np.array(pil_image)

                # RGB로 변환 (필요시)
                if len(image_array.shape) == 2:
                    # 그레이스케일 -> RGB
                    image_array = np.stack([image_array] * 3, axis=-1)
                elif image_array.shape[2] == 4:
                    # RGBA -> RGB
                    image_array = image_array[:, :, :3]

                # 캡처 완료 시그널 발생
                self.capture_completed.emit(image_array)

                # 메인 윈도우 다시 표시
                if main_window:
                    main_window.show()

        except Exception as e:
            self.capture_failed.emit(f"캡처 실패: {str(e)}")
            # 메인 윈도우 다시 표시
            if main_window:
                main_window.show()
