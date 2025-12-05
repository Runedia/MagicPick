"""
화면 캡처 메인 모듈

다양한 화면 캡처 방식을 통합 관리하는 메인 모듈입니다.
"""

import numpy as np
from PyQt5.QtCore import QObject, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication


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
            # 즉시 UI 업데이트 강제 실행
            QApplication.processEvents()

        # 지연 시간 결정
        if external_delay is not None:
            user_delay = external_delay  # TrayService에서 전달
        else:
            user_delay = self.delay_seconds  # toolbar 설정 사용

        # 사용자 지연 시간이 있으면 먼저 대기
        if user_delay > 0:
            # 사용자 지정 지연 후 윈도우 숨김 확인 시작
            QTimer.singleShot(
                int(user_delay * 1000),
                lambda: self._wait_for_hide_and_capture(capture_func, *args, **kwargs),
            )
        else:
            # 즉시 윈도우 숨김 확인 시작
            self._wait_for_hide_and_capture(capture_func, *args, **kwargs)

    def _wait_for_hide_and_capture(self, capture_func, *args, **kwargs):
        """
        윈도우가 완전히 숨겨질 때까지 대기 후 캡처

        Args:
            capture_func: 캡처 함수
            *args, **kwargs: 캡처 함수에 전달할 인자
        """
        main_window = self.parent()

        # 폴링 카운터 초기화 (최대 1초 = 20회 * 50ms)
        self._hide_check_count = 0
        self._max_hide_checks = 20  # 최대 20회 체크 (1초)
        self._capture_args = (capture_func, args, kwargs)

        # 50ms마다 윈도우 숨김 상태 확인
        self._check_window_hidden()

    def _check_window_hidden(self):
        """윈도우가 완전히 숨겨졌는지 확인하고 캡처 실행"""
        main_window = self.parent()
        capture_func, args, kwargs = self._capture_args

        # 윈도우가 완전히 숨겨졌거나 최대 체크 횟수 도달
        if not main_window or not main_window.isVisible():
            # Qt 상태로는 숨겨졌지만, Windows 애니메이션이 아직 진행 중일 수 있음
            # 200ms 추가 대기 후 캡처 (Windows fade-out 애니메이션 시간 고려)
            QTimer.singleShot(
                200, lambda: self._do_capture(capture_func, *args, **kwargs)
            )
        elif self._hide_check_count >= self._max_hide_checks:
            # 타임아웃 - 200ms 추가 대기 후 강제 진행
            QTimer.singleShot(
                200, lambda: self._do_capture(capture_func, *args, **kwargs)
            )
        else:
            # 아직 숨겨지지 않음 - 50ms 후 다시 체크
            self._hide_check_count += 1
            QTimer.singleShot(50, self._check_window_hidden)

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
