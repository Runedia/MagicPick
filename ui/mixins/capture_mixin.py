"""캡처 관련 기능 Mixin"""

from PIL import Image
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QMessageBox

from capture import fullscreen, monitor, region, window
from config.settings import settings
from config.translations import tr
from ui.dialogs.monitor_select_dialog import MonitorSelectDialog


class CaptureMixin:
    """화면 캡처 기능 (전체화면, 영역, 윈도우, 모니터)"""

    def setup_capture_actions(self):
        """캡처 액션 등록"""
        self.ribbon_menu.set_tool_action("capture.fullscreen", self.capture_fullscreen)
        self.ribbon_menu.set_tool_action("capture.region", self.capture_region)
        self.ribbon_menu.set_tool_action("capture.window", self.capture_window)
        self.ribbon_menu.set_tool_action("capture.monitor", self.capture_monitor)

    def _prepare_for_capture(self):
        """
        캡처 시작 전 정리 작업
        - 자르기 모드 취소
        - 열린 다이얼로그 닫기
        """
        from PyQt5.QtWidgets import QDialog

        # 자르기 모드 취소
        if hasattr(self, "image_viewer") and self.image_viewer.is_crop_mode():
            self.image_viewer.cancel_crop()

        # 열린 모달 다이얼로그 닫기
        for widget in self.findChildren(QDialog):
            if widget.isVisible() and widget.isModal():
                widget.reject()

    def capture_fullscreen(self):
        """전체 화면 캡처"""
        self._prepare_for_capture()

        delay = self.toolbar.get_timer_delay()
        self.screen_capture.set_delay(delay)

        if delay > 0:
            self.update_status(f"{delay}초 후 전체 화면 캡처 시작...")
        else:
            self.update_status("전체 화면 캡처 중...")

        self.screen_capture.execute_capture(fullscreen.capture_fullscreen)

    def capture_region(self):
        """영역 지정 캡처"""
        self._prepare_for_capture()

        delay = self.toolbar.get_timer_delay()
        self.screen_capture.set_delay(delay)

        if delay > 0:
            self.update_status(f"{delay}초 후 영역 선택 시작...")
        else:
            self.update_status("영역을 선택하세요...")

        self.screen_capture.execute_capture(region.capture_region)

    def capture_window(self):
        """활성 윈도우 캡처"""
        self._prepare_for_capture()

        delay = self.toolbar.get_timer_delay()
        self.screen_capture.set_delay(delay)

        if delay > 0:
            self.update_status(f"{delay}초 후 활성 윈도우 캡처 시작...")
        else:
            self.update_status("활성 윈도우 캡처 중...")

        self.screen_capture.execute_capture(window.capture_window)

    def capture_monitor(self):
        """모니터 선택 후 캡처"""
        self._prepare_for_capture()
        # 모니터 선택 다이얼로그 표시 (메인 윈도우 상태는 변경하지 않음)
        dialog = MonitorSelectDialog(self)

        # 다이얼로그에 포커스 주기
        dialog.show()
        dialog.activateWindow()
        dialog.raise_()

        result = dialog.exec_()

        if result == dialog.Accepted:
            monitor_index = dialog.get_selected_monitor()

            if monitor_index is not None:
                delay = self.toolbar.get_timer_delay()
                self.screen_capture.set_delay(delay)

                if delay > 0:
                    self.update_status(
                        f"{delay}초 후 모니터 {monitor_index} 캡처 시작..."
                    )
                else:
                    self.update_status(f"모니터 {monitor_index} 캡처 중...")

                # monitor_index를 위치 인자로 전달
                self.screen_capture.execute_capture(
                    monitor.capture_monitor, monitor_index
                )
        else:
            self.update_status("모니터 선택 취소됨")

    def on_capture_completed(self, image_array):
        """캡처 완료 시"""
        # 기존 이미지 무시하고 새 이미지로 교체
        self.original_image = image_array.copy()
        self.current_image = image_array
        self.image_viewer.set_image(image_array)

        # 히스토리 초기화 후 새 상태 추가
        self.history_manager.clear()
        self.history_manager.add_state(image_array, tr("capture.fullscreen"))

        # 알림음 재생 (설정 확인)
        if settings.get_bool("capture_options/sound_enabled"):
            self._play_capture_sound()

        # 자동 저장 처리
        saved_path = None
        if settings.get_bool("capture/auto_save"):
            saved_path = self._auto_save_capture(image_array)

        # 클립보드 자동 복사 처리
        if settings.get_bool("capture_options/clipboard_copy"):
            self._copy_to_clipboard(image_array)

        # 상태 메시지 업데이트
        if saved_path:
            self.update_status(f"캡처 완료 - 저장됨: {saved_path}")
        else:
            self.update_status("캡처 완료")

        # 메인 윈도우 표시 및 포커스 (Windows에서 최상단으로 가져오기)
        self._bring_window_to_front()

    def _bring_window_to_front(self):
        """
        창을 최상단으로 가져옵니다.

        Windows API (pywin32)를 사용하여 자연스럽게 창을 포그라운드로 가져옵니다.
        SetForegroundWindow + AttachThreadInput 조합으로 포커스 제한을 우회합니다.
        """
        # 먼저 창 표시
        self.show()

        try:
            import win32con
            import win32gui
            import win32process

            # Qt 창의 Windows 핸들 가져오기
            hwnd = int(self.winId())

            # 현재 포그라운드 창 정보
            foreground_hwnd = win32gui.GetForegroundWindow()
            foreground_thread_id = win32process.GetWindowThreadProcessId(
                foreground_hwnd
            )[0]
            current_thread_id = win32process.GetWindowThreadProcessId(hwnd)[0]

            # 포그라운드 스레드에 연결 (포커스 제한 우회)
            if foreground_thread_id != current_thread_id:
                win32process.AttachThreadInput(
                    foreground_thread_id, current_thread_id, True
                )

            # 창을 포그라운드로 가져오기
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)  # 최소화 상태면 복원
            win32gui.SetForegroundWindow(hwnd)
            win32gui.BringWindowToTop(hwnd)

            # 스레드 연결 해제
            if foreground_thread_id != current_thread_id:
                win32process.AttachThreadInput(
                    foreground_thread_id, current_thread_id, False
                )

        except ImportError:
            # pywin32가 없으면 Qt 기본 방식 사용
            self.activateWindow()
            self.raise_()
        except Exception as e:
            # 기타 오류 시 Qt 기본 방식으로 폴백
            print(f"[창 포커스] Windows API 오류: {e}")
            self.activateWindow()
            self.raise_()

    def _play_capture_sound(self):
        """캡처 완료 알림음 재생"""
        try:
            import winsound

            # Windows 기본 카메라 셔터 소리 (또는 시스템 알림음)
            # SND_ALIAS로 시스템 소리 재생, 없으면 기본 비프음
            winsound.PlaySound(
                "SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC
            )
        except Exception:
            # Windows가 아니거나 소리 재생 실패 시 무시
            pass

    def _auto_save_capture(self, image_array):
        """
        캡처 이미지 자동 저장

        Args:
            image_array: 저장할 이미지 (NumPy array)

        Returns:
            저장된 파일 경로 또는 None
        """
        try:
            # 저장 경로 생성
            save_path = settings.get_capture_full_path()

            # PIL 이미지로 변환 후 저장
            pil_image = Image.fromarray(image_array)
            pil_image.save(str(save_path))

            return str(save_path)
        except Exception as e:
            print(f"[캡처 자동 저장 실패] {e}")
            return None

    def _copy_to_clipboard(self, image_array):
        """
        이미지를 클립보드에 복사

        Args:
            image_array: 복사할 이미지 (NumPy array)
        """
        try:
            # NumPy array를 QImage로 변환
            height, width, channel = image_array.shape
            bytes_per_line = 3 * width
            q_image = QImage(
                image_array.data, width, height, bytes_per_line, QImage.Format_RGB888
            )

            # 클립보드에 복사
            clipboard = QApplication.clipboard()
            clipboard.setImage(q_image)
        except Exception as e:
            print(f"[클립보드 복사 실패] {e}")

    def on_capture_failed(self, error_message):
        """캡처 실패 시"""
        self.update_status(f"캡처 실패: {error_message}")
        QMessageBox.warning(self, "캡처 실패", error_message)
