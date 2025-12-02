"""캡처 관련 기능 Mixin"""
from PyQt5.QtWidgets import QMessageBox

from capture import fullscreen, monitor, region, window
from ui.dialogs.monitor_select_dialog import MonitorSelectDialog


class CaptureMixin:
    """화면 캡처 기능 (전체화면, 영역, 윈도우, 모니터)"""

    def setup_capture_actions(self):
        """캡처 액션 등록"""
        self.ribbon_menu.set_tool_action("전체화면", self.capture_fullscreen)
        self.ribbon_menu.set_tool_action("영역 지정", self.capture_region)
        self.ribbon_menu.set_tool_action("윈도우", self.capture_window)
        self.ribbon_menu.set_tool_action("모니터", self.capture_monitor)

    def capture_fullscreen(self):
        """전체 화면 캡처"""
        delay = self.toolbar.get_timer_delay()
        self.screen_capture.set_delay(delay)

        if delay > 0:
            self.update_status(f"{delay}초 후 전체 화면 캡처 시작...")
        else:
            self.update_status("전체 화면 캡처 중...")

        self.screen_capture.execute_capture(fullscreen.capture_fullscreen)

    def capture_region(self):
        """영역 지정 캡처"""
        delay = self.toolbar.get_timer_delay()
        self.screen_capture.set_delay(delay)

        if delay > 0:
            self.update_status(f"{delay}초 후 영역 선택 시작...")
        else:
            self.update_status("영역을 선택하세요...")

        self.screen_capture.execute_capture(region.capture_region)

    def capture_window(self):
        """활성 윈도우 캡처"""
        delay = self.toolbar.get_timer_delay()
        self.screen_capture.set_delay(delay)

        if delay > 0:
            self.update_status(f"{delay}초 후 활성 윈도우 캡처 시작...")
        else:
            self.update_status("활성 윈도우 캡처 중...")

        self.screen_capture.execute_capture(window.capture_window)

    def capture_monitor(self):
        """모니터 선택 후 캡처"""
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
        self.history_manager.add_state(image_array, "화면 캡처")

        self.update_status("캡처 완료")
        
        # 메인 윈도우 표시 및 포커스
        self.show()
        self.activateWindow()
        self.raise_()

    def on_capture_failed(self, error_message):
        """캡처 실패 시"""
        self.update_status(f"캡처 실패: {error_message}")
        QMessageBox.warning(self, "캡처 실패", error_message)
