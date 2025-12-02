"""UI 상태 관리 Mixin"""


class UIStateMixin:
    """UI 상태 관리 (창 상태, 줌, 메뉴, 툴바, 상태바)"""

    def restore_window_state(self):
        """저장된 창 상태 복원"""
        screen_geometry = self.screen().availableGeometry()

        if self.settings.contains("geometry"):
            self.restoreGeometry(self.settings.value("geometry"))
        else:
            width = int(screen_geometry.width() * 0.6)
            height = int(width * 3 / 4)
            x = (screen_geometry.width() - width) // 2
            y = (screen_geometry.height() - height) // 2
            self.setGeometry(x, y, width, height)

    def closeEvent(self, event):
        """창 종료 대신 숨기기 (트레이 서비스 계속 실행)"""
        # 창 geometry 저장 (기존 동작)
        self.settings.setValue("geometry", self.saveGeometry())

        # TrayService에 타이머 설정 저장 신호
        self.closing.emit()

        # 실제 종료 대신 숨기기
        event.ignore()
        self.hide()

    def resizeEvent(self, event):
        """윈도우 크기 변경 시 툴바 너비 조정"""
        # Mixin 패턴에서 super() 호출 시 MRO 문제 방지를 위해 QMainWindow 직접 호출
        from PyQt5.QtWidgets import QMainWindow
        QMainWindow.resizeEvent(self, event)
        if hasattr(self, "toolbar"):
            self.toolbar.setFixedWidth(self.centralWidget().width())

    def on_zoom_changed(self, factor):
        """배율 컨트롤 슬라이더 변경 시"""
        self.image_viewer.set_zoom(factor)

    def on_viewer_zoom_changed(self, factor):
        """이미지 뷰어 배율 변경 시 (CTRL+휠)"""
        self.zoom_control.set_zoom(factor)

    def update_status(self, message):
        """상태바 메시지 업데이트"""
        self.status_bar.showMessage(message)

    def on_tool_clicked(self, tool_name):
        """툴바 버튼 클릭 시"""
        self.ribbon_menu.execute_tool_action(tool_name)
        self.update_status(f"{tool_name} 실행됨")

    def on_menu_changed(self, menu_name):
        """리본 메뉴 변경 시"""
        tools = self.ribbon_menu.get_menu_tools(menu_name)
        self.toolbar.set_tools(tools, self.on_tool_clicked, menu_name)

        if menu_name == "셰이더":
            # 저장된 모든 프리셋 가져오기
            preset_names = self.reshade_manager.get_all_preset_names()

            # 각 프리셋에 대해 필터가 등록되어 있는지 확인하고 버튼 추가
            for preset_name in preset_names:
                # 필터가 이미 등록되어 있는지 확인
                if self.filter_manager.get_filter(preset_name) is None:
                    # 등록되어 있지 않으면 로드
                    result = self.reshade_manager.get_preset(preset_name)
                    if result is not None:
                        preset_data, reshade_filter = result
                        self.filter_manager.register_filter(reshade_filter)

                # 툴바에 버튼 추가
                if preset_name not in self.toolbar.tool_buttons:
                    self.toolbar.add_reshade_filter(
                        preset_name,
                        self.apply_reshade_filter,
                        self.delete_reshade_filter,
                        self.rename_reshade_filter,
                    )

        self.update_status(f"{menu_name} 메뉴 선택됨")
