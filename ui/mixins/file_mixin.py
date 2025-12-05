"""파일 관련 기능 Mixin"""

from config.translations import tr


class FileMixin:
    """파일 열기, 저장, 콜백 처리"""

    def setup_file_actions(self):
        """파일 메뉴 액션 설정"""
        self.ribbon_menu.set_tool_action("file.open", self.open_file)
        self.ribbon_menu.set_tool_action("file.save", self.save_file)
        self.ribbon_menu.set_tool_action("file.save_as", self.save_file_as)
        self.ribbon_menu.set_tool_action("file.exit", self.on_close)

    def open_file(self):
        """파일 열기"""
        image_data, file_path = self.file_manager.open_file(self)
        if image_data is not None:
            self.original_image = image_data.copy()  # 원본 이미지 저장
            self.current_image = image_data
            self.image_viewer.set_image(image_data)

            # 히스토리 초기화 및 초기 상태 저장
            self.history_manager.clear()
            self.history_manager.add_state(image_data, tr("file.open"))

            file_name = self.file_manager.get_current_file_name()
            self.update_status(f"{tr('file.open')}: {file_name}")

    def save_file(self):
        """현재 파일 저장"""
        if self.current_image is None:
            self.update_status(tr("status.no_image"))
            return

        if self.file_manager.save_file(self, self.current_image):
            file_name = self.file_manager.get_current_file_name()
            self.update_status(f"{tr('status.saved')}: {file_name}")

    def save_file_as(self):
        """다른 이름으로 저장"""
        if self.current_image is None:
            self.update_status(tr("status.no_image"))
            return

        if self.file_manager.save_file_as(self, self.current_image):
            file_name = self.file_manager.get_current_file_name()
            self.update_status(f"{tr('status.saved')}: {file_name}")

    def on_file_loaded(self, image_data, file_path):
        """파일 로드 완료 시 호출 (시그널 콜백)"""
        self.original_image = image_data.copy()  # 원본 이미지 저장
        self.current_image = image_data
        self.image_viewer.set_image(image_data)
        # 히스토리 초기화 및 초기 상태 저장
        self.history_manager.clear()
        self.history_manager.add_state(image_data, tr("file.open"))

    def on_file_saved(self, file_path):
        """파일 저장 완료 시 호출 (시그널 콜백)"""
        pass

    def on_close(self):
        """끝내기 버튼 클릭 시 호출"""
        # 이미지가 있는 경우에만 초기화
        if self.current_image is not None:
            self.original_image = None
            self.current_image = None
            self.image_viewer.clear_image()
            self.history_manager.clear()
            self.update_status(f"{tr('status.ready')} | Ctrl+Shift+F1~F4")

        self.close()
