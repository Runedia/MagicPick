"""파일 관련 기능 Mixin"""


class FileMixin:
    """파일 열기, 저장, 콜백 처리"""

    def setup_file_actions(self):
        """파일 메뉴 액션 설정"""
        self.ribbon_menu.set_tool_action("열기", self.open_file)
        self.ribbon_menu.set_tool_action("저장", self.save_file)
        self.ribbon_menu.set_tool_action("다른 이름으로 저장", self.save_file_as)
        self.ribbon_menu.set_tool_action("끝내기", self.close)

    def open_file(self):
        """파일 열기"""
        image_data, file_path = self.file_manager.open_file(self)
        if image_data is not None:
            self.original_image = image_data.copy()  # 원본 이미지 저장
            self.current_image = image_data
            self.image_viewer.set_image(image_data)
            file_name = self.file_manager.get_current_file_name()
            self.update_status(f"Opened: {file_name}")

    def save_file(self):
        """현재 파일 저장"""
        if self.current_image is None:
            self.update_status("No image to save")
            return

        if self.file_manager.save_file(self, self.current_image):
            file_name = self.file_manager.get_current_file_name()
            self.update_status(f"Saved: {file_name}")

    def save_file_as(self):
        """다른 이름으로 저장"""
        if self.current_image is None:
            self.update_status("No image to save")
            return

        if self.file_manager.save_file_as(self, self.current_image):
            file_name = self.file_manager.get_current_file_name()
            self.update_status(f"Saved as: {file_name}")

    def on_file_loaded(self, image_data, file_path):
        """파일 로드 완료 시 호출"""
        self.original_image = image_data.copy()  # 원본 이미지 저장
        self.current_image = image_data
        self.image_viewer.set_image(image_data)
        # 히스토리 초기화 및 초기 상태 저장
        self.history_manager.clear()
        self.history_manager.add_state(image_data, "파일 열기")

    def on_file_saved(self, file_path):
        """파일 저장 완료 시 호출"""
        pass
