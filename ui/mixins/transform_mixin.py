"""변환 및 조정 관련 기능 Mixin"""
from editor.adjustments import ImageAdjustments
from editor.transform import ImageTransform
from ui.dialogs.rotate_dialog import RotateDialog


class TransformMixin:
    """이미지 변환 (회전, 반전) 및 조정 (밝기, 대비, 채도, 감마)"""

    def setup_edit_actions(self):
        """편집 메뉴 액션 설정"""
        # Undo/Redo
        self.ribbon_menu.set_tool_action("실행 취소", self.undo)
        self.ribbon_menu.set_tool_action("다시 실행", self.redo)
        self.ribbon_menu.set_tool_action("초기화", self.reset_to_original)

        # 변형 기능
        self.ribbon_menu.set_tool_action("회전", self.show_rotate_dialog)
        self.ribbon_menu.set_tool_action(
            "좌우 반전", lambda: self.apply_transform("flip_horizontal")
        )
        self.ribbon_menu.set_tool_action(
            "상하 반전", lambda: self.apply_transform("flip_vertical")
        )

        # 조정 기능 (대화상자를 통해 값 입력받는 방식으로 나중에 개선 예정)
        self.ribbon_menu.set_tool_action("밝기", lambda: self.adjust_brightness(30))
        self.ribbon_menu.set_tool_action("대비", lambda: self.adjust_contrast(30))
        self.ribbon_menu.set_tool_action("채도", lambda: self.adjust_saturation(150))
        self.ribbon_menu.set_tool_action("감마", lambda: self.adjust_gamma(1.2))

    def apply_transform(self, transform_type):
        """이미지 변형 적용"""
        if self.current_image is None:
            self.update_status("변형할 이미지가 없습니다")
            return

        try:
            # 변형 적용
            if transform_type == "rotate_90":
                result = ImageTransform.rotate_90(self.current_image)
                description = "90도 회전"
            elif transform_type == "rotate_180":
                result = ImageTransform.rotate_180(self.current_image)
                description = "180도 회전"
            elif transform_type == "rotate_270":
                result = ImageTransform.rotate_270(self.current_image)
                description = "270도 회전"
            elif transform_type == "flip_horizontal":
                result = ImageTransform.flip_horizontal(self.current_image)
                description = "좌우 반전"
            elif transform_type == "flip_vertical":
                result = ImageTransform.flip_vertical(self.current_image)
                description = "상하 반전"
            else:
                self.update_status(f"알 수 없는 변형: {transform_type}")
                return

            # 결과 적용
            self.current_image = result
            self.image_viewer.set_image(result)
            self.history_manager.add_state(result, description)
            self.update_status(f"{description} 적용됨")

        except Exception as e:
            self.update_status(f"변형 오류: {str(e)}")

    def adjust_brightness(self, value):
        """밝기 조정"""
        if self.current_image is None:
            self.update_status("조정할 이미지가 없습니다")
            return

        try:
            result = ImageAdjustments.adjust_brightness(self.current_image, value)
            self.current_image = result
            self.image_viewer.set_image(result)
            self.history_manager.add_state(result, f"밝기 {value:+d}")
            self.update_status(f"밝기 {value:+d} 적용됨")
        except Exception as e:
            self.update_status(f"밝기 조정 오류: {str(e)}")

    def adjust_contrast(self, value):
        """대비 조정"""
        if self.current_image is None:
            self.update_status("조정할 이미지가 없습니다")
            return

        try:
            result = ImageAdjustments.adjust_contrast(self.current_image, value)
            self.current_image = result
            self.image_viewer.set_image(result)
            self.history_manager.add_state(result, f"대비 {value:+d}")
            self.update_status(f"대비 {value:+d} 적용됨")
        except Exception as e:
            self.update_status(f"대비 조정 오류: {str(e)}")

    def adjust_saturation(self, value):
        """채도 조정"""
        if self.current_image is None:
            self.update_status("조정할 이미지가 없습니다")
            return

        try:
            result = ImageAdjustments.adjust_saturation(self.current_image, value)
            self.current_image = result
            self.image_viewer.set_image(result)
            self.history_manager.add_state(result, f"채도 {value}")
            self.update_status(f"채도 {value} 적용됨")
        except Exception as e:
            self.update_status(f"채도 조정 오류: {str(e)}")

    def adjust_gamma(self, gamma):
        """감마 조정"""
        if self.current_image is None:
            self.update_status("조정할 이미지가 없습니다")
            return

        try:
            result = ImageAdjustments.adjust_gamma(self.current_image, gamma)
            self.current_image = result
            self.image_viewer.set_image(result)
            self.history_manager.add_state(result, f"감마 {gamma:.2f}")
            self.update_status(f"감마 {gamma:.2f} 적용됨")
        except Exception as e:
            self.update_status(f"감마 조정 오류: {str(e)}")

    def undo(self):
        """실행 취소"""
        if not self.history_manager.can_undo():
            self.update_status("실행 취소할 작업이 없습니다")
            return

        result = self.history_manager.undo()
        if result is not None:
            self.current_image = result
            self.image_viewer.set_image(result)
            description = self.history_manager.get_current_description()
            self.update_status(f"실행 취소: {description}")

    def redo(self):
        """다시 실행"""
        if not self.history_manager.can_redo():
            self.update_status("다시 실행할 작업이 없습니다")
            return

        result = self.history_manager.redo()
        if result is not None:
            self.current_image = result
            self.image_viewer.set_image(result)
            description = self.history_manager.get_current_description()
            self.update_status(f"다시 실행: {description}")

    def reset_to_original(self):
        """이미지를 원본 상태로 초기화"""
        if self.original_image is None:
            self.update_status("초기화할 이미지가 없습니다")
            return

        # original_image를 사용하여 초기화
        self.current_image = self.original_image.copy()
        self.image_viewer.set_image(self.current_image)

        # 히스토리 초기화 및 초기 상태 저장
        self.history_manager.clear()
        self.history_manager.add_state(self.current_image, "초기화")

        self.update_status("원본 이미지로 초기화됨")

    def show_rotate_dialog(self):
        """회전 다이얼로그 표시"""
        if self.current_image is None:
            self.update_status("회전할 이미지가 없습니다")
            return

        # 현재 상태 저장 (취소 시 복원용)
        backup_current = self.current_image.copy()

        dialog = RotateDialog(self)

        # 실시간 미리보기 연결
        dialog.rotation_preview.connect(self.apply_rotation_preview)

        # 확인 버튼 클릭 시
        dialog.rotation_accepted.connect(
            lambda angle, expand: self.apply_rotation_final(angle, expand)
        )

        result = dialog.exec_()

        # 취소 버튼 클릭 시 원래 상태로 복원
        if result == dialog.Rejected:
            self.current_image = backup_current
            self.image_viewer.set_image(backup_current)
            self.update_status("회전 취소됨")

    def apply_rotation_preview(self, angle, expand):
        """이미지 회전 미리보기 (실시간)"""
        if self.current_image is None:
            return

        try:
            # 각도가 0이면 현재 이미지 표시
            if angle == 0:
                self.image_viewer.set_image(self.current_image)
                self.update_status("회전 각도: 0도")
                return

            # 현재 이미지에서 회전 적용 (미리보기용)
            result = ImageTransform.rotate_custom(self.current_image, angle, expand)
            self.image_viewer.set_image(result)
            self.update_status(f"미리보기: {angle}도 회전")

        except Exception as e:
            self.update_status(f"미리보기 오류: {str(e)}")

    def apply_rotation_final(self, angle, expand):
        """이미지 회전 최종 적용 (확인 버튼 클릭 시)"""
        if self.current_image is None:
            return

        try:
            # 각도가 0이면 변경사항 없음
            if angle == 0:
                self.update_status("회전 각도가 0도입니다")
                return

            # 회전 적용 (현재 이미지 기준)
            result = ImageTransform.rotate_custom(self.current_image, angle, expand)

            # 결과 적용
            self.current_image = result
            self.image_viewer.set_image(result)
            self.history_manager.add_state(result, f"{angle}도 회전")
            self.update_status(f"{angle}도 회전 적용됨")

        except Exception as e:
            self.update_status(f"회전 오류: {str(e)}")
