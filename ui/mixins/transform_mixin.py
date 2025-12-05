"""변환 및 조정 관련 기능 Mixin"""

from config.translations import tr
from editor.transform import ImageTransform
from ui.dialogs.adjustment_dialog import AdjustmentDialog
from ui.dialogs.rotate_dialog import RotateDialog


class TransformMixin:
    """이미지 변환 (회전, 반전, 자르기) 및 조정 (밝기, 대비, 채도, 감마)"""

    def setup_edit_actions(self):
        """편집 메뉴 액션 설정"""
        # Undo/Redo
        self.ribbon_menu.set_tool_action("edit.undo", self.undo)
        self.ribbon_menu.set_tool_action("edit.redo", self.redo)
        self.ribbon_menu.set_tool_action("edit.reset", self.reset_to_original)

        # 변형 기능
        self.ribbon_menu.set_tool_action("edit.rotate", self.show_rotate_dialog)
        self.ribbon_menu.set_tool_action(
            "edit.flip_horizontal", lambda: self.apply_transform("flip_horizontal")
        )
        self.ribbon_menu.set_tool_action(
            "edit.flip_vertical", lambda: self.apply_transform("flip_vertical")
        )
        self.ribbon_menu.set_tool_action("edit.crop", self.start_crop_mode)
        self.ribbon_menu.set_tool_action("edit.resize", self.show_resize_dialog)

        # 조정 기능 (통합 다이얼로그 사용)
        self.ribbon_menu.set_tool_action("tone.brightness", self.show_adjustment_dialog)
        self.ribbon_menu.set_tool_action("tone.contrast", self.show_adjustment_dialog)
        self.ribbon_menu.set_tool_action("tone.saturation", self.show_adjustment_dialog)
        self.ribbon_menu.set_tool_action("tone.gamma", self.show_adjustment_dialog)

    def apply_transform(self, transform_type):
        """이미지 변형 적용"""
        if self.current_image is None:
            self.update_status(tr("status.no_image"))
            return

        try:
            # 변형 적용
            if transform_type == "rotate_90":
                result = ImageTransform.rotate_90(self.current_image)
                description = "90° " + tr("edit.rotate")
            elif transform_type == "rotate_180":
                result = ImageTransform.rotate_180(self.current_image)
                description = "180° " + tr("edit.rotate")
            elif transform_type == "rotate_270":
                result = ImageTransform.rotate_270(self.current_image)
                description = "270° " + tr("edit.rotate")
            elif transform_type == "flip_horizontal":
                result = ImageTransform.flip_horizontal(self.current_image)
                description = tr("edit.flip_horizontal")
            elif transform_type == "flip_vertical":
                result = ImageTransform.flip_vertical(self.current_image)
                description = tr("edit.flip_vertical")
            else:
                self.update_status(tr("status.unknown_transform", type=transform_type))
                return

            # 결과 적용
            self.current_image = result
            self.image_viewer.set_image(result)
            self.history_manager.add_state(result, description)
            self.update_status(tr("status.applied", action=description))

        except Exception as e:
            self.update_status(tr("status.error", message=str(e)))

    def show_adjustment_dialog(self):
        """이미지 조정 다이얼로그 표시"""
        if self.current_image is None:
            self.update_status(tr("status.no_image"))
            return

        # 현재 상태 저장 (취소 시 복원용)
        backup_current = self.current_image.copy()

        dialog = AdjustmentDialog(self.current_image, self)

        # 실시간 미리보기 연결
        dialog.adjustment_preview.connect(self.apply_adjustment_preview)

        # 확인 버튼 클릭 시
        dialog.adjustment_accepted.connect(self.apply_adjustment_final)

        result = dialog.exec_()

        # 취소 버튼 클릭 시 원래 상태로 복원
        if result == dialog.Rejected:
            self.current_image = backup_current
            self.image_viewer.set_image(backup_current)
            self.update_status(tr("edit.adjustment_cancelled"))

    def apply_adjustment_preview(self, adjusted_image):
        """이미지 조정 미리보기 (실시간)"""
        if adjusted_image is not None:
            self.image_viewer.set_image(adjusted_image)

    def apply_adjustment_final(self, adjusted_image, description):
        """이미지 조정 최종 적용 (확인 버튼 클릭 시)"""
        if adjusted_image is None:
            return

        # 결과 적용
        self.current_image = adjusted_image
        self.image_viewer.set_image(adjusted_image)
        self.history_manager.add_state(adjusted_image, description)
        self.update_status(tr("status.applied", action=description))

    def undo(self):
        """실행 취소"""
        if not self.history_manager.can_undo():
            self.update_status(tr("edit.undo_unavailable"))
            return

        result = self.history_manager.undo()
        if result is not None:
            self.current_image = result
            self.image_viewer.set_image(result)
            description = self.history_manager.get_current_description()
            self.update_status(f"{tr('edit.undo')}: {description}")

    def redo(self):
        """다시 실행"""
        if not self.history_manager.can_redo():
            self.update_status(tr("edit.redo_unavailable"))
            return

        result = self.history_manager.redo()
        if result is not None:
            self.current_image = result
            self.image_viewer.set_image(result)
            description = self.history_manager.get_current_description()
            self.update_status(f"{tr('edit.redo')}: {description}")

    def reset_to_original(self):
        """이미지를 원본 상태로 초기화"""
        if self.original_image is None:
            self.update_status(tr("status.no_image"))
            return

        # original_image를 사용하여 초기화
        self.current_image = self.original_image.copy()
        self.image_viewer.set_image(self.current_image)

        # 히스토리 초기화 및 초기 상태 저장
        self.history_manager.clear()
        self.history_manager.add_state(self.current_image, tr("edit.reset"))

        self.update_status(tr("edit.reset_complete"))

    def show_rotate_dialog(self):
        """회전 다이얼로그 표시"""
        if self.current_image is None:
            self.update_status(tr("status.no_image"))
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
            self.update_status(tr("edit.rotation_cancelled"))

    def apply_rotation_preview(self, angle, expand):
        """이미지 회전 미리보기 (실시간)"""
        if self.current_image is None:
            return

        try:
            # 각도가 0이면 현재 이미지 표시
            if angle == 0:
                self.image_viewer.set_image(self.current_image)
                self.update_status(tr("edit.rotation_preview", angle=0))
                return

            # 현재 이미지에서 회전 적용 (미리보기용)
            result = ImageTransform.rotate_custom(self.current_image, angle, expand)
            self.image_viewer.set_image(result)
            self.update_status(tr("edit.rotation_preview", angle=angle))

        except Exception as e:
            self.update_status(tr("status.error", message=str(e)))

    def apply_rotation_final(self, angle, expand):
        """이미지 회전 최종 적용 (확인 버튼 클릭 시)"""
        if self.current_image is None:
            return

        try:
            # 각도가 0이면 변경사항 없음
            if angle == 0:
                self.update_status(tr("edit.rotation_zero"))
                return

            # 회전 적용 (현재 이미지 기준)
            result = ImageTransform.rotate_custom(self.current_image, angle, expand)

            # 결과 적용
            self.current_image = result
            self.image_viewer.set_image(result)
            description = f"{angle}° {tr('edit.rotate')}"
            self.history_manager.add_state(result, description)
            self.update_status(tr("status.applied", action=description))

        except Exception as e:
            self.update_status(tr("status.error", message=str(e)))

    def show_resize_dialog(self):
        """크기 조절 다이얼로그 표시"""
        if self.current_image is None:
            self.update_status(tr("status.no_image"))
            return

        from ui.dialogs.resize_dialog import ResizeDialog

        dialog = ResizeDialog(self, self.current_image)

        # 확인 버튼 클릭 시
        dialog.resize_applied.connect(self.apply_resize_final)

        dialog.exec_()

    def apply_resize_final(self, resized_image, description):
        """크기 조절 최종 적용"""
        if resized_image is None:
            return

        # 결과 적용
        self.current_image = resized_image
        self.image_viewer.set_image(resized_image)
        self.history_manager.add_state(resized_image, description)
        self.update_status(tr("status.applied", action=description))

    # ===== 자르기 모드 =====

    def setup_crop_signals(self):
        """자르기 시그널 연결 (MainWindow.connect_signals에서 호출)"""
        self.image_viewer.crop_confirmed.connect(self._on_crop_confirmed)
        self.image_viewer.crop_cancelled.connect(self._on_crop_cancelled)
        self.image_viewer.crop_size_changed.connect(self._on_crop_size_changed)

    def setup_crop_actions(self):
        """자르기 메뉴 액션 등록"""
        self.ribbon_menu.set_tool_action("crop.confirm", self._confirm_crop)
        self.ribbon_menu.set_tool_action("crop.cancel", self._cancel_crop)
        self.ribbon_menu.set_tool_action("crop.reset", self._reset_crop)

    def start_crop_mode(self):
        """자르기 모드 시작 (편집 메뉴에서 호출)"""
        if self.current_image is None:
            self.update_status(tr("crop.no_image"))
            return

        # 자르기 모드 시작
        if self.image_viewer.start_crop_mode():
            self.update_status(tr("crop.instruction"))

            # 자르기 액션 등록 (처음 한 번만)
            self.setup_crop_actions()

            # 리본 메뉴에 "자르기" 컨텍스트 메뉴 추가 (PowerPoint 스타일)
            self.ribbon_menu.add_context_menu("crop", tr("menu.crop"), auto_select=True)

    def _confirm_crop(self):
        """자르기 확정"""
        self.image_viewer.confirm_crop()

    def _cancel_crop(self):
        """자르기 취소"""
        self.image_viewer.cancel_crop()

    def _reset_crop(self):
        """자르기 영역 초기화"""
        self.image_viewer.reset_crop()

    def _on_crop_confirmed(self, crop_rect: tuple):
        """자르기 확정 시그널 처리"""
        x, y, w, h = crop_rect

        # 이미지 자르기
        import numpy as np

        img_array = np.array(self.current_image)
        cropped = img_array[y : y + h, x : x + w].copy()

        # 결과 적용
        self.current_image = cropped
        self.image_viewer.set_image(cropped)
        self.history_manager.add_state(cropped, tr("edit.crop"))
        self.update_status(tr("crop.applied"))

        # 컨텍스트 메뉴 제거
        self._end_crop_mode()

    def _on_crop_cancelled(self):
        """자르기 취소 시그널 처리"""
        self.update_status(tr("crop.cancelled"))

        # 컨텍스트 메뉴 제거
        self._end_crop_mode()

    def _on_crop_size_changed(self, width: int, height: int):
        """자르기 영역 크기 변경 시그널 처리"""
        self.update_status(tr("crop.size_info", width=width, height=height))

    def _end_crop_mode(self):
        """자르기 모드 종료 - 컨텍스트 메뉴 제거"""
        self.ribbon_menu.remove_context_menu("crop")
        self.toolbar.set_tools([])
