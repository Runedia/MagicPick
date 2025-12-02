"""ReShade 관련 기능 Mixin"""


class ReshadeMixin:
    """ReShade 프리셋 로드, 적용, 삭제, 이름 변경, 성능 측정"""

    def load_saved_reshade_presets(self):
        """저장된 ReShade 프리셋을 로드하여 Shader 메뉴에 추가"""
        preset_names = self.reshade_manager.get_all_preset_names()

        for preset_name in preset_names:
            result = self.reshade_manager.get_preset(preset_name)
            if result is not None:
                preset_data, reshade_filter = result
                self.filter_manager.register_filter(reshade_filter)

    def load_reshade_preset(self):
        """ReShade 프리셋 불러오기 다이얼로그 표시"""
        from ui.dialogs.reshade_load_dialog import ReShadeLoadDialog

        # MainWindow의 reshade_manager를 전달
        dialog = ReShadeLoadDialog(self.reshade_manager, self)
        dialog.preset_loaded.connect(self.on_reshade_preset_loaded)

        if dialog.exec_():
            preset_name, reshade_filter, unsupported_effects = dialog.get_result()

            if reshade_filter is not None:
                # FilterManager에 필터 등록
                self.filter_manager.register_filter(reshade_filter)

                # 현재 메뉴 상태 확인
                current_menu = self.ribbon_menu.current_menu

                # 셰이더 메뉴가 현재 열려있는 경우에만 즉시 버튼 추가
                if current_menu == "셰이더":
                    # 이미 추가된 버튼이 아닌 경우에만 추가
                    if preset_name not in self.toolbar.tool_buttons:
                        self.toolbar.add_reshade_filter(
                            preset_name,
                            self.apply_reshade_filter,
                            self.delete_reshade_filter,
                            self.rename_reshade_filter,
                        )

                self.update_status(f"ReShade 프리셋 '{preset_name}' 로드됨")

    def on_reshade_preset_loaded(
        self, preset_name, reshade_filter, unsupported_effects
    ):
        """ReShade 프리셋 로드 완료 시 호출"""
        pass

    def apply_reshade_filter(self, preset_name):
        """ReShade 필터 적용"""
        if self.original_image is None:
            self.update_status("필터를 적용할 이미지가 없습니다")
            return

        try:
            enable_perf_log = getattr(self, "_reshade_performance_logging", False)
            result = self.filter_manager.apply_filter(
                self.original_image,
                preset_name,
                _enable_performance_logging=enable_perf_log,
            )

            if result is not None:
                self.current_image = result
                self.image_viewer.set_image(result)
                self.history_manager.add_state(result, f"ReShade: {preset_name}")
                self.update_status(f"ReShade '{preset_name}' 적용됨")

        except Exception as e:
            self.update_status(f"ReShade 필터 적용 오류: {str(e)}")

    def delete_reshade_filter(self, preset_name):
        """ReShade 필터 삭제"""
        if self.reshade_manager.delete_preset(preset_name):
            self.filter_manager.unregister_filter(preset_name)
            self.toolbar.remove_reshade_filter(preset_name)
            self.update_status(f"'{preset_name}' 필터 삭제됨")
        else:
            self.update_status(f"'{preset_name}' 필터 삭제 실패")

    def rename_reshade_filter(self, old_name, new_name):
        """ReShade 필터 이름 변경"""
        if self.reshade_manager.rename_preset(old_name, new_name):
            result = self.reshade_manager.get_preset(new_name)
            if result is not None:
                preset_data, reshade_filter = result

                self.filter_manager.unregister_filter(old_name)
                self.filter_manager.register_filter(reshade_filter)

                self.toolbar.rename_reshade_filter(old_name, new_name)

                self.update_status(f"'{old_name}' -> '{new_name}' 이름 변경됨")
        else:
            self.update_status(f"'{old_name}' 이름 변경 실패")

    def toggle_performance_logging(self):
        """ReShade 성능 측정 토글"""
        self._reshade_performance_logging = not self._reshade_performance_logging

        status = "활성화" if self._reshade_performance_logging else "비활성화"
        self.update_status(f"ReShade 성능 측정 {status}")

        if self._reshade_performance_logging:
            print("\n" + "=" * 80)
            print(
                "[성능 측정 활성화] ReShade 필터 적용 시 콘솔에 성능 정보가 출력됩니다."
            )
            print("=" * 80 + "\n")
        else:
            print("\n" + "=" * 80)
            print("[성능 측정 비활성화]")
            print("=" * 80 + "\n")
