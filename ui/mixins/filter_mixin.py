"""필터 관련 기능 Mixin"""
from ui.dialogs.pixel_effect_dialog import PixelEffectDialog


class FilterMixin:
    """필터 시스템 초기화, 적용, 다이얼로그 표시"""

    def setup_filters(self):
        """필터 시스템 초기화"""
        from filters.artistic import (
            CartoonFilter,
            FilmGrainFilter,
            OilPaintingFilter,
            SketchFilter,
            VintageFilter,
        )
        from filters.basic_filters import (
            CoolFilter,
            GrayscaleFilter,
            InvertFilter,
            SepiaFilter,
            SharpFilter,
            SoftFilter,
            VignetteFilter,
            WarmFilter,
        )
        from filters.photo_filter import PhotoFilter
        from filters.pixel_effects import (
            AverageBlurFilter,
            EmbossFilter,
            GaussianBlurFilter,
            MedianBlurFilter,
            MosaicFilter,
            SharpenFilter,
        )

        # 기본 필터 등록
        self.filter_manager.register_filter(GrayscaleFilter())
        self.filter_manager.register_filter(SepiaFilter())
        self.filter_manager.register_filter(InvertFilter())
        self.filter_manager.register_filter(SoftFilter())
        self.filter_manager.register_filter(SharpFilter())
        self.filter_manager.register_filter(WarmFilter())
        self.filter_manager.register_filter(CoolFilter())
        self.filter_manager.register_filter(VignetteFilter())

        # Photo Filter 등록
        self.filter_manager.register_filter(PhotoFilter())

        # 픽셀 효과 필터 등록
        self.filter_manager.register_filter(MosaicFilter())
        self.filter_manager.register_filter(GaussianBlurFilter())
        self.filter_manager.register_filter(AverageBlurFilter())
        self.filter_manager.register_filter(MedianBlurFilter())
        self.filter_manager.register_filter(SharpenFilter())
        self.filter_manager.register_filter(EmbossFilter())

        # 예술적 효과 필터 등록
        self.filter_manager.register_filter(CartoonFilter())
        self.filter_manager.register_filter(SketchFilter())
        self.filter_manager.register_filter(OilPaintingFilter())
        self.filter_manager.register_filter(FilmGrainFilter())
        self.filter_manager.register_filter(VintageFilter())

        # 필터 메뉴 액션 설정
        self.ribbon_menu.set_tool_action(
            "부드러운", lambda: self.apply_filter("부드러운")
        )
        self.ribbon_menu.set_tool_action("선명한", lambda: self.apply_filter("선명한"))
        self.ribbon_menu.set_tool_action("따뜻한", lambda: self.apply_filter("따뜻한"))
        self.ribbon_menu.set_tool_action("차가운", lambda: self.apply_filter("차가운"))
        self.ribbon_menu.set_tool_action("회색조", lambda: self.apply_filter("회색조"))
        self.ribbon_menu.set_tool_action("세피아", lambda: self.apply_filter("세피아"))
        self.ribbon_menu.set_tool_action("반전", lambda: self.apply_filter("반전"))
        self.ribbon_menu.set_tool_action("비네팅", lambda: self.apply_filter("비네팅"))
        self.ribbon_menu.set_tool_action("Photo Filter", self.show_photo_filter_dialog)

        # 예술적 효과 액션 설정
        self.ribbon_menu.set_tool_action("카툰", self.show_cartoon_dialog)
        self.ribbon_menu.set_tool_action("스케치", self.show_sketch_dialog)
        self.ribbon_menu.set_tool_action("유화", self.show_oil_painting_dialog)
        self.ribbon_menu.set_tool_action("필름 그레인", self.show_film_grain_dialog)
        self.ribbon_menu.set_tool_action("빈티지", self.show_vintage_dialog)

        # 픽셀 효과 액션 설정
        self.setup_pixel_effects()

        # ReShade 액션 설정
        self.ribbon_menu.set_tool_action("ReShade 불러오기", self.load_reshade_preset)
        self.ribbon_menu.set_tool_action("성능 측정", self.toggle_performance_logging)

        # 저장된 ReShade 프리셋 로드 및 툴바에 추가
        self.load_saved_reshade_presets()

    def setup_pixel_effects(self):
        """픽셀 효과 액션 설정"""
        self.ribbon_menu.set_tool_action("모자이크", self.show_mosaic_dialog)
        self.ribbon_menu.set_tool_action(
            "가우시안 블러", self.show_gaussian_blur_dialog
        )
        self.ribbon_menu.set_tool_action("평균 블러", self.show_average_blur_dialog)
        self.ribbon_menu.set_tool_action("중앙값 블러", self.show_median_blur_dialog)
        self.ribbon_menu.set_tool_action("샤프닝", self.show_sharpen_dialog)
        self.ribbon_menu.set_tool_action("엠보싱", self.show_emboss_dialog)

    def apply_filter(self, filter_name, **params):
        """필터 적용 (항상 원본 이미지 기준)"""
        if self.original_image is None:
            self.update_status("필터를 적용할 이미지가 없습니다")
            return

        # 원본 이미지에서 필터 적용
        result = self.filter_manager.apply_filter(
            self.original_image, filter_name, **params
        )
        if result is not None:
            self.current_image = result
            self.image_viewer.set_image(result)
            self.history_manager.add_state(result, f"{filter_name} 필터")

    def apply_pixel_effect_preview(self, filter_name, params):
        """픽셀 효과 미리보기 (실시간)"""
        if self.current_image is None:
            return

        try:
            # 현재 이미지에 필터 적용 (미리보기용)
            result = self.filter_manager.apply_filter(
                self.current_image, filter_name, **params
            )
            if result is not None:
                self.image_viewer.set_image(result)
                self.update_status(f"미리보기: {filter_name}")
        except Exception as e:
            self.update_status(f"미리보기 오류: {str(e)}")

    def apply_pixel_effect_final(self, filter_name, params):
        """픽셀 효과 최종 적용 (확인 버튼)"""
        if self.current_image is None:
            return

        try:
            # 현재 이미지에 필터 적용
            result = self.filter_manager.apply_filter(
                self.current_image, filter_name, **params
            )
            if result is not None:
                self.current_image = result
                self.image_viewer.set_image(result)
                self.history_manager.add_state(result, f"{filter_name}")
                self.update_status(f"{filter_name} 적용됨")
        except Exception as e:
            self.update_status(f"{filter_name} 적용 오류: {str(e)}")

    def on_filter_started(self, filter_name):
        """필터 시작 시 호출"""
        self.update_status(f"{filter_name} 필터 적용 중...")

    def on_filter_completed(self, result, elapsed_time):
        """필터 완료 시 호출"""
        self.update_status(f"필터 적용 완료 ({elapsed_time:.2f}초)")

    def on_filter_failed(self, error_msg):
        """필터 실패 시 호출"""
        self.update_status(f"필터 적용 실패: {error_msg}")
        print(error_msg)

    # 픽셀 효과 다이얼로그 공통 처리
    def _show_pixel_effect_dialog(self, filter_name):
        """픽셀 효과 다이얼로그 공통 처리 헬퍼 메서드
        
        Args:
            filter_name: 필터 이름 (예: "모자이크", "가우시안 블러")
        """
        if self.current_image is None:
            self.update_status("효과를 적용할 이미지가 없습니다")
            return

        # 현재 상태 백업 (취소 시 복원용)
        backup_image = self.current_image.copy()

        filter_obj = self.filter_manager.get_filter(filter_name)
        default_params = filter_obj.get_default_params()

        dialog = PixelEffectDialog(filter_name, default_params, self)

        # 실시간 미리보기
        dialog.parameters_changed.connect(
            lambda params: self.apply_pixel_effect_preview(filter_name, params)
        )

        # 확인 버튼
        dialog.parameters_accepted.connect(
            lambda params: self.apply_pixel_effect_final(filter_name, params)
        )

        result = dialog.exec_()

        # 취소 시 복원
        if result == dialog.Rejected:
            self.current_image = backup_image
            self.image_viewer.set_image(backup_image)
            self.update_status(f"{filter_name} 효과 취소됨")

    # Artistic 필터 다이얼로그 공통 처리
    def _show_artistic_filter_dialog(
        self, dialog_class, filter_name, description_callback=None
    ):
        """Artistic 필터 다이얼로그 공통 처리 헬퍼 메서드
        
        Args:
            dialog_class: 다이얼로그 클래스 (예: CartoonDialog)
            filter_name: 필터 이름 (예: "카툰 효과")
            description_callback: Optional. 다이얼로그 객체를 받아
                description을 반환하는 함수. 없으면 filter_name 사용
        """
        if self.original_image is None:
            self.update_status(f"{filter_name}를 적용할 이미지가 없습니다")
            return

        dialog = dialog_class(self.original_image, self)
        dialog.filter_applied.connect(self.on_artistic_filter_preview)

        result = dialog.exec_()

        if result == dialog_class.Accepted:
            filtered_image = dialog.get_filtered_image()
            if filtered_image is not None:
                self.current_image = filtered_image
                self.image_viewer.set_image(filtered_image)

                # description 생성 (커스텀 콜백 또는 기본 filter_name 사용)
                if description_callback:
                    description = description_callback(dialog)
                else:
                    description = filter_name

                self.history_manager.add_state(filtered_image, description)
                self.update_status(f"{description} 적용됨")
        else:
            self.image_viewer.set_image(self.current_image)
            self.update_status(f"{filter_name} 취소됨")

    # 픽셀 효과 다이얼로그들
    def show_mosaic_dialog(self):
        """모자이크 효과 다이얼로그 표시"""
        self._show_pixel_effect_dialog("모자이크")


    def show_gaussian_blur_dialog(self):
        """가우시안 블러 효과 다이얼로그 표시"""
        self._show_pixel_effect_dialog("가우시안 블러")

    def show_average_blur_dialog(self):
        """평균 블러 효과 다이얼로그 표시"""
        self._show_pixel_effect_dialog("평균 블러")

    def show_median_blur_dialog(self):
        """중앙값 블러 효과 다이얼로그 표시"""
        self._show_pixel_effect_dialog("중앙값 블러")

    def show_sharpen_dialog(self):
        """샤프닝 효과 다이얼로그 표시"""
        self._show_pixel_effect_dialog("샤프닝")

    def show_emboss_dialog(self):
        """엠보싱 효과 다이얼로그 표시"""
        self._show_pixel_effect_dialog("엠보싱")

    def show_photo_filter_dialog(self):
        """Photo Filter 다이얼로그 표시"""
        if self.original_image is None:
            self.update_status("Photo Filter를 적용할 이미지가 없습니다")
            return

        from ui.dialogs.photo_filter_dialog import PhotoFilterDialog

        dialog = PhotoFilterDialog(self.original_image, self)

        # 미리보기: 다이얼로그에서 필터 조정 시 메인 창에 실시간 표시
        dialog.filter_applied.connect(self.on_photo_filter_preview)

        # 초기 미리보기 적용 (시그널 연결 후 호출)
        dialog.apply_filter()

        # 다이얼로그 실행
        result = dialog.exec_()

        if result == PhotoFilterDialog.Accepted:
            # 확인 버튼 클릭 시: 필터 적용된 이미지를 최종 저장
            filtered_image = dialog.get_filtered_image()
            if filtered_image is not None:
                self.current_image = filtered_image
                self.image_viewer.set_image(filtered_image)

                settings = dialog.get_settings()
                description = f"Photo Filter ({settings['filter_name']})"
                self.history_manager.add_state(filtered_image, description)
                self.update_status(f"{description} 적용됨")
        else:
            # 취소 버튼 클릭 시: 원본으로 복원
            self.image_viewer.set_image(self.current_image)
            self.update_status("Photo Filter 취소됨")

    def show_cartoon_dialog(self):
        """카툰 효과 다이얼로그 표시"""
        from ui.dialogs.cartoon_dialog import CartoonDialog
        
        self._show_artistic_filter_dialog(CartoonDialog, "카툰 효과")

    def show_sketch_dialog(self):
        """스케치 효과 다이얼로그 표시"""
        from ui.dialogs.sketch_dialog import SketchDialog
        
        def get_sketch_description(dialog):
            settings = dialog.get_settings()
            sketch_type = "연필" if settings["sketch_type"] == "pencil" else "숯"
            return f"스케치 효과 ({sketch_type})"
        
        self._show_artistic_filter_dialog(
            SketchDialog, "스케치 효과", get_sketch_description
        )

    def show_oil_painting_dialog(self):
        """유화 효과 다이얼로그 표시"""
        from ui.dialogs.oil_painting_dialog import OilPaintingDialog
        
        self._show_artistic_filter_dialog(OilPaintingDialog, "유화 효과")

    def show_film_grain_dialog(self):
        """필름 그레인 다이얼로그 표시"""
        from ui.dialogs.film_grain_dialog import FilmGrainDialog
        
        self._show_artistic_filter_dialog(FilmGrainDialog, "필름 그레인")

    def show_vintage_dialog(self):
        """빈티지 효과 다이얼로그 표시"""
        from ui.dialogs.vintage_dialog import VintageDialog
        
        self._show_artistic_filter_dialog(VintageDialog, "빈티지 효과")

    def on_artistic_filter_preview(self, preview_image):
        """예술적 효과 미리보기"""
        self.image_viewer.set_image(preview_image)

    def on_photo_filter_preview(self, preview_image):
        """Photo Filter 미리보기"""
        # 다이얼로그에서 설정 변경 시 메인 창에 실시간 미리보기 표시
        self.image_viewer.set_image(preview_image)
