from PyQt5.QtCore import QSettings, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QStatusBar, QVBoxLayout, QWidget

from capture import fullscreen, monitor, region, window
from capture.screen_capture import ScreenCapture
from editor.adjustments import ImageAdjustments
from editor.transform import ImageTransform
from filters.base_filter import FilterManager
from ui.dialogs.monitor_select_dialog import MonitorSelectDialog
from ui.dialogs.pixel_effect_dialog import PixelEffectDialog
from ui.dialogs.rotate_dialog import RotateDialog
from ui.menu_bar import RibbonMenuBar
from ui.toolbar import ToolBar
from ui.widgets.image_viewer import ImageViewer
from ui.widgets.zoom_control import ZoomControl
from utils.file_manager import FileManager
from utils.history import HistoryManager


class MainWindow(QMainWindow):
    closing = pyqtSignal()  # 창이 숨겨지기 전에 발생하는 시그널

    def __init__(self):
        super().__init__()
        self.settings = QSettings("MagicPick", "Settings")
        self.file_manager = FileManager(self)
        self.history_manager = HistoryManager(max_history=20)
        self.filter_manager = FilterManager()
        self.screen_capture = ScreenCapture(self)
        self.original_image = None
        self.current_image = None

        from config.reshade_config import ReShadePresetManager

        self.reshade_manager = ReShadePresetManager()
        self._reshade_performance_logging = False

        self.init_ui()
        self.restore_window_state()
        self.connect_signals()

    def init_ui(self):
        self.setWindowTitle("MagicPick")
        self.setWindowIcon(QIcon("assets/logo.png"))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.ribbon_menu = RibbonMenuBar(self)
        layout.addWidget(self.ribbon_menu)

        # 이미지 뷰어는 레이아웃에 추가
        self.image_viewer = ImageViewer(self)
        layout.addWidget(self.image_viewer, stretch=1)

        central_widget.setLayout(layout)

        # 툴바는 central_widget의 자식으로 설정하되 레이아웃에는 추가하지 않음 (오버레이)
        self.toolbar = ToolBar(central_widget)
        self.toolbar.move(0, self.ribbon_menu.height())
        self.toolbar.raise_()

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("준비 | 단축키: Ctrl+Shift+F1~F4 (전체/영역/윈도우/모니터)")

        # 배율 컨트롤 추가 (상태바 우측)
        self.zoom_control = ZoomControl()
        self.status_bar.addPermanentWidget(self.zoom_control)

        self.setup_file_actions()
        self.setup_edit_actions()
        self.setup_capture_actions()
        self.setup_filters()

    def resizeEvent(self, event):
        """윈도우 크기 변경 시 툴바 너비 조정"""
        super().resizeEvent(event)
        if hasattr(self, "toolbar"):
            self.toolbar.setFixedWidth(self.centralWidget().width())

    def connect_signals(self):
        self.ribbon_menu.menu_changed.connect(self.on_menu_changed)
        self.file_manager.file_loaded.connect(self.on_file_loaded)
        self.file_manager.file_saved.connect(self.on_file_saved)
        self.filter_manager.filter_started.connect(self.on_filter_started)
        self.filter_manager.filter_completed.connect(self.on_filter_completed)
        self.filter_manager.filter_failed.connect(self.on_filter_failed)
        self.screen_capture.capture_completed.connect(self.on_capture_completed)
        self.screen_capture.capture_failed.connect(self.on_capture_failed)

        # 배율 컨트롤 연결
        self.zoom_control.zoom_changed.connect(self.on_zoom_changed)
        self.image_viewer.zoom_changed.connect(self.on_viewer_zoom_changed)

    def setup_file_actions(self):
        self.ribbon_menu.set_tool_action("열기", self.open_file)
        self.ribbon_menu.set_tool_action("저장", self.save_file)
        self.ribbon_menu.set_tool_action("다른 이름으로 저장", self.save_file_as)
        self.ribbon_menu.set_tool_action("끝내기", self.close)

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

    def on_tool_clicked(self, tool_name):
        self.ribbon_menu.execute_tool_action(tool_name)
        self.update_status(f"{tool_name} 실행됨")

    def on_menu_changed(self, menu_name):
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

    def restore_window_state(self):
        screen_geometry = self.screen().availableGeometry()

        if self.settings.contains("geometry"):
            self.restoreGeometry(self.settings.value("geometry"))
        else:
            width = int(screen_geometry.width() * 0.6)
            height = int(width * 3 / 4)
            x = (screen_geometry.width() - width) // 2
            y = (screen_geometry.height() - height) // 2
            self.setGeometry(x, y, width, height)

    def on_zoom_changed(self, factor):
        """배율 컨트롤 슬라이더 변경 시"""
        self.image_viewer.set_zoom(factor)

    def on_viewer_zoom_changed(self, factor):
        """이미지 뷰어 배율 변경 시 (CTRL+휠)"""
        self.zoom_control.set_zoom(factor)

    def closeEvent(self, event):
        """창 종료 대신 숨기기 (트레이 서비스 계속 실행)"""
        # 창 geometry 저장 (기존 동작)
        self.settings.setValue("geometry", self.saveGeometry())

        # TrayService에 타이머 설정 저장 신호
        self.closing.emit()

        # 실제 종료 대신 숨기기
        event.ignore()
        self.hide()

    def update_status(self, message):
        self.status_bar.showMessage(message)

    def open_file(self):
        image_data, file_path = self.file_manager.open_file(self)
        if image_data is not None:
            self.original_image = image_data.copy()  # 원본 이미지 저장
            self.current_image = image_data
            self.image_viewer.set_image(image_data)
            file_name = self.file_manager.get_current_file_name()
            self.update_status(f"Opened: {file_name}")

    def save_file(self):
        if self.current_image is None:
            self.update_status("No image to save")
            return

        if self.file_manager.save_file(self, self.current_image):
            file_name = self.file_manager.get_current_file_name()
            self.update_status(f"Saved: {file_name}")

    def save_file_as(self):
        if self.current_image is None:
            self.update_status("No image to save")
            return

        if self.file_manager.save_file_as(self, self.current_image):
            file_name = self.file_manager.get_current_file_name()
            self.update_status(f"Saved as: {file_name}")

    def on_file_loaded(self, image_data, file_path):
        self.original_image = image_data.copy()  # 원본 이미지 저장
        self.current_image = image_data
        self.image_viewer.set_image(image_data)
        # 히스토리 초기화 및 초기 상태 저장
        self.history_manager.clear()
        self.history_manager.add_state(image_data, "파일 열기")

    def on_file_saved(self, file_path):
        pass

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

    def setup_capture_actions(self):
        """캡처 액션 등록"""
        self.ribbon_menu.set_tool_action("전체화면", self.capture_fullscreen)
        self.ribbon_menu.set_tool_action("영역 지정", self.capture_region)
        self.ribbon_menu.set_tool_action("윈도우", self.capture_window)
        self.ribbon_menu.set_tool_action("모니터", self.capture_monitor)

    def show_mosaic_dialog(self):
        """모자이크 효과 다이얼로그 표시"""
        if self.current_image is None:
            self.update_status("효과를 적용할 이미지가 없습니다")
            return

        # 현재 상태 백업 (취소 시 복원용)
        backup_image = self.current_image.copy()

        filter_obj = self.filter_manager.get_filter("모자이크")
        default_params = filter_obj.get_default_params()

        dialog = PixelEffectDialog("모자이크", default_params, self)

        # 실시간 미리보기
        dialog.parameters_changed.connect(
            lambda params: self.apply_pixel_effect_preview("모자이크", params)
        )

        # 확인 버튼
        dialog.parameters_accepted.connect(
            lambda params: self.apply_pixel_effect_final("모자이크", params)
        )

        result = dialog.exec_()

        # 취소 시 복원
        if result == dialog.Rejected:
            self.current_image = backup_image
            self.image_viewer.set_image(backup_image)
            self.update_status("모자이크 효과 취소됨")

    def show_gaussian_blur_dialog(self):
        """가우시안 블러 효과 다이얼로그 표시"""
        if self.current_image is None:
            self.update_status("효과를 적용할 이미지가 없습니다")
            return

        # 현재 상태 백업
        backup_image = self.current_image.copy()

        filter_obj = self.filter_manager.get_filter("가우시안 블러")
        default_params = filter_obj.get_default_params()

        dialog = PixelEffectDialog("가우시안 블러", default_params, self)

        # 실시간 미리보기
        dialog.parameters_changed.connect(
            lambda params: self.apply_pixel_effect_preview("가우시안 블러", params)
        )

        # 확인 버튼
        dialog.parameters_accepted.connect(
            lambda params: self.apply_pixel_effect_final("가우시안 블러", params)
        )

        result = dialog.exec_()

        # 취소 시 복원
        if result == dialog.Rejected:
            self.current_image = backup_image
            self.image_viewer.set_image(backup_image)
            self.update_status("가우시안 블러 효과 취소됨")

    def show_average_blur_dialog(self):
        """평균 블러 효과 다이얼로그 표시"""
        if self.current_image is None:
            self.update_status("효과를 적용할 이미지가 없습니다")
            return

        # 현재 상태 백업
        backup_image = self.current_image.copy()

        filter_obj = self.filter_manager.get_filter("평균 블러")
        default_params = filter_obj.get_default_params()

        dialog = PixelEffectDialog("평균 블러", default_params, self)

        # 실시간 미리보기
        dialog.parameters_changed.connect(
            lambda params: self.apply_pixel_effect_preview("평균 블러", params)
        )

        # 확인 버튼
        dialog.parameters_accepted.connect(
            lambda params: self.apply_pixel_effect_final("평균 블러", params)
        )

        result = dialog.exec_()

        # 취소 시 복원
        if result == dialog.Rejected:
            self.current_image = backup_image
            self.image_viewer.set_image(backup_image)
            self.update_status("평균 블러 효과 취소됨")

    def show_median_blur_dialog(self):
        """중앙값 블러 효과 다이얼로그 표시"""
        if self.current_image is None:
            self.update_status("효과를 적용할 이미지가 없습니다")
            return

        # 현재 상태 백업
        backup_image = self.current_image.copy()

        filter_obj = self.filter_manager.get_filter("중앙값 블러")
        default_params = filter_obj.get_default_params()

        dialog = PixelEffectDialog("중앙값 블러", default_params, self)

        # 실시간 미리보기
        dialog.parameters_changed.connect(
            lambda params: self.apply_pixel_effect_preview("중앙값 블러", params)
        )

        # 확인 버튼
        dialog.parameters_accepted.connect(
            lambda params: self.apply_pixel_effect_final("중앙값 블러", params)
        )

        result = dialog.exec_()

        # 취소 시 복원
        if result == dialog.Rejected:
            self.current_image = backup_image
            self.image_viewer.set_image(backup_image)
            self.update_status("중앙값 블러 효과 취소됨")

    def show_sharpen_dialog(self):
        """샤프닝 효과 다이얼로그 표시"""
        if self.current_image is None:
            self.update_status("효과를 적용할 이미지가 없습니다")
            return

        # 현재 상태 백업
        backup_image = self.current_image.copy()

        filter_obj = self.filter_manager.get_filter("샤프닝")
        default_params = filter_obj.get_default_params()

        dialog = PixelEffectDialog("샤프닝", default_params, self)

        # 실시간 미리보기
        dialog.parameters_changed.connect(
            lambda params: self.apply_pixel_effect_preview("샤프닝", params)
        )

        # 확인 버튼
        dialog.parameters_accepted.connect(
            lambda params: self.apply_pixel_effect_final("샤프닝", params)
        )

        result = dialog.exec_()

        # 취소 시 복원
        if result == dialog.Rejected:
            self.current_image = backup_image
            self.image_viewer.set_image(backup_image)
            self.update_status("샤프닝 효과 취소됨")

    def show_emboss_dialog(self):
        """엠보싱 효과 다이얼로그 표시"""
        if self.current_image is None:
            self.update_status("효과를 적용할 이미지가 없습니다")
            return

        # 현재 상태 백업
        backup_image = self.current_image.copy()

        filter_obj = self.filter_manager.get_filter("엠보싱")
        default_params = filter_obj.get_default_params()

        dialog = PixelEffectDialog("엠보싱", default_params, self)

        # 실시간 미리보기
        dialog.parameters_changed.connect(
            lambda params: self.apply_pixel_effect_preview("엠보싱", params)
        )

        # 확인 버튼
        dialog.parameters_accepted.connect(
            lambda params: self.apply_pixel_effect_final("엠보싱", params)
        )

        result = dialog.exec_()

        # 취소 시 복원
        if result == dialog.Rejected:
            self.current_image = backup_image
            self.image_viewer.set_image(backup_image)
            self.update_status("엠보싱 효과 취소됨")

    def show_photo_filter_dialog(self):
        """Photo Filter 다이얼로그 표시"""
        if self.original_image is None:
            self.update_status("Photo Filter를 적용할 이미지가 없습니다")
            return

        from ui.dialogs.photo_filter_dialog import PhotoFilterDialog

        dialog = PhotoFilterDialog(self.original_image, self)

        # 미리보기: 다이얼로그에서 필터 조정 시 메인 창에 실시간 표시
        dialog.filter_applied.connect(self.on_photo_filter_preview)

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
        if self.original_image is None:
            self.update_status("카툰 효과를 적용할 이미지가 없습니다")
            return

        from ui.dialogs.cartoon_dialog import CartoonDialog

        dialog = CartoonDialog(self.original_image, self)
        dialog.filter_applied.connect(self.on_artistic_filter_preview)

        result = dialog.exec_()

        if result == CartoonDialog.Accepted:
            filtered_image = dialog.get_filtered_image()
            if filtered_image is not None:
                self.current_image = filtered_image
                self.image_viewer.set_image(filtered_image)

                self.history_manager.add_state(filtered_image, "카툰 효과")
                self.update_status("카툰 효과 적용됨")
        else:
            self.image_viewer.set_image(self.current_image)
            self.update_status("카툰 효과 취소됨")

    def show_sketch_dialog(self):
        """스케치 효과 다이얼로그 표시"""
        if self.original_image is None:
            self.update_status("스케치 효과를 적용할 이미지가 없습니다")
            return

        from ui.dialogs.sketch_dialog import SketchDialog

        dialog = SketchDialog(self.original_image, self)
        dialog.filter_applied.connect(self.on_artistic_filter_preview)

        result = dialog.exec_()

        if result == SketchDialog.Accepted:
            filtered_image = dialog.get_filtered_image()
            if filtered_image is not None:
                self.current_image = filtered_image
                self.image_viewer.set_image(filtered_image)

                settings = dialog.get_settings()
                sketch_type = "연필" if settings["sketch_type"] == "pencil" else "숯"
                description = f"스케치 효과 ({sketch_type})"
                self.history_manager.add_state(filtered_image, description)
                self.update_status(f"{description} 적용됨")
        else:
            self.image_viewer.set_image(self.current_image)
            self.update_status("스케치 효과 취소됨")

    def show_oil_painting_dialog(self):
        """유화 효과 다이얼로그 표시"""
        if self.original_image is None:
            self.update_status("유화 효과를 적용할 이미지가 없습니다")
            return

        from ui.dialogs.oil_painting_dialog import OilPaintingDialog

        dialog = OilPaintingDialog(self.original_image, self)
        dialog.filter_applied.connect(self.on_artistic_filter_preview)

        result = dialog.exec_()

        if result == OilPaintingDialog.Accepted:
            filtered_image = dialog.get_filtered_image()
            if filtered_image is not None:
                self.current_image = filtered_image
                self.image_viewer.set_image(filtered_image)

                self.history_manager.add_state(filtered_image, "유화 효과")
                self.update_status("유화 효과 적용됨")
        else:
            self.image_viewer.set_image(self.current_image)
            self.update_status("유화 효과 취소됨")

    def show_film_grain_dialog(self):
        """필름 그레인 다이얼로그 표시"""
        if self.original_image is None:
            self.update_status("필름 그레인을 적용할 이미지가 없습니다")
            return

        from ui.dialogs.film_grain_dialog import FilmGrainDialog

        dialog = FilmGrainDialog(self.original_image, self)
        dialog.filter_applied.connect(self.on_artistic_filter_preview)

        result = dialog.exec_()

        if result == FilmGrainDialog.Accepted:
            filtered_image = dialog.get_filtered_image()
            if filtered_image is not None:
                self.current_image = filtered_image
                self.image_viewer.set_image(filtered_image)

                self.history_manager.add_state(filtered_image, "필름 그레인")
                self.update_status("필름 그레인 적용됨")
        else:
            self.image_viewer.set_image(self.current_image)
            self.update_status("필름 그레인 취소됨")

    def show_vintage_dialog(self):
        """빈티지 효과 다이얼로그 표시"""
        if self.original_image is None:
            self.update_status("빈티지 효과를 적용할 이미지가 없습니다")
            return

        from ui.dialogs.vintage_dialog import VintageDialog

        dialog = VintageDialog(self.original_image, self)
        dialog.filter_applied.connect(self.on_artistic_filter_preview)

        result = dialog.exec_()

        if result == VintageDialog.Accepted:
            filtered_image = dialog.get_filtered_image()
            if filtered_image is not None:
                self.current_image = filtered_image
                self.image_viewer.set_image(filtered_image)

                self.history_manager.add_state(filtered_image, "빈티지 효과")
                self.update_status("빈티지 효과 적용됨")
        else:
            self.image_viewer.set_image(self.current_image)
            self.update_status("빈티지 효과 취소됨")

    def on_artistic_filter_preview(self, preview_image):
        """예술적 효과 미리보기"""
        self.image_viewer.set_image(preview_image)

    def on_photo_filter_preview(self, preview_image):
        """Photo Filter 미리보기"""
        # 다이얼로그에서 설정 변경 시 메인 창에 실시간 미리보기 표시
        self.image_viewer.set_image(preview_image)

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

    # 캡처 관련 메서드
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
        # 모니터 선택 다이얼로그 표시
        dialog = MonitorSelectDialog(self)
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

    def on_capture_failed(self, error_message):
        """캡처 실패 시"""
        self.update_status(f"캡처 실패: {error_message}")
        QMessageBox.warning(self, "캡처 실패", error_message)
