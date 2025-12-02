from PyQt5.QtCore import QSettings, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QStatusBar, QVBoxLayout, QWidget

from capture.screen_capture import ScreenCapture
from filters.base_filter import FilterManager
from ui.menu_bar import RibbonMenuBar
from ui.mixins import (
    CaptureMixin,
    FileMixin,
    FilterMixin,
    ReshadeMixin,
    TransformMixin,
    UIStateMixin,
)
from ui.toolbar import ToolBar
from ui.widgets.image_viewer import ImageViewer
from ui.widgets.zoom_control import ZoomControl
from utils.file_manager import FileManager
from utils.history import HistoryManager


class MainWindow(
    QMainWindow,
    FileMixin,
    FilterMixin,
    ReshadeMixin,
    CaptureMixin,
    TransformMixin,
    UIStateMixin,
):
    """
    메인 애플리케이션 창

    Mixin 구성:
    - FileMixin: 파일 열기, 저장, 저장 콜백
    - FilterMixin: 필터 시스템, 픽셀 효과, 예술적 효과, Photo Filter
    - ReshadeMixin: ReShade 프리셋 관리 (로드, 적용, 삭제, 이름 변경)
    - CaptureMixin: 화면 캡처 (전체, 영역, 윈도우, 모니터)
    - TransformMixin: 이미지 변환 (회전, 반전) 및 조정 (밝기, 대비, 채도, 감마)
    - UIStateMixin: UI 상태 관리 (창 상태, 줌, 메뉴, 툴바)
    """

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
        """UI 구성"""
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
        self.toolbar.setFixedWidth(central_widget.width())  # 초기 너비 설정
        self.toolbar.move(0, self.ribbon_menu.height())
        self.toolbar.raise_()

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("준비 | 단축키: Ctrl+Shift+F1~F4 (전체/영역/윈도우/모니터)")

        # 배율 컨트롤 추가 (상태바 우측)
        self.zoom_control = ZoomControl()
        self.status_bar.addPermanentWidget(self.zoom_control)

        # Mixin 메서드 호출로 액션 설정
        self.setup_file_actions()  # FileMixin
        self.setup_edit_actions()  # TransformMixin
        self.setup_capture_actions()  # CaptureMixin
        self.setup_filters()  # FilterMixin

    def connect_signals(self):
        """시그널 연결"""
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
