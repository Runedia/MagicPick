from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QStatusBar
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QIcon
from ui.menu_bar import RibbonMenuBar
from ui.toolbar import ToolBar
from ui.widgets.image_viewer import ImageViewer
from utils.file_manager import FileManager
from utils.history import HistoryManager
from editor.transform import ImageTransform
from editor.adjustments import ImageAdjustments
from filters.base_filter import FilterManager


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings('ImageEditor', 'Settings')
        self.file_manager = FileManager(self)
        self.history_manager = HistoryManager(max_history=20)
        self.filter_manager = FilterManager()
        self.original_image = None  # 원본 이미지 저장
        self.current_image = None   # 현재 표시 중인 이미지
        self.init_ui()
        self.restore_window_state()
        self.connect_signals()

    def init_ui(self):
        self.setWindowTitle('Image Filter & Screenshot Editor')

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
        self.status_bar.showMessage('준비')
        
        self.setup_file_actions()
        self.setup_edit_actions()
        self.setup_filters()
    
    def resizeEvent(self, event):
        """윈도우 크기 변경 시 툴바 너비 조정"""
        super().resizeEvent(event)
        if hasattr(self, 'toolbar'):
            self.toolbar.setFixedWidth(self.centralWidget().width())

    def connect_signals(self):
        self.ribbon_menu.menu_changed.connect(self.on_menu_changed)
        self.file_manager.file_loaded.connect(self.on_file_loaded)
        self.file_manager.file_saved.connect(self.on_file_saved)
        self.filter_manager.filter_started.connect(self.on_filter_started)
        self.filter_manager.filter_completed.connect(self.on_filter_completed)
        self.filter_manager.filter_failed.connect(self.on_filter_failed)

    def setup_file_actions(self):
        self.ribbon_menu.set_tool_action('열기', self.open_file)
        self.ribbon_menu.set_tool_action('저장', self.save_file)
        self.ribbon_menu.set_tool_action('다른 이름으로 저장', self.save_file_as)
        self.ribbon_menu.set_tool_action('끝내기', self.close)

    def setup_edit_actions(self):
        """편집 메뉴 액션 설정"""
        # 변형 기능
        self.ribbon_menu.set_tool_action('회전 90도', lambda: self.apply_transform('rotate_90'))
        self.ribbon_menu.set_tool_action('회전 180도', lambda: self.apply_transform('rotate_180'))
        self.ribbon_menu.set_tool_action('회전 270도', lambda: self.apply_transform('rotate_270'))
        self.ribbon_menu.set_tool_action('좌우 반전', lambda: self.apply_transform('flip_horizontal'))
        self.ribbon_menu.set_tool_action('상하 반전', lambda: self.apply_transform('flip_vertical'))
        
        # 조정 기능 (대화상자를 통해 값 입력받는 방식으로 나중에 개선 예정)
        self.ribbon_menu.set_tool_action('밝기', lambda: self.adjust_brightness(30))
        self.ribbon_menu.set_tool_action('대비', lambda: self.adjust_contrast(30))
        self.ribbon_menu.set_tool_action('채도', lambda: self.adjust_saturation(150))
        self.ribbon_menu.set_tool_action('감마', lambda: self.adjust_gamma(1.2))
        
        # Undo/Redo
        self.ribbon_menu.set_tool_action('실행 취소', self.undo)
        self.ribbon_menu.set_tool_action('다시 실행', self.redo)

    def setup_filters(self):
        """필터 시스템 초기화"""
        from filters.basic_filters import (
            GrayscaleFilter, SepiaFilter, InvertFilter,
            SoftFilter, SharpFilter, WarmFilter, CoolFilter, VignetteFilter
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
        
        # 필터 메뉴 액션 설정
        self.ribbon_menu.set_tool_action('부드러운', lambda: self.apply_filter('부드러운'))
        self.ribbon_menu.set_tool_action('선명한', lambda: self.apply_filter('선명한'))
        self.ribbon_menu.set_tool_action('따뜻한', lambda: self.apply_filter('따뜻한'))
        self.ribbon_menu.set_tool_action('차가운', lambda: self.apply_filter('차가운'))
        self.ribbon_menu.set_tool_action('회색조', lambda: self.apply_filter('회색조'))
        self.ribbon_menu.set_tool_action('세피아', lambda: self.apply_filter('세피아'))
        self.ribbon_menu.set_tool_action('반전', lambda: self.apply_filter('반전'))
        self.ribbon_menu.set_tool_action('비네팅', lambda: self.apply_filter('비네팅'))
    
    def apply_filter(self, filter_name, **params):
        """필터 적용 (항상 원본 이미지 기준)"""
        if self.original_image is None:
            self.update_status("필터를 적용할 이미지가 없습니다")
            return
        
        # 원본 이미지에서 필터 적용
        result = self.filter_manager.apply_filter(self.original_image, filter_name, **params)
        if result is not None:
            self.current_image = result
            self.image_viewer.set_image(result)
            self.history_manager.add_state(result, f"{filter_name} 필터")
    
    def on_filter_started(self, filter_name):
        """필터 시작 시 호출"""
        self.update_status(f"{filter_name} 필터 적용 중...")
    
    def on_filter_completed(self, result, elapsed_time):
        """필터 완료 시 호출"""
        self.update_status(f"필터 적용 완료 ({elapsed_time:.2f}초)")
    
    def on_filter_failed(self, error_msg):
        """필터 실패 시 호출"""
        self.update_status(f"필터 적용 실패: {error_msg}")
    
    def on_tool_clicked(self, tool_name):
        self.ribbon_menu.execute_tool_action(tool_name)
        self.update_status(f'{tool_name} 실행됨')

    def on_menu_changed(self, menu_name):
        tools = self.ribbon_menu.get_menu_tools(menu_name)
        self.toolbar.set_tools(tools, self.on_tool_clicked)
        self.update_status(f'{menu_name} 메뉴 선택됨')

    def restore_window_state(self):
        screen_geometry = self.screen().availableGeometry()

        if self.settings.contains('geometry'):
            self.restoreGeometry(self.settings.value('geometry'))
        else:
            width = int(screen_geometry.width() * 0.6)
            height = int(width * 3 / 4)
            x = (screen_geometry.width() - width) // 2
            y = (screen_geometry.height() - height) // 2
            self.setGeometry(x, y, width, height)

    def closeEvent(self, event):
        self.settings.setValue('geometry', self.saveGeometry())
        event.accept()

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
            if transform_type == 'rotate_90':
                result = ImageTransform.rotate_90(self.current_image)
                description = "90도 회전"
            elif transform_type == 'rotate_180':
                result = ImageTransform.rotate_180(self.current_image)
                description = "180도 회전"
            elif transform_type == 'rotate_270':
                result = ImageTransform.rotate_270(self.current_image)
                description = "270도 회전"
            elif transform_type == 'flip_horizontal':
                result = ImageTransform.flip_horizontal(self.current_image)
                description = "좌우 반전"
            elif transform_type == 'flip_vertical':
                result = ImageTransform.flip_vertical(self.current_image)
                description = "상하 반전"
            else:
                self.update_status(f"알 수 없는 변형: {transform_type}")
                return
            
            # 결과 적용 (편집은 원본도 함께 업데이트)
            self.original_image = result.copy()
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
            self.original_image = result.copy()  # 원본도 업데이트
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
            self.original_image = result.copy()  # 원본도 업데이트
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
            self.original_image = result.copy()  # 원본도 업데이트
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
            self.original_image = result.copy()  # 원본도 업데이트
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
            self.original_image = result.copy()  # 원본도 업데이트
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
            self.original_image = result.copy()  # 원본도 업데이트
            self.current_image = result
            self.image_viewer.set_image(result)
            description = self.history_manager.get_current_description()
            self.update_status(f"다시 실행: {description}")
