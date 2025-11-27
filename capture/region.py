"""
영역 지정 캡처 모듈

투명 오버레이 윈도우를 통해 사용자가 드래그로 영역을 선택하여 캡처합니다.
"""

import mss
from PIL import Image
from PyQt5.QtWidgets import QWidget, QApplication, QDesktopWidget
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush


class RegionSelector(QWidget):
    """영역 선택 투명 오버레이 윈도우"""
    
    region_selected = pyqtSignal(tuple)  # (x, y, width, height)
    selection_cancelled = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.begin = QPoint()
        self.end = QPoint()
        self.is_selecting = False
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        # 윈도우 플래그 먼저 설정
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint | 
            Qt.FramelessWindowHint |
            Qt.Tool  # 태스크바에 표시되지 않도록
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # 모든 스크린의 가상 데스크탑 영역 계산
        app = QApplication.instance()
        desktop = app.desktop()
        
        # 모든 스크린을 포함하는 전체 영역 계산
        total_rect = desktop.screenGeometry(0)
        for i in range(desktop.screenCount()):
            screen_rect = desktop.screenGeometry(i)
            total_rect = total_rect.united(screen_rect)
        
        # 전체 가상 데스크탑 영역으로 설정
        self.setGeometry(total_rect)
        
        # showFullScreen() 대신 show() 사용 (다중 모니터 지원)
        self.show()
        
        # 전체 화면 커서
        self.setCursor(Qt.CrossCursor)
        
        # 포커스 설정
        self.setFocusPolicy(Qt.StrongFocus)
        self.activateWindow()
        self.raise_()
    
    def paintEvent(self, event):
        """그리기 이벤트"""
        painter = QPainter(self)
        
        # 항상 반투명 검은색 배경 표시
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))
        
        # 선택 중일 때만 선택 영역 표시
        if self.is_selecting:
            # 선택 영역 계산
            selection_rect = QRect(self.begin, self.end).normalized()
            
            # 선택 영역 지우기 (투명하게)
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.fillRect(selection_rect, Qt.transparent)
            
            # 선택 영역 테두리 그리기
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            pen = QPen(QColor(0, 120, 215), 2)
            painter.setPen(pen)
            painter.drawRect(selection_rect)
            
            # 크기 정보 표시
            width = selection_rect.width()
            height = selection_rect.height()
            info_text = f"{width} x {height}"
            
            painter.setPen(QPen(Qt.white))
            painter.drawText(selection_rect.topLeft() + QPoint(5, -5), info_text)
        else:
            # 초기 상태: 안내 문구 표시
            painter.setPen(QPen(Qt.white))
            font = painter.font()
            font.setPointSize(14)
            painter.setFont(font)
            
            screen_center = self.rect().center()
            painter.drawText(screen_center.x() - 150, screen_center.y(), 
                           "드래그하여 캡처 영역을 선택하세요 (ESC: 취소)")
    
    def mousePressEvent(self, event):
        """마우스 누름 이벤트"""
        if event.button() == Qt.LeftButton:
            self.begin = event.pos()
            self.end = event.pos()
            self.is_selecting = True
            self.update()
    
    def mouseMoveEvent(self, event):
        """마우스 이동 이벤트"""
        if self.is_selecting:
            self.end = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event):
        """마우스 놓음 이벤트"""
        if event.button() == Qt.LeftButton and self.is_selecting:
            self.end = event.pos()
            
            # 선택 영역 계산
            selection_rect = QRect(self.begin, self.end).normalized()
            
            # 최소 크기 체크 (10x10 이상)
            if selection_rect.width() > 10 and selection_rect.height() > 10:
                # 영역 정보 전달
                region = (selection_rect.x(), selection_rect.y(),
                         selection_rect.width(), selection_rect.height())
                self.region_selected.emit(region)
            else:
                # 너무 작은 영역은 취소
                self.selection_cancelled.emit()
            
            self.close()
    
    def keyPressEvent(self, event):
        """키 입력 이벤트"""
        if event.key() == Qt.Key_Escape:
            # ESC 키로 취소
            self.selection_cancelled.emit()
            self.close()


def capture_region(callback=None):
    """
    영역 지정 캡처
    
    사용자가 드래그로 선택한 영역을 캡처합니다.
    콜백 방식으로 작동하여 기존 이벤트 루프와 충돌하지 않습니다.
    
    Args:
        callback: 캡처 완료 시 호출될 콜백 함수 (PIL.Image 또는 None을 인자로 받음)
    
    Returns:
        RegionSelector: 생성된 선택기 위젯 (참조 유지용)
    """
    def on_region_selected(region):
        """영역 선택 완료"""
        x, y, width, height = region
        
        try:
            with mss.mss() as sct:
                # 선택 영역 캡처
                monitor = {'top': y, 'left': x, 'width': width, 'height': height}
                screenshot = sct.grab(monitor)
                
                # PIL Image로 변환
                img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
                
                if callback:
                    callback(img)
                
        except Exception as e:
            print(f"영역 캡처 실패: {str(e)}")
            if callback:
                callback(None)
    
    def on_cancelled():
        """선택 취소"""
        if callback:
            callback(None)
    
    # 영역 선택 위젯 생성
    selector = RegionSelector()
    selector.region_selected.connect(on_region_selected)
    selector.selection_cancelled.connect(on_cancelled)
    
    # 위젯 표시는 init_ui에서 처리됨
    
    return selector
