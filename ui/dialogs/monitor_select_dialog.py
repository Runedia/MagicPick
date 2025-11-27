"""
모니터 선택 다이얼로그 모듈

다중 모니터 환경에서 캡처할 모니터를 선택할 수 있는 다이얼로그를 제공합니다.
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QListWidget, QListWidgetItem)
from PyQt5.QtCore import Qt, pyqtSignal
from capture.monitor import get_monitor_list


class MonitorSelectDialog(QDialog):
    """모니터 선택 다이얼로그"""
    
    monitor_selected = pyqtSignal(int)  # 선택된 모니터 인덱스
    
    def __init__(self, parent=None):
        """
        Args:
            parent: 부모 위젯
        """
        super().__init__(parent)
        self.selected_monitor = None
        self.init_ui()
        self.load_monitors()
    
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle('모니터 선택')
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)
        
        layout = QVBoxLayout()
        
        # 안내 메시지
        info_label = QLabel('캡처할 모니터를 선택하세요:')
        info_label.setStyleSheet('font-size: 11pt; margin-bottom: 10px;')
        layout.addWidget(info_label)
        
        # 모니터 목록
        self.monitor_list = QListWidget()
        self.monitor_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #c0c0c0;
                border-radius: 4px;
                padding: 5px;
                font-size: 10pt;
            }
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #e0e0e0;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #e5f3ff;
            }
        """)
        self.monitor_list.itemDoubleClicked.connect(self.on_accept)
        layout.addWidget(self.monitor_list)
        
        # 버튼
        button_layout = QHBoxLayout()
        
        ok_button = QPushButton('확인')
        ok_button.clicked.connect(self.on_accept)
        ok_button.setDefault(True)
        ok_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 20px;
                font-size: 10pt;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #006cc1;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
        """)
        
        cancel_button = QPushButton('취소')
        cancel_button.clicked.connect(self.reject)
        cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                color: #333;
                border: 1px solid #c0c0c0;
                border-radius: 4px;
                padding: 8px 20px;
                font-size: 10pt;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """)
        
        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def load_monitors(self):
        """모니터 목록 로드"""
        monitors = get_monitor_list()
        
        if not monitors:
            # 모니터를 찾을 수 없는 경우
            item = QListWidgetItem("모니터를 찾을 수 없습니다")
            item.setFlags(Qt.NoItemFlags)
            self.monitor_list.addItem(item)
            return
        
        for monitor in monitors:
            index = monitor['index']
            width = monitor['width']
            height = monitor['height']
            left = monitor['left']
            top = monitor['top']
            
            # 주 모니터 표시
            is_primary = (index == 1)
            primary_text = " (주 모니터)" if is_primary else ""
            
            # 목록 아이템 텍스트
            text = f"모니터 {index}{primary_text}\n{width} x {height}  위치: ({left}, {top})"
            
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, index)  # 모니터 인덱스 저장
            self.monitor_list.addItem(item)
        
        # 첫 번째 아이템 선택
        if self.monitor_list.count() > 0:
            self.monitor_list.setCurrentRow(0)
    
    def on_accept(self):
        """확인 버튼 클릭 시"""
        current_item = self.monitor_list.currentItem()
        
        if current_item is None:
            # 선택된 항목이 없으면 취소
            self.reject()
            return
        
        # 선택된 모니터 인덱스 가져오기
        monitor_index = current_item.data(Qt.UserRole)
        
        if monitor_index is not None:
            self.selected_monitor = monitor_index
            self.monitor_selected.emit(monitor_index)
            self.accept()
        else:
            self.reject()
    
    def get_selected_monitor(self):
        """
        선택된 모니터 인덱스 반환
        
        Returns:
            int: 모니터 인덱스 (선택되지 않았으면 None)
        """
        return self.selected_monitor
