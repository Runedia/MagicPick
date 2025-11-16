from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QScrollArea
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFontMetrics


class ToolBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.current_tools = []
        self.tool_buttons = {}
        self.auto_hide = True
        self.hide_timer = QTimer(self)
        self.hide_timer.timeout.connect(self.auto_hide_toolbar)
        self.hide_timer.setSingleShot(True)
        self.init_ui()

    def init_ui(self):
        self.setMaximumHeight(60)
        self.setMinimumHeight(60)
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(245, 245, 245, 240);
                border-bottom: 1px solid #d0d0d0;
            }
        """)
        # 오버레이를 위한 속성 설정
        self.setAttribute(Qt.WA_StyledBackground, True)

        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(5)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)

        self.tool_container = QWidget()
        self.tool_container.setStyleSheet("background-color: transparent;")
        self.tool_layout = QHBoxLayout()
        self.tool_layout.setContentsMargins(5, 0, 5, 0)
        self.tool_layout.setSpacing(5)
        self.tool_layout.setAlignment(Qt.AlignLeft)
        self.tool_container.setLayout(self.tool_layout)

        self.scroll_area.setWidget(self.tool_container)
        self.layout.addWidget(self.scroll_area)

        self.setLayout(self.layout)
        self.setVisible(False)

    def set_tools(self, tool_names, action_callback=None):
        for i in reversed(range(self.tool_layout.count())):
            widget = self.tool_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
            else:
                item = self.tool_layout.itemAt(i)
                if item:
                    self.tool_layout.removeItem(item)

        self.current_tools = []
        self.tool_buttons = {}

        for name in tool_names:
            btn = QPushButton(name)
            
            # 텍스트 길이에 따른 버튼 너비 계산
            font_metrics = QFontMetrics(btn.font())
            text_width = font_metrics.horizontalAdvance(name)
            # 최소 100px, 텍스트 + 좌우 패딩(40px)
            button_width = max(100, text_width + 40)
            
            btn.setMinimumWidth(button_width)
            btn.setFixedHeight(40)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #ffffff;
                    border: 1px solid #c0c0c0;
                    border-radius: 4px;
                    padding: 8px 16px;
                    font-size: 10pt;
                    color: #333;
                }
                QPushButton:hover {
                    background-color: #e5f3ff;
                    border: 1px solid #0078d4;
                }
                QPushButton:pressed {
                    background-color: #cce4f7;
                    border: 1px solid #005a9e;
                }
            """)
            
            if action_callback:
                btn.clicked.connect(lambda checked, n=name: action_callback(n))
            self.tool_layout.addWidget(btn)
            self.current_tools.append(btn)
            self.tool_buttons[name] = btn
        
        self.tool_layout.addStretch()

        if tool_names:
            self.setVisible(True)
            if self.auto_hide:
                self.hide_timer.start(3000)
        else:
            self.setVisible(False)

    def auto_hide_toolbar(self):
        if self.auto_hide and self.current_tools:
            self.setVisible(False)
            # parent는 central_widget이므로 MainWindow에 접근하기 위해 window() 사용
            main_window = self.window()
            if main_window and hasattr(main_window, 'ribbon_menu'):
                main_window.ribbon_menu.clear_menu_selection()

    def enterEvent(self, event):
        self.hide_timer.stop()
        if self.current_tools:
            self.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        if self.auto_hide:
            self.hide_timer.start(3000)
        super().leaveEvent(event)

    def wheelEvent(self, event):
        if event.modifiers() == Qt.ShiftModifier:
            self.scroll_area.horizontalScrollBar().wheelEvent(event)
        else:
            super().wheelEvent(event)
