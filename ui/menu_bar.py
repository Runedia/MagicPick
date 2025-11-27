from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QPushButton, QWidget


class RibbonMenuBar(QWidget):
    menu_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.current_menu = None
        self.menu_buttons = {}
        self.tool_actions = {}
        self.init_ui()

    def init_ui(self):
        self.setMaximumHeight(40)
        self.setMinimumHeight(40)
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                border-bottom: 1px solid #ccc;
            }
            QPushButton {
                background-color: transparent;
                border: none;
                padding: 8px 16px;
                font-size: 11pt;
                color: #333;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:checked {
                background-color: #d0d0d0;
                border-bottom: 2px solid #0078d4;
            }
        """)

        layout = QHBoxLayout()
        layout.setContentsMargins(5, 0, 5, 0)
        layout.setSpacing(2)

        menu_names = ["파일", "편집", "캡처", "필터", "색조", "스타일", "셰이더"]

        for name in menu_names:
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, n=name: self.on_menu_clicked(n))
            layout.addWidget(btn)
            self.menu_buttons[name] = btn

        layout.addStretch()

        settings_btn = QPushButton("설정")
        settings_btn.clicked.connect(self.open_settings)
        layout.addWidget(settings_btn)
        self.menu_buttons["설정"] = settings_btn

        self.setLayout(layout)

    def on_menu_clicked(self, menu_name):
        if self.current_menu == menu_name:
            for name, btn in self.menu_buttons.items():
                if btn.isCheckable():
                    btn.setChecked(False)
            self.current_menu = None
            self.menu_changed.emit("")
        else:
            for name, btn in self.menu_buttons.items():
                if name == menu_name:
                    btn.setChecked(True)
                    self.current_menu = menu_name
                else:
                    if btn.isCheckable():
                        btn.setChecked(False)

            self.menu_changed.emit(menu_name)

    def get_menu_tools(self, menu_name):
        if menu_name == "파일":
            return ["열기", "저장", "다른 이름으로 저장", "끝내기"]
        elif menu_name == "편집":
            return [
                "실행 취소",
                "다시 실행",
                "초기화",
                "회전",
                "좌우 반전",
                "상하 반전",
            ]
        elif menu_name == "캡처":
            return ["전체화면", "영역 지정", "윈도우", "모니터"]
        elif menu_name == "필터":
            return [
                "부드러운",
                "선명한",
                "따뜻한",
                "차가운",
                "회색조",
                "세피아",
                "Photo Filter",
            ]
        elif menu_name == "색조":
            return ["밝기", "대비", "채도", "감마"]
        elif menu_name == "스타일":
            return [
                "카툰",
                "스케치",
                "유화",
                "필름 그레인",
                "빈티지",
                "모자이크",
                "가우시안 블러",
                "평균 블러",
                "중앙값 블러",
                "샤프닝",
                "엠보싱",
            ]
        elif menu_name == "셰이더":
            return ["ReShade 불러오기"]
        return []

    def set_tool_action(self, tool_name, action_func):
        self.tool_actions[tool_name] = action_func

    def execute_tool_action(self, tool_name):
        if tool_name in self.tool_actions:
            self.tool_actions[tool_name]()

    def open_settings(self):
        pass

    def clear_menu_selection(self):
        for name, btn in self.menu_buttons.items():
            if btn.isCheckable():
                btn.setChecked(False)
        self.current_menu = None
