from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QPushButton, QWidget

from config.translations import tr


class RibbonMenuBar(QWidget):
    menu_changed = pyqtSignal(str)

    # 메뉴 키 (내부 식별자)
    MENU_KEYS = ["file", "edit", "capture", "filter", "tone", "style", "shader"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.current_menu = None
        self.menu_buttons = {}  # key -> button
        self.tool_actions = {}  # tool_key -> action_func
        self.dynamic_menus = {}  # 동적 메뉴 버튼
        self.layout = None
        self.stretch_item = None
        self.settings_btn = None
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
            QPushButton[contextMenu="true"] {
                background-color: #fff3cd;
                border-bottom: 2px solid #ffc107;
            }
            QPushButton[contextMenu="true"]:checked {
                background-color: #ffe69c;
                border-bottom: 2px solid #fd7e14;
            }
        """)

        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(5, 0, 5, 0)
        self.layout.setSpacing(2)

        for key in self.MENU_KEYS:
            btn = QPushButton(tr(f"menu.{key}"))
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, k=key: self.on_menu_clicked(k))
            self.layout.addWidget(btn)
            self.menu_buttons[key] = btn

        self.layout.addStretch()

        self.settings_btn = QPushButton(tr("menu.settings"))
        self.settings_btn.clicked.connect(self.open_settings)
        self.layout.addWidget(self.settings_btn)
        self.menu_buttons["settings"] = self.settings_btn

        self.setLayout(self.layout)

    def on_menu_clicked(self, menu_key):
        if self.current_menu == menu_key:
            for key, btn in self.menu_buttons.items():
                if btn.isCheckable():
                    btn.setChecked(False)
            for key, btn in self.dynamic_menus.items():
                btn.setChecked(False)
            self.current_menu = None
            self.menu_changed.emit("")
        else:
            for key, btn in self.menu_buttons.items():
                if key == menu_key:
                    btn.setChecked(True)
                    self.current_menu = menu_key
                else:
                    if btn.isCheckable():
                        btn.setChecked(False)

            for key, btn in self.dynamic_menus.items():
                if key == menu_key:
                    btn.setChecked(True)
                    self.current_menu = menu_key
                else:
                    btn.setChecked(False)

            self.menu_changed.emit(menu_key)

    def add_context_menu(
        self, menu_key: str, display_name: str, auto_select: bool = True
    ):
        """
        컨텍스트 메뉴 추가 (PowerPoint 스타일)

        Args:
            menu_key: 메뉴 키 (예: "crop")
            display_name: 표시 이름
            auto_select: True면 추가 후 자동 선택
        """
        if menu_key in self.dynamic_menus:
            return  # 이미 존재

        # 새 버튼 생성 (컨텍스트 메뉴 스타일)
        btn = QPushButton(display_name)
        btn.setCheckable(True)
        btn.setProperty("contextMenu", True)
        btn.setStyleSheet("")  # 스타일시트 재적용 트리거
        btn.clicked.connect(lambda checked, k=menu_key: self.on_menu_clicked(k))

        # settings 버튼 앞에 삽입
        settings_index = self.layout.indexOf(self.settings_btn)
        self.layout.insertWidget(settings_index - 1, btn)  # stretch 앞에

        self.dynamic_menus[menu_key] = btn

        # 자동 선택
        if auto_select:
            self.on_menu_clicked(menu_key)

    def remove_context_menu(self, menu_key: str):
        """컨텍스트 메뉴 제거"""
        if menu_key not in self.dynamic_menus:
            return

        btn = self.dynamic_menus.pop(menu_key)
        self.layout.removeWidget(btn)
        btn.deleteLater()

        # 현재 메뉴가 제거된 메뉴면 선택 해제
        if self.current_menu == menu_key:
            self.current_menu = None
            self.menu_changed.emit("")

    def has_context_menu(self, menu_key: str) -> bool:
        """컨텍스트 메뉴 존재 여부"""
        return menu_key in self.dynamic_menus

    def get_menu_tools(self, menu_key):
        """메뉴별 도구 목록 반환 (키, 표시 이름) 튜플 리스트"""
        tool_map = {
            "file": [
                ("file.open", tr("file.open")),
                ("file.save", tr("file.save")),
                ("file.save_as", tr("file.save_as")),
                ("file.exit", tr("file.exit")),
            ],
            "edit": [
                ("edit.undo", tr("edit.undo")),
                ("edit.redo", tr("edit.redo")),
                ("edit.reset", tr("edit.reset")),
                ("edit.rotate", tr("edit.rotate")),
                ("edit.flip_horizontal", tr("edit.flip_horizontal")),
                ("edit.flip_vertical", tr("edit.flip_vertical")),
                ("edit.crop", tr("edit.crop")),
                ("edit.resize", tr("edit.resize")),
            ],
            "capture": [
                ("capture.fullscreen", tr("capture.fullscreen")),
                ("capture.region", tr("capture.region")),
                ("capture.window", tr("capture.window")),
                ("capture.monitor", tr("capture.monitor")),
            ],
            "filter": [
                ("filter.soft", tr("filter.soft")),
                ("filter.sharp", tr("filter.sharp")),
                ("filter.warm", tr("filter.warm")),
                ("filter.cool", tr("filter.cool")),
                ("filter.grayscale", tr("filter.grayscale")),
                ("filter.sepia", tr("filter.sepia")),
                ("filter.photo_filter", tr("filter.photo_filter")),
            ],
            "tone": [
                ("tone.brightness", tr("tone.brightness")),
                ("tone.contrast", tr("tone.contrast")),
                ("tone.saturation", tr("tone.saturation")),
                ("tone.gamma", tr("tone.gamma")),
            ],
            "style": [
                ("style.cartoon", tr("style.cartoon")),
                ("style.sketch", tr("style.sketch")),
                ("style.oil_painting", tr("style.oil_painting")),
                ("style.film_grain", tr("style.film_grain")),
                ("style.vintage", tr("style.vintage")),
                ("style.mosaic", tr("style.mosaic")),
                ("style.gaussian_blur", tr("style.gaussian_blur")),
                ("style.average_blur", tr("style.average_blur")),
                ("style.median_blur", tr("style.median_blur")),
                ("style.sharpen", tr("style.sharpen")),
                ("style.emboss", tr("style.emboss")),
            ],
            "shader": [
                ("shader.reshade_load", tr("shader.reshade_load")),
                # ("shader.performance", tr("shader.performance")),  # 비활성화
            ],
            # 자르기 컨텍스트 메뉴
            "crop": [
                ("crop.confirm", tr("button.ok")),
                ("crop.cancel", tr("button.cancel")),
                ("crop.reset", tr("crop.reset")),
            ],
        }
        return tool_map.get(menu_key, [])

    def set_tool_action(self, tool_key, action_func):
        """도구 액션 등록 (tool_key 사용)"""
        self.tool_actions[tool_key] = action_func

    def execute_tool_action(self, tool_key):
        """도구 액션 실행"""
        if tool_key in self.tool_actions:
            self.tool_actions[tool_key]()

    def open_settings(self):
        from ui.dialogs.settings_dialog import SettingsDialog

        dialog = SettingsDialog(self.parent)
        dialog.exec_()

    def clear_menu_selection(self):
        for key, btn in self.menu_buttons.items():
            if btn.isCheckable():
                btn.setChecked(False)
        for key, btn in self.dynamic_menus.items():
            btn.setChecked(False)
        self.current_menu = None
