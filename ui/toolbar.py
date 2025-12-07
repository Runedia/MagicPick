from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QWidget,
)


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

        # 타이머 옵션 위젯들
        self.timer_checkbox = None
        self.timer_combo = None

        # 타이머 상태 저장 변수 (메뉴 전환 시에도 유지)
        self.saved_timer_checked = False
        self.saved_timer_index = 0

        # ReShade 필터 관리 (필터명 -> (콜백, 삭제콜백, 이름변경콜백))
        self.reshade_filters = {}

        self.init_ui()

    def init_ui(self):
        """UI 초기화"""

        # 스타일
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

        # 메인 레이아웃
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(5)
        self.setLayout(layout)

        # 스크롤 영역 (가로 스크롤)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setFrameShape(QScrollArea.NoFrame)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: #f5f5f5;
                border: none;
            }
            /* 가로 스크롤바 스타일 - 매우 얇게 */
            QScrollBar:horizontal {
                height: 4px;
                background: transparent;
                border: none;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background: rgba(150, 150, 150, 0.3);
                border-radius: 2px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background: rgba(100, 100, 100, 0.5);
            }
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                width: 0px;
                height: 0px;
            }
            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {
                background: none;
            }
        """)
        layout.addWidget(self.scroll_area)

        # 도구 컨테이너
        self.tool_container = QWidget()
        self.tool_layout = QHBoxLayout()
        self.tool_layout.setContentsMargins(5, 0, 5, 0)
        self.tool_layout.setSpacing(10)
        self.tool_container.setLayout(self.tool_layout)
        self.scroll_area.setWidget(self.tool_container)

        # 초기에는 숨김
        self.setVisible(False)

    def set_tools(self, tools, action_callback=None, menu_key=None):
        """
        도구 버튼 설정

        Args:
            tools: [(tool_key, display_name), ...] 튜플 리스트 또는 [name, ...] 문자열 리스트 (호환성)
            action_callback: 클릭 콜백 (tool_key를 인자로 받음)
            menu_key: 메뉴 키 (예: "capture")
        """
        # 타이머 상태 저장 (캡처 메뉴에서 벗어날 때)
        if self.timer_checkbox is not None:
            self.saved_timer_checked = self.timer_checkbox.isChecked()
        if self.timer_combo is not None:
            self.saved_timer_index = self.timer_combo.currentIndex()

        # 기존 위젯 제거
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
        self.timer_checkbox = None
        self.timer_combo = None

        # 도구 버튼 생성
        for tool in tools:
            # 튜플 (key, display_name) 또는 문자열 호환
            if isinstance(tool, tuple):
                tool_key, display_name = tool
            else:
                tool_key = tool
                display_name = tool

            btn = QPushButton(display_name)

            # 텍스트 길이에 따른 버튼 너비 계산
            font_metrics = QFontMetrics(btn.font())
            text_width = font_metrics.horizontalAdvance(display_name)
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
                btn.clicked.connect(lambda checked, k=tool_key: action_callback(k))
            self.tool_layout.addWidget(btn)
            self.current_tools.append(btn)
            self.tool_buttons[tool_key] = btn

        # 캡처 메뉴일 경우 타이머 옵션 추가
        if menu_key == "capture":
            self._add_timer_options()

        self.tool_layout.addStretch()

        # 표시/숨김
        if tools:
            self.setVisible(True)
            if self.auto_hide:
                self.hide_timer.start(3000)
        else:
            self.setVisible(False)

    def auto_hide_toolbar(self):
        """자동 숨김"""
        if self.auto_hide and self.current_tools:
            self.setVisible(False)
            # parent는 central_widget이므로 MainWindow에 접근하기 위해 window() 사용
            main_window = self.window()
            if main_window and hasattr(main_window, "ribbon_menu"):
                main_window.ribbon_menu.clear_menu_selection()

    def enterEvent(self, event):
        """마우스 진입 이벤트"""
        self.hide_timer.stop()
        if self.current_tools:
            self.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        """마우스 이탈 이벤트"""
        if self.auto_hide:
            self.hide_timer.start(3000)
        super().leaveEvent(event)

    def wheelEvent(self, event):
        """마우스 휠 이벤트 (SHIFT+휠로 가로 스크롤)"""
        if event.modifiers() == Qt.ShiftModifier:
            self.scroll_area.horizontalScrollBar().wheelEvent(event)
        else:
            super().wheelEvent(event)

    def _add_timer_options(self):
        """타이머 옵션 추가 (캡처 메뉴 전용)"""
        # 구분선 (여백)
        self.tool_layout.addSpacing(20)

        # 타이머 체크박스
        self.timer_checkbox = QCheckBox("타이머")
        self.timer_checkbox.setStyleSheet("""
            QCheckBox {
                font-size: 10pt;
                color: #333;
            }
        """)
        self.timer_checkbox.stateChanged.connect(self._on_timer_checkbox_changed)
        self.tool_layout.addWidget(self.timer_checkbox)

        # 타이머 시간 선택 콤보박스
        self.timer_combo = QComboBox()
        self.timer_combo.addItems(["3초", "5초", "10초"])
        self.timer_combo.setCurrentIndex(self.saved_timer_index)  # 저장된 값 복원
        self.timer_combo.setFixedHeight(40)
        self.timer_combo.setMinimumWidth(80)
        self.timer_combo.setStyleSheet("""
            QComboBox {
                background-color: #ffffff;
                border: 1px solid #c0c0c0;
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 10pt;
                color: #333;
            }
            QComboBox:disabled {
                background-color: #f0f0f0;
                color: #999;
            }
            QComboBox:hover {
                border: 1px solid #0078d4;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #333;
                margin-right: 5px;
            }
        """)
        self.tool_layout.addWidget(self.timer_combo)

        # 저장된 체크 상태 복원 및 콤보박스 활성화 상태 동기화
        if self.saved_timer_checked:
            self.timer_checkbox.setChecked(True)
            self.timer_combo.setEnabled(True)
        else:
            self.timer_checkbox.setChecked(False)
            self.timer_combo.setEnabled(False)

    def _on_timer_checkbox_changed(self, state):
        """타이머 체크박스 상태 변경"""
        if self.timer_combo:
            self.timer_combo.setEnabled(state == Qt.Checked)

    def get_timer_delay(self):
        """
        타이머 지연 시간 반환

        Returns:
            int: 지연 시간 (초), 타이머 비활성화 시 0
        """
        if not self.timer_checkbox or not self.timer_checkbox.isChecked():
            return 0

        if not self.timer_combo:
            return 0

        # 콤보박스 텍스트에서 숫자 추출
        text = self.timer_combo.currentText()  # '3초', '5초', '10초'
        try:
            delay = int(text.replace("초", ""))
            return delay
        except ValueError:
            return 0

    def add_reshade_filter(
        self, filter_name, click_callback, delete_callback, rename_callback
    ):
        """
        ReShade 필터 버튼 추가

        Args:
            filter_name: 필터 이름
            click_callback: 클릭 시 호출될 콜백 함수
            delete_callback: 삭제 시 호출될 콜백 함수
            rename_callback: 이름 변경 시 호출될 콜백 함수
        """
        self.reshade_filters[filter_name] = (
            click_callback,
            delete_callback,
            rename_callback,
        )

        btn = QPushButton(filter_name)

        font_metrics = QFontMetrics(btn.font())
        text_width = font_metrics.horizontalAdvance(filter_name)
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

        btn.clicked.connect(lambda: click_callback(filter_name))
        btn.setContextMenuPolicy(Qt.CustomContextMenu)
        btn.customContextMenuRequested.connect(
            lambda pos, name=filter_name: self._show_filter_context_menu(btn, name)
        )

        insert_position = self.tool_layout.count() - 1
        if insert_position < 0:
            insert_position = 0

        self.tool_layout.insertWidget(insert_position, btn)
        self.tool_buttons[filter_name] = btn

        if filter_name not in self.current_tools:
            self.current_tools.append(btn)

    def remove_reshade_filter(self, filter_name):
        """
        ReShade 필터 버튼 제거

        Args:
            filter_name: 제거할 필터 이름
        """
        if filter_name in self.tool_buttons:
            btn = self.tool_buttons[filter_name]
            self.tool_layout.removeWidget(btn)
            btn.deleteLater()
            del self.tool_buttons[filter_name]

        if filter_name in self.reshade_filters:
            del self.reshade_filters[filter_name]

        self.current_tools = [
            tool for tool in self.current_tools if isinstance(tool, QPushButton)
        ]

    def rename_reshade_filter(self, old_name, new_name):
        """
        ReShade 필터 이름 변경

        Args:
            old_name: 기존 이름
            new_name: 새 이름
        """
        if old_name not in self.tool_buttons or old_name not in self.reshade_filters:
            return

        callbacks = self.reshade_filters[old_name]

        btn = self.tool_buttons[old_name]
        btn.setText(new_name)

        font_metrics = QFontMetrics(btn.font())
        text_width = font_metrics.horizontalAdvance(new_name)
        button_width = max(100, text_width + 40)
        btn.setMinimumWidth(button_width)

        btn.clicked.disconnect()
        btn.clicked.connect(lambda: callbacks[0](new_name))
        btn.customContextMenuRequested.disconnect()
        btn.customContextMenuRequested.connect(
            lambda pos, name=new_name: self._show_filter_context_menu(btn, name)
        )

        del self.tool_buttons[old_name]
        del self.reshade_filters[old_name]

        self.tool_buttons[new_name] = btn
        self.reshade_filters[new_name] = callbacks

    def _show_filter_context_menu(self, button, filter_name):
        """
        필터 버튼 우클릭 컨텍스트 메뉴 표시

        Args:
            button: 버튼 위젯
            filter_name: 필터 이름
        """
        menu = QMenu(self)

        rename_action = menu.addAction("이름 변경")
        delete_action = menu.addAction("필터 삭제")

        action = menu.exec_(button.mapToGlobal(button.rect().bottomLeft()))

        if action == rename_action:
            self._handle_rename_filter(filter_name)
        elif action == delete_action:
            self._handle_delete_filter(filter_name)

    def _handle_rename_filter(self, filter_name):
        """
        필터 이름 변경 처리

        Args:
            filter_name: 필터 이름
        """
        if filter_name not in self.reshade_filters:
            return

        new_name, ok = QInputDialog.getText(
            self, "필터 이름 변경", "새 이름:", text=filter_name
        )

        if ok and new_name and new_name != filter_name:
            if new_name in self.tool_buttons:
                QMessageBox.warning(
                    self,
                    "이름 중복",
                    f"'{new_name}' 필터가 이미 존재합니다.",
                )
                return

            rename_callback = self.reshade_filters[filter_name][2]
            rename_callback(filter_name, new_name)

    def _handle_delete_filter(self, filter_name):
        """
        필터 삭제 처리

        Args:
            filter_name: 필터 이름
        """
        if filter_name not in self.reshade_filters:
            return

        reply = QMessageBox.question(
            self,
            "필터 삭제",
            f"'{filter_name}' 필터를 삭제하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            delete_callback = self.reshade_filters[filter_name][1]
            delete_callback(filter_name)
