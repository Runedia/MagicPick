"""
설정 다이얼로그

애플리케이션 설정을 관리하는 다이얼로그입니다.
- 일반 탭: 언어, Windows 시작 시 실행
- 캡처 탭: 단축키, 저장 위치, 파일명 규칙, 파일 형식
- 고급 탭: 캡처 옵션 (알림음, 확대창, 클립보드)
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from config.settings import settings
from config.translations import set_language, tr


class SettingsDialog(QDialog):
    """설정 다이얼로그"""

    # 설정이 변경되었을 때 발생하는 시그널
    settings_changed = pyqtSignal()
    # 언어가 변경되었을 때 발생하는 시그널
    language_changed = pyqtSignal(str)
    # 단축키가 변경되었을 때 발생하는 시그널 (단축키 키, 새 값)
    hotkey_changed = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("settings.title"))
        self.setMinimumSize(500, 450)
        self.setModal(True)

        # 임시 설정값 저장 (OK 버튼 클릭 전까지 실제 설정에 반영하지 않음)
        self._temp_settings = {}

        # 전역 단축키 매니저 참조 (있으면 가져옴)
        self._hotkey_manager = None

        self._init_ui()
        self._load_current_settings()

    def _init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 탭 위젯
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self._create_general_tab(), tr("settings.tab.general"))
        self.tab_widget.addTab(self._create_capture_tab(), tr("settings.tab.capture"))
        self.tab_widget.addTab(self._create_advanced_tab(), tr("settings.tab.advanced"))
        layout.addWidget(self.tab_widget)

        # 버튼 영역
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.reset_btn = QPushButton(tr("button.reset"))
        self.reset_btn.clicked.connect(self._on_reset_clicked)
        button_layout.addWidget(self.reset_btn)

        self.cancel_btn = QPushButton(tr("button.cancel"))
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        self.ok_btn = QPushButton(tr("button.ok"))
        self.ok_btn.setDefault(True)
        self.ok_btn.clicked.connect(self._on_ok_clicked)
        button_layout.addWidget(self.ok_btn)

        layout.addLayout(button_layout)

    def _create_general_tab(self) -> QWidget:
        """일반 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)

        # 언어 설정 그룹
        lang_group = QGroupBox(tr("settings.language"))
        lang_layout = QFormLayout(lang_group)

        self.language_combo = QComboBox()
        self.language_combo.addItem(tr("settings.language.korean"), "ko")
        self.language_combo.addItem(tr("settings.language.english"), "en")
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)
        lang_layout.addRow(tr("settings.language") + ":", self.language_combo)

        layout.addWidget(lang_group)

        # 시작 프로그램 설정 그룹
        startup_group = QGroupBox(tr("settings.start_with_windows"))
        startup_layout = QVBoxLayout(startup_group)

        self.start_with_windows_check = QCheckBox(tr("settings.start_with_windows"))
        self.start_with_windows_check.stateChanged.connect(
            self._on_start_with_windows_changed
        )
        startup_layout.addWidget(self.start_with_windows_check)

        # 개발 중 비활성화 노트
        note_label = QLabel(tr("settings.start_with_windows.note"))
        note_label.setStyleSheet("color: #888; font-size: 10pt; font-style: italic;")
        startup_layout.addWidget(note_label)

        layout.addWidget(startup_group)
        layout.addStretch()

        return widget

    def _create_capture_tab(self) -> QWidget:
        """캡처 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)

        # 단축키 설정 그룹
        hotkey_group = QGroupBox(tr("settings.hotkeys"))
        hotkey_layout = QFormLayout(hotkey_group)

        # 전체화면 캡처 단축키
        self.hotkey_fullscreen = QKeySequenceEdit()
        self.hotkey_fullscreen.editingFinished.connect(
            lambda: self._on_hotkey_changed("fullscreen")
        )
        hotkey_layout.addRow(
            tr("settings.hotkey.fullscreen") + ":", self.hotkey_fullscreen
        )

        # 영역 지정 캡처 단축키
        self.hotkey_region = QKeySequenceEdit()
        self.hotkey_region.editingFinished.connect(
            lambda: self._on_hotkey_changed("region")
        )
        hotkey_layout.addRow(tr("settings.hotkey.region") + ":", self.hotkey_region)

        # 윈도우 캡처 단축키
        self.hotkey_window = QKeySequenceEdit()
        self.hotkey_window.editingFinished.connect(
            lambda: self._on_hotkey_changed("window")
        )
        hotkey_layout.addRow(tr("settings.hotkey.window") + ":", self.hotkey_window)

        # 모니터 캡처 단축키
        self.hotkey_monitor = QKeySequenceEdit()
        self.hotkey_monitor.editingFinished.connect(
            lambda: self._on_hotkey_changed("monitor")
        )
        hotkey_layout.addRow(tr("settings.hotkey.monitor") + ":", self.hotkey_monitor)

        layout.addWidget(hotkey_group)

        # 저장 설정 그룹
        save_group = QGroupBox(tr("settings.save_location"))
        save_layout = QFormLayout(save_group)

        # 저장 경로
        path_layout = QHBoxLayout()
        self.save_path_edit = QLineEdit()
        self.save_path_edit.setReadOnly(True)
        path_layout.addWidget(self.save_path_edit)

        browse_btn = QPushButton(tr("settings.browse"))
        browse_btn.clicked.connect(self._browse_save_path)
        path_layout.addWidget(browse_btn)

        save_layout.addRow(tr("settings.save_location") + ":", path_layout)

        # 파일명 형식
        self.filename_format_edit = QLineEdit()
        self.filename_format_edit.setPlaceholderText(
            tr("settings.filename_format.hint")
        )
        self.filename_format_edit.textChanged.connect(self._on_filename_format_changed)
        save_layout.addRow(
            tr("settings.filename_format") + ":", self.filename_format_edit
        )

        # 기본 파일 형식
        self.format_combo = QComboBox()
        self.format_combo.addItem("PNG", "png")
        self.format_combo.addItem("JPEG", "jpg")
        self.format_combo.addItem("BMP", "bmp")
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        save_layout.addRow(tr("settings.default_format") + ":", self.format_combo)

        # 자동 저장 체크박스
        self.auto_save_check = QCheckBox(tr("settings.auto_save"))
        self.auto_save_check.stateChanged.connect(self._on_auto_save_changed)
        save_layout.addRow("", self.auto_save_check)

        layout.addWidget(save_group)
        layout.addStretch()

        return widget

    def _create_advanced_tab(self) -> QWidget:
        """고급 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)

        # 캡처 옵션 그룹
        options_group = QGroupBox(tr("settings.capture_options"))
        options_layout = QVBoxLayout(options_group)

        # 알림음 활성화
        self.sound_enabled_check = QCheckBox(tr("settings.sound_enabled"))
        self.sound_enabled_check.stateChanged.connect(
            lambda state: self._update_temp_setting(
                "capture_options/sound_enabled", state == Qt.Checked
            )
        )
        options_layout.addWidget(self.sound_enabled_check)

        # 클립보드 자동 복사
        self.clipboard_copy_check = QCheckBox(tr("settings.clipboard_copy"))
        self.clipboard_copy_check.stateChanged.connect(
            lambda state: self._update_temp_setting(
                "capture_options/clipboard_copy", state == Qt.Checked
            )
        )
        options_layout.addWidget(self.clipboard_copy_check)

        layout.addWidget(options_group)
        layout.addStretch()

        return widget

    def _load_current_settings(self):
        """현재 설정을 UI에 로드"""
        # 언어
        lang = settings.get("general/language")
        index = self.language_combo.findData(lang)
        if index >= 0:
            self.language_combo.setCurrentIndex(index)

        # Windows 시작 시 실행
        self.start_with_windows_check.setChecked(
            settings.get_bool("general/start_with_windows")
        )

        # 단축키
        self.hotkey_fullscreen.setKeySequence(
            QKeySequence(settings.get("hotkey/fullscreen"))
        )
        self.hotkey_region.setKeySequence(QKeySequence(settings.get("hotkey/region")))
        self.hotkey_window.setKeySequence(QKeySequence(settings.get("hotkey/window")))
        self.hotkey_monitor.setKeySequence(QKeySequence(settings.get("hotkey/monitor")))

        # 저장 경로
        self.save_path_edit.setText(settings.get("capture/save_path"))

        # 파일명 형식
        self.filename_format_edit.setText(settings.get("capture/filename_format"))

        # 기본 파일 형식
        fmt = settings.get("capture/default_format")
        index = self.format_combo.findData(fmt)
        if index >= 0:
            self.format_combo.setCurrentIndex(index)

        # 자동 저장
        self.auto_save_check.setChecked(settings.get_bool("capture/auto_save"))

        # 캡처 옵션
        self.sound_enabled_check.setChecked(
            settings.get_bool("capture_options/sound_enabled")
        )
        self.clipboard_copy_check.setChecked(
            settings.get_bool("capture_options/clipboard_copy")
        )

    def _update_temp_setting(self, key: str, value):
        """임시 설정값 업데이트"""
        self._temp_settings[key] = value

    def _on_language_changed(self, index: int):
        """언어 변경 처리"""
        lang = self.language_combo.currentData()
        self._update_temp_setting("general/language", lang)

    def _on_start_with_windows_changed(self, state: int):
        """Windows 시작 시 실행 변경 처리"""
        enabled = state == Qt.Checked
        self._update_temp_setting("general/start_with_windows", enabled)

    def _on_hotkey_changed(self, hotkey_type: str):
        """단축키 변경 처리"""
        hotkey_edits = {
            "fullscreen": self.hotkey_fullscreen,
            "region": self.hotkey_region,
            "window": self.hotkey_window,
            "monitor": self.hotkey_monitor,
        }
        edit = hotkey_edits.get(hotkey_type)
        if edit:
            key_str = edit.keySequence().toString()
            self._update_temp_setting(f"hotkey/{hotkey_type}", key_str)

    def _browse_save_path(self):
        """저장 경로 선택 다이얼로그"""
        current_path = self.save_path_edit.text()
        path = QFileDialog.getExistingDirectory(
            self, tr("dialog.select_folder"), current_path
        )
        if path:
            self.save_path_edit.setText(path)
            self._update_temp_setting("capture/save_path", path)

    def _on_filename_format_changed(self, text: str):
        """파일명 형식 변경 처리"""
        self._update_temp_setting("capture/filename_format", text)

    def _on_format_changed(self, index: int):
        """기본 파일 형식 변경 처리"""
        fmt = self.format_combo.currentData()
        self._update_temp_setting("capture/default_format", fmt)

    def _on_auto_save_changed(self, state: int):
        """자동 저장 변경 처리"""
        enabled = state == Qt.Checked
        self._update_temp_setting("capture/auto_save", enabled)

    def _on_reset_clicked(self):
        """초기화 버튼 클릭 처리"""
        settings.reset_to_defaults()
        self._temp_settings.clear()
        self._load_current_settings()

    def _on_ok_clicked(self):
        """확인 버튼 클릭 처리"""
        # 임시 설정을 실제 설정에 저장
        old_lang = settings.get("general/language")
        old_hotkeys = {
            "fullscreen": settings.get("hotkey/fullscreen"),
            "region": settings.get("hotkey/region"),
            "window": settings.get("hotkey/window"),
            "monitor": settings.get("hotkey/monitor"),
        }

        for key, value in self._temp_settings.items():
            settings.set(key, value)

        # Windows 시작 시 실행 처리 (개발 중 비활성화)
        if self._temp_settings.get("general/start_with_windows", False):
            print("[설정] Windows 시작 시 실행: 개발 중에는 지원되지 않습니다.")

        # 언어 변경 확인
        new_lang = settings.get("general/language")
        language_changed = old_lang != new_lang
        if language_changed:
            set_language(new_lang)
            self.language_changed.emit(new_lang)

        # 단축키 변경 확인
        for hotkey_type in ["fullscreen", "region", "window", "monitor"]:
            new_hotkey = settings.get(f"hotkey/{hotkey_type}")
            if old_hotkeys[hotkey_type] != new_hotkey:
                self.hotkey_changed.emit(hotkey_type, new_hotkey)

        self.settings_changed.emit()
        self.accept()

        # 언어 변경 시 재시작 안내 (다이얼로그 닫힌 후 표시)
        if language_changed:
            from PyQt5.QtWidgets import QMessageBox

            QMessageBox.information(
                self.parent(),
                "Language Changed" if new_lang == "en" else "언어 변경",
                "Please restart the application to apply the language change."
                if new_lang == "en"
                else "언어 변경을 적용하려면 프로그램을 다시 시작해주세요.",
            )

    def showEvent(self, event):
        """다이얼로그가 표시될 때 전역 단축키 일시 중지"""
        super().showEvent(event)
        self._suspend_hotkeys()

    def closeEvent(self, event):
        """다이얼로그가 닫힐 때 전역 단축키 재개"""
        self._resume_hotkeys()
        super().closeEvent(event)

    def reject(self):
        """취소 버튼 클릭 시 전역 단축키 재개"""
        self._resume_hotkeys()
        super().reject()

    def accept(self):
        """확인 버튼으로 다이얼로그 닫을 때 전역 단축키 재개"""
        self._resume_hotkeys()
        super().accept()

    def _suspend_hotkeys(self):
        """전역 단축키 일시 중지"""
        try:
            # TrayService에서 hotkey_manager 찾기
            from PyQt5.QtWidgets import QApplication

            app = QApplication.instance()
            if hasattr(app, "tray_service") and app.tray_service:
                self._hotkey_manager = app.tray_service.hotkey_manager
                if self._hotkey_manager:
                    self._hotkey_manager.suspend()
        except Exception:
            pass

    def _resume_hotkeys(self):
        """전역 단축키 재개"""
        if self._hotkey_manager:
            self._hotkey_manager.resume()
            self._hotkey_manager = None
