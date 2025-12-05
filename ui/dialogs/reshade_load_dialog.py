"""
ReShade INI 파일 불러오기 다이얼로그

ReShade 프리셋 INI 파일을 선택하고 로드하는 다이얼로그입니다.
"""

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)


class ReShadeLoadDialog(QDialog):
    """ReShade INI 파일 불러오기 다이얼로그"""

    preset_loaded = pyqtSignal(str, object, list)

    def __init__(self, preset_manager, parent=None):
        """
        초기화

        Args:
            preset_manager: ReShadePresetManager 인스턴스
            parent: 부모 위젯
        """
        super().__init__(parent)
        self.preset_manager = preset_manager
        self.selected_ini_path = None
        self.preset_name = None
        self.reshade_filter = None
        self.unsupported_effects = []

        self.init_ui()

    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("ReShade 프리셋 불러오기")
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)

        layout = QVBoxLayout()

        file_group = QGroupBox("INI 파일 선택")
        file_layout = QVBoxLayout()

        file_select_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setPlaceholderText("INI 파일을 선택하세요...")
        file_select_layout.addWidget(self.file_path_edit)

        self.browse_button = QPushButton("찾아보기")
        self.browse_button.clicked.connect(self.browse_file)
        file_select_layout.addWidget(self.browse_button)

        file_layout.addLayout(file_select_layout)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        name_group = QGroupBox("프리셋 이름")
        name_layout = QHBoxLayout()

        name_layout.addWidget(QLabel("이름:"))
        self.preset_name_edit = QLineEdit()
        self.preset_name_edit.setPlaceholderText("프리셋 이름 (비워두면 파일명 사용)")
        name_layout.addWidget(self.preset_name_edit)

        name_group.setLayout(name_layout)
        layout.addWidget(name_group)

        effects_group = QGroupBox("효과 정보")
        effects_layout = QVBoxLayout()

        self.effects_text = QTextEdit()
        self.effects_text.setReadOnly(True)
        self.effects_text.setMaximumHeight(150)
        self.effects_text.setPlaceholderText(
            "INI 파일을 선택하면 효과 정보가 표시됩니다"
        )
        effects_layout.addWidget(self.effects_text)

        effects_group.setLayout(effects_layout)
        layout.addWidget(effects_group)

        unsupported_group = QGroupBox("미구현 효과")
        unsupported_layout = QVBoxLayout()

        self.unsupported_list = QListWidget()
        self.unsupported_list.setMaximumHeight(100)
        unsupported_layout.addWidget(self.unsupported_list)

        info_label = QLabel("위 효과들은 현재 구현되지 않아 적용되지 않습니다.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { color: #888; font-size: 9pt; }")
        unsupported_layout.addWidget(info_label)

        unsupported_group.setLayout(unsupported_layout)
        layout.addWidget(unsupported_group)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.load_button = QPushButton("불러오기")
        self.load_button.setEnabled(False)
        self.load_button.setDefault(True)
        self.load_button.clicked.connect(self.load_preset)
        button_layout.addWidget(self.load_button)

        self.cancel_button = QPushButton("취소")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def browse_file(self):
        """파일 선택 다이얼로그 열기"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "ReShade INI 파일 선택", "", "INI Files (*.ini);;All Files (*)"
        )

        if file_path:
            self.selected_ini_path = file_path
            self.file_path_edit.setText(file_path)
            self.preview_preset()

    def preview_preset(self):
        """선택된 INI 파일 미리보기"""
        if not self.selected_ini_path:
            return

        try:
            (
                preset_name,
                reshade_filter,
                unsupported_effects,
            ) = self.preset_manager.load_preset_from_ini(self.selected_ini_path)

            if preset_name is None or reshade_filter is None:
                QMessageBox.warning(
                    self,
                    "오류",
                    "INI 파일을 파싱할 수 없거나 구현 가능한 효과가 없습니다.",
                )
                self.load_button.setEnabled(False)
                return

            self.preset_name = preset_name
            self.reshade_filter = reshade_filter
            self.unsupported_effects = unsupported_effects

            if not self.preset_name_edit.text():
                self.preset_name_edit.setText(preset_name)

            effects_info = []
            for effect_name, params in reshade_filter.effects.items():
                effects_info.append(f"[{effect_name}]")
                for param_name, param_value in params.items():
                    if isinstance(param_value, tuple):
                        param_value = ", ".join([f"{v:.3f}" for v in param_value])
                    elif isinstance(param_value, float):
                        param_value = f"{param_value:.3f}"
                    effects_info.append(f"  {param_name}: {param_value}")
                effects_info.append("")

            self.effects_text.setText("\n".join(effects_info))

            self.unsupported_list.clear()
            if unsupported_effects:
                self.unsupported_list.addItems(unsupported_effects)

            self.load_button.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "오류", f"INI 파일 로드 실패:\n{str(e)}")
            self.load_button.setEnabled(False)

    def load_preset(self):
        """프리셋 불러오기"""
        # 사용자 정의 이름 가져오기 (비어 있으면 원본 이름 사용)
        final_name = self.preset_name_edit.text().strip() or self.preset_name

        # 이미 존재하는 프리셋인지 확인 (덮어쓰기 확인)
        if self.preset_manager.preset_exists(final_name):
            reply = QMessageBox.question(
                self,
                "이름 중복",
                f"'{final_name}' 프리셋이 이미 존재합니다.\n덮어쓰시겠습니까?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return

        # INI 파일에서 프리셋을 로드하고 저장 (save=True)
        (
            preset_name,
            reshade_filter,
            unsupported_effects,
        ) = self.preset_manager.load_preset_from_ini(
            self.selected_ini_path, final_name, save=True
        )

        if preset_name is None:
            QMessageBox.critical(self, "오류", "프리셋 저장 중 오류가 발생했습니다.")
            return

        self.preset_name = preset_name
        self.reshade_filter = reshade_filter
        self.unsupported_effects = unsupported_effects

        if self.unsupported_effects:
            unsupported_str = "\n".join(f"- {e}" for e in self.unsupported_effects)
            QMessageBox.information(
                self,
                "미구현 효과",
                f"다음 효과는 구현되지 않아 적용되지 않습니다:\n\n{unsupported_str}",
            )

        self.preset_loaded.emit(
            self.preset_name, self.reshade_filter, self.unsupported_effects
        )
        self.accept()

    def get_result(self):
        """로드된 프리셋 정보 반환"""
        return self.preset_name, self.reshade_filter, self.unsupported_effects
