"""
회전 다이얼로그 모듈

다이얼(Dial) UI를 통해 이미지 회전 각도를 조절할 수 있는 다이얼로그를 제공합니다.
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QDial, QPushButton, QGroupBox, QCheckBox, QSpinBox)
from PyQt5.QtCore import Qt, pyqtSignal


class RotateDialog(QDialog):
    """이미지 회전 다이얼로그"""

    rotation_accepted = pyqtSignal(float, bool)  # angle, expand
    rotation_preview = pyqtSignal(float, bool)  # angle, expand (실시간 미리보기용)

    def __init__(self, parent=None):
        """
        Args:
            parent: 부모 위젯
        """
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle('이미지 회전')
        self.setMinimumWidth(400)
        self.setMinimumHeight(350)

        layout = QVBoxLayout()

        # 회전 각도 그룹박스
        rotation_group = QGroupBox('회전 각도')
        rotation_layout = QVBoxLayout()

        # 현재 각도 표시
        angle_layout = QHBoxLayout()
        angle_label = QLabel('각도:')
        self.angle_value_label = QLabel('0°')
        self.angle_value_label.setStyleSheet('font-size: 16pt; font-weight: bold;')
        angle_layout.addWidget(angle_label)
        angle_layout.addStretch()
        angle_layout.addWidget(self.angle_value_label)
        rotation_layout.addLayout(angle_layout)

        # 다이얼 (0-360도)
        self.dial = QDial()
        self.dial.setMinimum(0)
        self.dial.setMaximum(359)
        self.dial.setValue(0)
        self.dial.setWrapping(True)
        self.dial.setNotchesVisible(True)
        self.dial.valueChanged.connect(self.on_dial_changed)

        rotation_layout.addWidget(self.dial, alignment=Qt.AlignCenter)

        # 정확한 각도 입력 (SpinBox)
        spinbox_layout = QHBoxLayout()
        spinbox_label = QLabel('정확한 각도:')
        self.angle_spinbox = QSpinBox()
        self.angle_spinbox.setMinimum(0)
        self.angle_spinbox.setMaximum(359)
        self.angle_spinbox.setValue(0)
        self.angle_spinbox.setSuffix('°')
        self.angle_spinbox.valueChanged.connect(self.on_spinbox_changed)

        spinbox_layout.addWidget(spinbox_label)
        spinbox_layout.addWidget(self.angle_spinbox)
        spinbox_layout.addStretch()
        rotation_layout.addLayout(spinbox_layout)

        rotation_group.setLayout(rotation_layout)
        layout.addWidget(rotation_group)

        # 옵션 그룹박스
        options_group = QGroupBox('옵션')
        options_layout = QVBoxLayout()

        # 캔버스 확장 옵션
        self.expand_checkbox = QCheckBox('회전된 이미지 전체를 포함하도록 캔버스 확장')
        self.expand_checkbox.setChecked(True)
        options_layout.addWidget(self.expand_checkbox)

        # 90도 스냅 옵션
        self.snap_checkbox = QCheckBox('90도 단위로 스냅')
        self.snap_checkbox.setChecked(True)
        self.snap_checkbox.stateChanged.connect(self.on_snap_changed)
        options_layout.addWidget(self.snap_checkbox)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # 빠른 회전 버튼
        quick_rotate_layout = QHBoxLayout()
        quick_rotate_label = QLabel('빠른 회전:')

        btn_90 = QPushButton('90°')
        btn_90.clicked.connect(lambda: self.set_angle(90))

        btn_180 = QPushButton('180°')
        btn_180.clicked.connect(lambda: self.set_angle(180))

        btn_270 = QPushButton('270°')
        btn_270.clicked.connect(lambda: self.set_angle(270))

        btn_reset = QPushButton('0°')
        btn_reset.clicked.connect(lambda: self.set_angle(0))

        quick_rotate_layout.addWidget(quick_rotate_label)
        quick_rotate_layout.addWidget(btn_reset)
        quick_rotate_layout.addWidget(btn_90)
        quick_rotate_layout.addWidget(btn_180)
        quick_rotate_layout.addWidget(btn_270)
        quick_rotate_layout.addStretch()

        layout.addLayout(quick_rotate_layout)

        # 확인/취소 버튼
        button_layout = QHBoxLayout()

        ok_button = QPushButton('확인')
        ok_button.clicked.connect(self.on_accept)
        ok_button.setDefault(True)

        cancel_button = QPushButton('취소')
        cancel_button.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def on_dial_changed(self, value):
        """다이얼 값 변경 시"""
        if self.snap_checkbox.isChecked():
            # 90도 단위로 스냅
            snapped = round(value / 90) * 90
            if snapped != value:
                self.dial.setValue(snapped)
                return

        self.angle_spinbox.blockSignals(True)
        self.angle_spinbox.setValue(value)
        self.angle_spinbox.blockSignals(False)

        self.angle_value_label.setText(f'{value}°')

        # 실시간 미리보기
        self.rotation_preview.emit(value, self.get_expand())

    def on_spinbox_changed(self, value):
        """SpinBox 값 변경 시"""
        self.dial.blockSignals(True)
        self.dial.setValue(value)
        self.dial.blockSignals(False)

        self.angle_value_label.setText(f'{value}°')

        # 실시간 미리보기
        self.rotation_preview.emit(value, self.get_expand())

    def on_snap_changed(self, state):
        """스냅 옵션 변경 시"""
        if state == Qt.Checked:
            # 현재 값을 가장 가까운 90도 배수로 조정
            current = self.dial.value()
            snapped = round(current / 90) * 90
            self.set_angle(snapped)

    def set_angle(self, angle):
        """각도 설정"""
        angle = angle % 360
        self.dial.setValue(angle)

    def get_angle(self):
        """현재 각도 반환"""
        return self.dial.value()

    def get_expand(self):
        """캔버스 확장 옵션 반환"""
        return self.expand_checkbox.isChecked()

    def on_accept(self):
        """확인 버튼 클릭 시"""
        angle = self.get_angle()
        expand = self.get_expand()
        self.rotation_accepted.emit(angle, expand)
        self.accept()
