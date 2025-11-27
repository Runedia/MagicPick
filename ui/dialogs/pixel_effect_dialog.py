"""
픽셀 효과 파라미터 다이얼로그 모듈

픽셀 효과 필터의 파라미터를 조절할 수 있는 다이얼로그를 제공합니다.
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)


class PixelEffectDialog(QDialog):
    """픽셀 효과 파라미터 조절 다이얼로그"""

    parameters_accepted = pyqtSignal(dict)
    parameters_changed = pyqtSignal(dict)  # 실시간 미리보기용

    def __init__(self, effect_name, default_params, parent=None):
        """
        Args:
            effect_name: 효과 이름
            default_params: 기본 파라미터 딕셔너리
            parent: 부모 위젯
        """
        super().__init__(parent)
        self.effect_name = effect_name
        self.default_params = default_params
        self.param_widgets = {}

        self.init_ui()

        # 초기 미리보기 적용
        self.emit_parameters()

    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle(f"{self.effect_name} 설정")
        self.setMinimumWidth(400)

        layout = QVBoxLayout()

        # 파라미터 그룹박스
        param_group = QGroupBox("파라미터")
        param_layout = QVBoxLayout()

        # 효과별로 다른 파라미터 UI 생성
        if "pixel_size" in self.default_params:
            self._add_slider_param(param_layout, "pixel_size", "픽셀 크기", 2, 50, 1)

        if "kernel_size" in self.default_params:
            # 가우시안/중앙값 블러는 홀수만
            if self.effect_name in ["가우시안 블러", "중앙값 블러"]:
                self._add_slider_param(
                    param_layout, "kernel_size", "커널 크기", 3, 25, 2
                )
            else:
                self._add_slider_param(
                    param_layout, "kernel_size", "커널 크기", 3, 25, 1
                )

        if "strength" in self.default_params:
            # strength는 float이므로 100배 스케일
            self._add_slider_param(
                param_layout, "strength", "강도", 50, 300, 10, scale=100
            )

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # 버튼
        button_layout = QHBoxLayout()

        ok_button = QPushButton("확인")
        ok_button.clicked.connect(self.on_accept)
        ok_button.setDefault(True)

        cancel_button = QPushButton("취소")
        cancel_button.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _add_slider_param(
        self, layout, param_name, label_text, min_val, max_val, step, scale=1
    ):
        """
        슬라이더 파라미터 추가

        Args:
            layout: 추가할 레이아웃
            param_name: 파라미터 이름
            label_text: 레이블 텍스트
            min_val: 최소값
            max_val: 최대값
            step: 스텝 크기
            scale: 스케일 (float 값을 int로 변환할 때 사용)
        """
        container = QVBoxLayout()

        # 레이블과 현재 값
        top_layout = QHBoxLayout()
        label = QLabel(label_text)
        value_label = QLabel()

        default_value = self.default_params[param_name]
        scaled_value = int(default_value * scale)

        if scale == 1:
            value_label.setText(str(default_value))
        else:
            value_label.setText(f"{default_value:.1f}")

        top_layout.addWidget(label)
        top_layout.addStretch()
        top_layout.addWidget(value_label)

        # 슬라이더
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(scaled_value)
        slider.setSingleStep(step)
        slider.setPageStep(step * 5)

        # 슬라이더 값 변경 시 레이블 업데이트 및 미리보기
        if scale == 1:
            slider.valueChanged.connect(
                lambda value: (value_label.setText(str(value)), self.emit_parameters())
            )
        else:
            slider.valueChanged.connect(
                lambda value: (
                    value_label.setText(f"{value / scale:.1f}"),
                    self.emit_parameters(),
                )
            )

        container.addLayout(top_layout)
        container.addWidget(slider)

        layout.addLayout(container)

        # 위젯 저장 (나중에 값 가져오기 위해)
        self.param_widgets[param_name] = (slider, scale)

    def get_parameters(self):
        """
        현재 파라미터 값 반환

        Returns:
            파라미터 딕셔너리
        """
        params = {}
        for param_name, (slider, scale) in self.param_widgets.items():
            value = slider.value()
            if scale == 1:
                params[param_name] = value
            else:
                params[param_name] = value / scale

        return params

    def emit_parameters(self):
        """현재 파라미터로 미리보기 시그널 발생"""
        params = self.get_parameters()
        self.parameters_changed.emit(params)

    def on_accept(self):
        """확인 버튼 클릭 시"""
        params = self.get_parameters()
        self.parameters_accepted.emit(params)
        self.accept()
