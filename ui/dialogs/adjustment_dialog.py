"""
이미지 조정 다이얼로그 모듈

밝기, 대비, 채도, 감마를 통합적으로 조정할 수 있는 다이얼로그를 제공합니다.
"""

import numpy as np
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

from editor.adjustments import ImageAdjustments


class AdjustmentDialog(QDialog):
    """이미지 조정 다이얼로그 (밝기, 대비, 채도, 감마)"""

    # 조정 미리보기 시그널 (실시간)
    adjustment_preview = pyqtSignal(np.ndarray)
    # 조정 확정 시그널
    adjustment_accepted = pyqtSignal(np.ndarray, str)  # image, description

    def __init__(self, original_image: np.ndarray, parent=None):
        """
        초기화

        Args:
            original_image: 원본 이미지 (NumPy array, RGB 형식)
            parent: 부모 위젯
        """
        super().__init__(parent)
        self.original_image = original_image.copy()
        self.current_image = original_image.copy()

        # 슬라이더 위젯 저장용
        self.brightness_slider = None
        self.contrast_slider = None
        self.saturation_slider = None
        self.gamma_slider = None

        # 값 레이블 저장용
        self.brightness_label = None
        self.contrast_label = None
        self.saturation_label = None
        self.gamma_label = None

        # 마우스 드래그 상태 플래그
        self.is_dragging = False

        self.init_ui()

        # 초기 미리보기 표시
        self.apply_adjustments()

    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("이미지 조정")
        self.setModal(True)
        self.setMinimumWidth(450)
        self.setMinimumHeight(400)

        layout = QVBoxLayout()

        # 조정 항목 그룹박스
        adjustments_group = QGroupBox("조정")
        adjustments_layout = QVBoxLayout()

        # 밝기 슬라이더 (-100 ~ 100, 기본값 0)
        self._add_adjustment_slider(
            adjustments_layout,
            "밝기",
            -100,
            100,
            0,
            "brightness_slider",
            "brightness_label",
        )

        # 대비 슬라이더 (-100 ~ 100, 기본값 0)
        self._add_adjustment_slider(
            adjustments_layout,
            "대비",
            -100,
            100,
            0,
            "contrast_slider",
            "contrast_label",
        )

        # 채도 슬라이더 (0 ~ 200, 기본값 100)
        self._add_adjustment_slider(
            adjustments_layout,
            "채도",
            0,
            200,
            100,
            "saturation_slider",
            "saturation_label",
        )

        # 감마 슬라이더 (0.3 ~ 3.0, 기본값 1.0, 0.1 단위)
        # 슬라이더는 정수만 사용하므로 10배 스케일 (3 ~ 30, 기본값 10)
        self._add_adjustment_slider(
            adjustments_layout,
            "감마",
            3,
            30,
            10,
            "gamma_slider",
            "gamma_label",
            scale=10,
            format_str="{:.1f}",
        )

        adjustments_group.setLayout(adjustments_layout)
        layout.addWidget(adjustments_group)

        # 미리보기 안내
        preview_group = QGroupBox("미리보기")
        preview_layout = QVBoxLayout()

        preview_label = QLabel("조정 결과가 메인 창에 실시간으로 표시됩니다")
        preview_label.setAlignment(Qt.AlignCenter)
        preview_label.setMinimumHeight(50)
        preview_label.setStyleSheet(
            "QLabel { background-color: #f0f0f0; border: 1px solid #ccc; "
            "padding: 10px; }"
        )
        preview_layout.addWidget(preview_label)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        # 버튼 레이아웃
        button_layout = QHBoxLayout()

        reset_button = QPushButton("초기화")
        reset_button.clicked.connect(self.reset_adjustments)
        button_layout.addWidget(reset_button)

        button_layout.addStretch()

        ok_button = QPushButton("확인")
        ok_button.setDefault(True)
        ok_button.clicked.connect(self.on_accept)
        button_layout.addWidget(ok_button)

        cancel_button = QPushButton("취소")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _add_adjustment_slider(
        self,
        layout,
        label_text,
        min_val,
        max_val,
        default_val,
        slider_attr,
        label_attr,
        scale=1,
        format_str="{}",
    ):
        """
        조정 슬라이더 추가

        Args:
            layout: 추가할 레이아웃
            label_text: 레이블 텍스트 (예: '밝기')
            min_val: 최소값
            max_val: 최대값
            default_val: 기본값
            slider_attr: 슬라이더 속성 이름
            label_attr: 값 레이블 속성 이름
            scale: 스케일 (float을 int로 변환)
            format_str: 값 포맷 문자열
        """
        container = QVBoxLayout()

        # 상단 레이블 (이름 + 현재 값)
        top_layout = QHBoxLayout()
        name_label = QLabel(label_text)
        value_label = QLabel()

        # 기본값 표시
        if scale == 1:
            value_label.setText(format_str.format(default_val))
        else:
            value_label.setText(format_str.format(default_val / scale))

        value_label.setStyleSheet("font-weight: bold;")

        top_layout.addWidget(name_label)
        top_layout.addStretch()
        top_layout.addWidget(value_label)

        # 슬라이더
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval((max_val - min_val) // 10)

        # 슬라이더 값 변경 시 레이블 업데이트 및 조건부 미리보기
        def on_value_changed(value):
            # 레이블 업데이트 (항상)
            if scale == 1:
                value_label.setText(format_str.format(value))
            else:
                value_label.setText(format_str.format(value / scale))

            # 미리보기 적용 (드래그 중이 아닐 때만 - 키보드 입력)
            if not self.is_dragging:
                self.apply_adjustments()

        slider.valueChanged.connect(on_value_changed)

        # 마우스 드래그 시작/종료 시그널 연결
        slider.sliderPressed.connect(self.on_slider_pressed)
        slider.sliderReleased.connect(self.on_slider_released)

        container.addLayout(top_layout)
        container.addWidget(slider)

        layout.addLayout(container)

        # 위젯을 인스턴스 속성으로 저장
        setattr(self, slider_attr, slider)
        setattr(self, label_attr, value_label)

    def apply_adjustments(self):
        """현재 슬라이더 값으로 모든 조정 적용"""
        # 원본 이미지에서 시작
        result = self.original_image.copy()

        # 밝기 조정
        brightness = self.brightness_slider.value()
        if brightness != 0:
            result = ImageAdjustments.adjust_brightness(result, brightness)

        # 대비 조정
        contrast = self.contrast_slider.value()
        if contrast != 0:
            result = ImageAdjustments.adjust_contrast(result, contrast)

        # 채도 조정
        saturation = self.saturation_slider.value()
        if saturation != 100:
            result = ImageAdjustments.adjust_saturation(result, saturation)

        # 감마 조정
        gamma = self.gamma_slider.value() / 10.0
        if gamma != 1.0:
            result = ImageAdjustments.adjust_gamma(result, gamma)

        self.current_image = result

        # 메인 창에 미리보기 표시
        self.adjustment_preview.emit(result)

    def on_slider_pressed(self):
        """슬라이더 마우스 드래그 시작"""
        self.is_dragging = True

    def on_slider_released(self):
        """슬라이더 마우스 드래그 종료 시 미리보기 적용"""
        self.is_dragging = False
        self.apply_adjustments()

    def reset_adjustments(self):
        """모든 조정 값을 기본값으로 초기화"""
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(0)
        self.saturation_slider.setValue(100)
        self.gamma_slider.setValue(10)

    def get_description(self):
        """현재 조정 내용을 설명하는 문자열 반환"""
        parts = []

        brightness = self.brightness_slider.value()
        if brightness != 0:
            parts.append(f"밝기{brightness:+d}")

        contrast = self.contrast_slider.value()
        if contrast != 0:
            parts.append(f"대비{contrast:+d}")

        saturation = self.saturation_slider.value()
        if saturation != 100:
            parts.append(f"채도{saturation}")

        gamma = self.gamma_slider.value() / 10.0
        if gamma != 1.0:
            parts.append(f"감마{gamma:.1f}")

        if not parts:
            return "조정 없음"

        return ", ".join(parts)

    def on_accept(self):
        """확인 버튼 클릭 시"""
        description = self.get_description()
        self.adjustment_accepted.emit(self.current_image, description)
        self.accept()

    def get_adjusted_image(self):
        """조정된 이미지 반환"""
        return self.current_image
