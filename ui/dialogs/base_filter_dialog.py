"""
필터 다이얼로그 기본 클래스

모든 필터 다이얼로그가 상속받는 추상 기반 클래스입니다.
공통 기능을 제공하여 코드 중복을 줄이고 일관성을 향상시킵니다.
"""

from abc import abstractmethod

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)


class BaseFilterDialog(QDialog):
    """필터 다이얼로그 기본 클래스 (추상 클래스)"""

    filter_applied = pyqtSignal(np.ndarray)  # 필터 적용된 이미지

    def __init__(self, original_image: np.ndarray, parent=None):
        """
        초기화

        Args:
            original_image: 원본 이미지 (NumPy array, RGB 형식)
            parent: 부모 위젯
        """
        super().__init__(parent)
        self.original_image = original_image.copy()
        self.filtered_image = None

        # 마우스 드래그 상태 플래그
        self.is_dragging = False

        # 초기 필터 적용 플래그
        self._initial_filter_applied = False

        # 서브클래스에서 필터 객체 생성
        self.filter_instance = self.create_filter()

        self.init_ui()

    @abstractmethod
    def create_filter(self):
        """
        서브클래스에서 구현: 필터 인스턴스 반환

        Returns:
            필터 인스턴스 (예: CartoonFilter())
        """
        pass

    @abstractmethod
    def build_parameter_ui(self, layout):
        """
        서브클래스에서 구현: 파라미터 UI 구성

        Args:
            layout: UI를 추가할 레이아웃
        """
        pass

    @abstractmethod
    def get_filter_params(self):
        """
        서브클래스에서 구현: 필터 파라미터 딕셔너리 반환

        Returns:
            필터 파라미터 딕셔너리
        """
        pass

    def init_ui(self):
        """공통 UI 초기화"""
        self.setModal(True)
        self.setMinimumWidth(400)

        layout = QVBoxLayout()

        # 서브클래스에서 파라미터 UI 추가
        self.build_parameter_ui(layout)

        # 공통 버튼 (확인/취소)
        button_layout = self._create_button_layout()
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _create_button_layout(self):
        """확인/취소 버튼 레이아웃 생성"""
        button_layout = QHBoxLayout()

        ok_button = QPushButton("확인")
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)

        cancel_button = QPushButton("취소")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        return button_layout

    def _create_slider_with_label(
        self,
        group_name,
        min_val,
        max_val,
        default_val,
        on_changed_callback,
        suffix="",
        tick_interval=None,
    ):
        """
        슬라이더 + 레이블 생성 헬퍼 (드래그 최적화 포함)

        Args:
            group_name: 그룹박스 이름
            min_val: 최소값
            max_val: 최대값
            default_val: 기본값
            on_changed_callback: 값 변경 시 호출할 콜백 (value 인자 받음)
            suffix: 값 뒤에 붙일 문자열 (예: "%")
            tick_interval: 틱 간격 (None이면 자동 계산)

        Returns:
            (group, slider, value_label) 튜플
        """
        group = QGroupBox(group_name)
        layout = QVBoxLayout()

        # 레이블
        value_label = QLabel(f"{default_val}{suffix}")
        value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(value_label)

        # 슬라이더
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)
        slider.setTickPosition(QSlider.TicksBelow)

        if tick_interval is None:
            tick_interval = max(1, (max_val - min_val) // 10)
        slider.setTickInterval(tick_interval)

        # 드래그 최적화 적용
        def on_value_changed(value):
            value_label.setText(f"{value}{suffix}")
            if not self.is_dragging:
                on_changed_callback(value)

        slider.valueChanged.connect(on_value_changed)
        slider.sliderPressed.connect(self.on_slider_pressed)
        slider.sliderReleased.connect(self.on_slider_released)

        layout.addWidget(slider)
        group.setLayout(layout)

        return group, slider, value_label

    def _create_combo_box(self, group_name, label_text, items, on_changed_callback):
        """
        콤보박스 생성 헬퍼

        Args:
            group_name: 그룹박스 이름
            label_text: 레이블 텍스트
            items: 콤보박스 아이템 리스트
            on_changed_callback: 인덱스 변경 시 호출할 콜백

        Returns:
            (group, combo_box) 튜플
        """
        group = QGroupBox(group_name)
        layout = QHBoxLayout()

        layout.addWidget(QLabel(label_text))

        combo = QComboBox()
        combo.addItems(items)
        combo.currentIndexChanged.connect(on_changed_callback)
        layout.addWidget(combo)

        group.setLayout(layout)

        return group, combo

    def on_slider_pressed(self):
        """슬라이더 마우스 드래그 시작"""
        self.is_dragging = True

    def on_slider_released(self):
        """슬라이더 마우스 드래그 종료 시 필터 적용"""
        self.is_dragging = False
        self.apply_filter()

    def apply_filter(self):
        """필터 적용 (공통 로직)"""
        params = self.get_filter_params()
        self.filtered_image = self.filter_instance.apply(self.original_image, **params)
        self.filter_applied.emit(self.filtered_image)

    def get_filtered_image(self):
        """필터가 적용된 이미지 반환"""
        return self.filtered_image

    def get_settings(self):
        """현재 설정 반환 (기본값: get_filter_params와 동일)"""
        return self.get_filter_params()

    def showEvent(self, event):
        """다이얼로그가 표시될 때 초기 필터 적용"""
        super().showEvent(event)
        if not self._initial_filter_applied:
            self._initial_filter_applied = True
            self.apply_filter()
