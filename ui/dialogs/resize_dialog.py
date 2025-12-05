"""
이미지 크기 조절 다이얼로그

리사이즈 모드:
- 크기 변경 안함
- 비율 유지
- 폭맞춤
- 높이맞춤
- 여백 붙이기
- 여백 자르기
- 꽉차게 늘리기
"""

from enum import Enum

import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from config.translations import tr


class ResizeMode(Enum):
    """리사이즈 모드"""

    NO_RESIZE = "no_resize"  # 크기 변경 안함
    MAINTAIN_ASPECT = "maintain_aspect"  # 비율 유지
    FIT_WIDTH = "fit_width"  # 폭맞춤
    FIT_HEIGHT = "fit_height"  # 높이맞춤
    ADD_PADDING = "add_padding"  # 여백 붙이기
    CROP_TO_FIT = "crop_to_fit"  # 여백 자르기
    STRETCH = "stretch"  # 꽉차게 늘리기


class ResizeDialog(QDialog):
    """이미지 크기 조절 다이얼로그"""

    preview_requested = pyqtSignal(np.ndarray)
    resize_applied = pyqtSignal(np.ndarray, str)

    def __init__(self, parent=None, image: np.ndarray = None):
        super().__init__(parent)
        self.original_image = image
        self.current_image = image
        self.original_width = image.shape[1] if image is not None else 1920
        self.original_height = image.shape[0] if image is not None else 1080
        self.aspect_ratio = self.original_width / self.original_height

        self._updating = False  # 순환 업데이트 방지

        self._init_ui()
        self._load_current_size()

    def _init_ui(self):
        """UI 초기화"""
        self.setWindowTitle(tr("resize.title"))
        self.setMinimumWidth(350)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        layout = QVBoxLayout(self)

        # 크기 조절 그룹
        size_group = QGroupBox(tr("resize.title"))
        size_layout = QVBoxLayout(size_group)

        # 리사이즈 모드 선택
        mode_layout = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(tr("resize.no_resize"), ResizeMode.NO_RESIZE)
        self.mode_combo.addItem(
            tr("resize.maintain_aspect"), ResizeMode.MAINTAIN_ASPECT
        )
        self.mode_combo.addItem(tr("resize.fit_width"), ResizeMode.FIT_WIDTH)
        self.mode_combo.addItem(tr("resize.fit_height"), ResizeMode.FIT_HEIGHT)
        self.mode_combo.addItem(tr("resize.add_padding"), ResizeMode.ADD_PADDING)
        self.mode_combo.addItem(tr("resize.crop_to_fit"), ResizeMode.CROP_TO_FIT)
        self.mode_combo.addItem(tr("resize.stretch"), ResizeMode.STRETCH)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        size_layout.addLayout(mode_layout)

        # 너비/높이 입력
        dim_layout = QHBoxLayout()

        # 너비
        w_layout = QHBoxLayout()
        w_label = QLabel("W")
        w_label.setFixedWidth(20)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 32768)
        self.width_spin.setValue(self.original_width)
        self.width_spin.valueChanged.connect(self._on_width_changed)
        w_layout.addWidget(w_label)
        w_layout.addWidget(self.width_spin)
        dim_layout.addLayout(w_layout)

        # 높이
        h_layout = QHBoxLayout()
        h_label = QLabel("H")
        h_label.setFixedWidth(20)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 32768)
        self.height_spin.setValue(self.original_height)
        self.height_spin.valueChanged.connect(self._on_height_changed)
        h_layout.addWidget(h_label)
        h_layout.addWidget(self.height_spin)
        dim_layout.addLayout(h_layout)

        size_layout.addLayout(dim_layout)

        # 작은 이미지도 리사이징 체크박스
        self.upscale_check = QCheckBox(tr("resize.upscale_small"))
        self.upscale_check.setChecked(False)
        self.upscale_check.stateChanged.connect(self._on_upscale_changed)
        size_layout.addWidget(self.upscale_check)

        layout.addWidget(size_group)

        # 원본 크기 정보
        info_layout = QFormLayout()
        self.original_size_label = QLabel(
            f"{self.original_width} x {self.original_height}"
        )
        info_layout.addRow(tr("resize.original_size"), self.original_size_label)

        self.result_size_label = QLabel(
            f"{self.original_width} x {self.original_height}"
        )
        info_layout.addRow(tr("resize.result_size"), self.result_size_label)
        layout.addLayout(info_layout)

        # 버튼
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._on_ok)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # 초기 상태: 크기 변경 안함
        self.mode_combo.setCurrentIndex(0)
        self._update_ui_state()

    def _load_current_size(self):
        """현재 이미지 크기 로드"""
        if self.original_image is not None:
            self.original_width = self.original_image.shape[1]
            self.original_height = self.original_image.shape[0]
            self.aspect_ratio = self.original_width / self.original_height

            self._updating = True
            self.width_spin.setValue(self.original_width)
            self.height_spin.setValue(self.original_height)
            self._updating = False

            self.original_size_label.setText(
                f"{self.original_width} x {self.original_height}"
            )
            self._update_result_size()

    def _on_mode_changed(self, index: int):
        """모드 변경 처리"""
        self._update_ui_state()
        self._update_result_size()

    def _update_ui_state(self):
        """UI 상태 업데이트"""
        mode = self.mode_combo.currentData()

        if mode == ResizeMode.NO_RESIZE:
            # 크기 변경 안함: 입력 비활성화
            self.width_spin.setEnabled(False)
            self.height_spin.setEnabled(False)
            self._updating = True
            self.width_spin.setValue(self.original_width)
            self.height_spin.setValue(self.original_height)
            self._updating = False
        elif mode in (
            ResizeMode.MAINTAIN_ASPECT,
            ResizeMode.ADD_PADDING,
            ResizeMode.CROP_TO_FIT,
        ):
            # 비율 유지 관련: 둘 다 활성화
            self.width_spin.setEnabled(True)
            self.height_spin.setEnabled(True)
        elif mode == ResizeMode.FIT_WIDTH:
            # 폭맞춤: 너비만 활성화
            self.width_spin.setEnabled(True)
            self.height_spin.setEnabled(False)
        elif mode == ResizeMode.FIT_HEIGHT:
            # 높이맞춤: 높이만 활성화
            self.width_spin.setEnabled(False)
            self.height_spin.setEnabled(True)
        elif mode == ResizeMode.STRETCH:
            # 꽉차게 늘리기: 둘 다 활성화
            self.width_spin.setEnabled(True)
            self.height_spin.setEnabled(True)

    def _on_width_changed(self, value: int):
        """너비 변경 처리"""
        if self._updating:
            return

        mode = self.mode_combo.currentData()

        if mode == ResizeMode.MAINTAIN_ASPECT:
            # 비율 유지: 높이 자동 계산
            self._updating = True
            new_height = int(value / self.aspect_ratio)
            self.height_spin.setValue(max(1, new_height))
            self._updating = False
        elif mode == ResizeMode.FIT_WIDTH:
            # 폭맞춤: 높이 자동 계산
            self._updating = True
            new_height = int(value / self.aspect_ratio)
            self.height_spin.setValue(max(1, new_height))
            self._updating = False

        self._update_result_size()

    def _on_height_changed(self, value: int):
        """높이 변경 처리"""
        if self._updating:
            return

        mode = self.mode_combo.currentData()

        if mode == ResizeMode.MAINTAIN_ASPECT:
            # 비율 유지: 너비 자동 계산
            self._updating = True
            new_width = int(value * self.aspect_ratio)
            self.width_spin.setValue(max(1, new_width))
            self._updating = False
        elif mode == ResizeMode.FIT_HEIGHT:
            # 높이맞춤: 너비 자동 계산
            self._updating = True
            new_width = int(value * self.aspect_ratio)
            self.width_spin.setValue(max(1, new_width))
            self._updating = False

        self._update_result_size()

    def _on_upscale_changed(self, state: int):
        """작은 이미지 리사이징 체크박스 변경"""
        self._update_result_size()

    def _update_result_size(self):
        """결과 크기 업데이트"""
        result_width, result_height = self._calculate_result_size()
        self.result_size_label.setText(f"{result_width} x {result_height}")

    def _calculate_result_size(self) -> tuple:
        """결과 크기 계산"""
        mode = self.mode_combo.currentData()
        target_width = self.width_spin.value()
        target_height = self.height_spin.value()
        allow_upscale = self.upscale_check.isChecked()

        if mode == ResizeMode.NO_RESIZE:
            return self.original_width, self.original_height

        elif mode == ResizeMode.MAINTAIN_ASPECT:
            # 비율 유지하면서 지정 크기에 맞춤 (작은 쪽 기준)
            scale = min(
                target_width / self.original_width, target_height / self.original_height
            )
            if not allow_upscale:
                scale = min(scale, 1.0)
            return int(self.original_width * scale), int(self.original_height * scale)

        elif mode == ResizeMode.FIT_WIDTH:
            # 폭맞춤
            scale = target_width / self.original_width
            if not allow_upscale:
                scale = min(scale, 1.0)
            return int(self.original_width * scale), int(self.original_height * scale)

        elif mode == ResizeMode.FIT_HEIGHT:
            # 높이맞춤
            scale = target_height / self.original_height
            if not allow_upscale:
                scale = min(scale, 1.0)
            return int(self.original_width * scale), int(self.original_height * scale)

        elif mode == ResizeMode.ADD_PADDING:
            # 여백 붙이기: 지정 크기 그대로 (이미지는 비율 유지 후 여백)
            return target_width, target_height

        elif mode == ResizeMode.CROP_TO_FIT:
            # 여백 자르기: 지정 크기 그대로 (이미지는 비율 유지 후 자르기)
            return target_width, target_height

        elif mode == ResizeMode.STRETCH:
            # 꽉차게 늘리기
            return target_width, target_height

        return self.original_width, self.original_height

    def _apply_resize(self) -> np.ndarray:
        """리사이즈 적용"""
        if self.original_image is None:
            return None

        mode = self.mode_combo.currentData()
        target_width = self.width_spin.value()
        target_height = self.height_spin.value()
        allow_upscale = self.upscale_check.isChecked()

        pil_image = Image.fromarray(self.original_image)

        if mode == ResizeMode.NO_RESIZE:
            return self.original_image.copy()

        elif mode == ResizeMode.MAINTAIN_ASPECT:
            # 비율 유지
            scale = min(
                target_width / self.original_width, target_height / self.original_height
            )
            if not allow_upscale:
                scale = min(scale, 1.0)
            new_size = (
                int(self.original_width * scale),
                int(self.original_height * scale),
            )
            resized = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            return np.array(resized)

        elif mode == ResizeMode.FIT_WIDTH:
            # 폭맞춤
            scale = target_width / self.original_width
            if not allow_upscale:
                scale = min(scale, 1.0)
            new_size = (
                int(self.original_width * scale),
                int(self.original_height * scale),
            )
            resized = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            return np.array(resized)

        elif mode == ResizeMode.FIT_HEIGHT:
            # 높이맞춤
            scale = target_height / self.original_height
            if not allow_upscale:
                scale = min(scale, 1.0)
            new_size = (
                int(self.original_width * scale),
                int(self.original_height * scale),
            )
            resized = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            return np.array(resized)

        elif mode == ResizeMode.ADD_PADDING:
            # 여백 붙이기
            scale = min(
                target_width / self.original_width, target_height / self.original_height
            )
            if not allow_upscale:
                scale = min(scale, 1.0)
            new_size = (
                int(self.original_width * scale),
                int(self.original_height * scale),
            )
            resized = pil_image.resize(new_size, Image.Resampling.LANCZOS)

            # 검은 배경에 중앙 배치
            result = Image.new("RGB", (target_width, target_height), (0, 0, 0))
            x = (target_width - new_size[0]) // 2
            y = (target_height - new_size[1]) // 2
            result.paste(resized, (x, y))
            return np.array(result)

        elif mode == ResizeMode.CROP_TO_FIT:
            # 여백 자르기
            scale = max(
                target_width / self.original_width, target_height / self.original_height
            )
            if not allow_upscale:
                scale = min(scale, 1.0)
            new_size = (
                int(self.original_width * scale),
                int(self.original_height * scale),
            )
            resized = pil_image.resize(new_size, Image.Resampling.LANCZOS)

            # 중앙에서 자르기
            x = (new_size[0] - target_width) // 2
            y = (new_size[1] - target_height) // 2
            cropped = resized.crop((x, y, x + target_width, y + target_height))
            return np.array(cropped)

        elif mode == ResizeMode.STRETCH:
            # 꽉차게 늘리기
            resized = pil_image.resize(
                (target_width, target_height), Image.Resampling.LANCZOS
            )
            return np.array(resized)

        return self.original_image.copy()

    def _on_ok(self):
        """확인 버튼"""
        mode = self.mode_combo.currentData()

        if mode == ResizeMode.NO_RESIZE:
            self.reject()
            return

        result = self._apply_resize()
        if result is not None:
            mode_name = self.mode_combo.currentText()
            result_w, result_h = result.shape[1], result.shape[0]
            description = f"{tr('resize.title')}: {mode_name} ({result_w}x{result_h})"
            self.resize_applied.emit(result, description)
            self.accept()
