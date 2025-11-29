from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QWidget,
)


class ZoomControl(QWidget):
    """상태바 우측 배율 컨트롤 위젯"""

    zoom_changed = pyqtSignal(float)  # 배율 변경 시그널

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_zoom = 1.0
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 0, 5, 0)
        layout.setSpacing(3)  # 간격 5 -> 3으로 축소

        # - 버튼
        self.zoom_out_btn = QPushButton("-")
        self.zoom_out_btn.setFixedSize(25, 25)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        layout.addWidget(self.zoom_out_btn)

        # 슬라이더 (10% ~ 500%)
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(10)  # 10%
        self.zoom_slider.setMaximum(500)  # 500%
        self.zoom_slider.setValue(100)  # 100%
        self.zoom_slider.setFixedWidth(100)  # 120 -> 100으로 축소
        self.zoom_slider.valueChanged.connect(self.on_slider_changed)
        layout.addWidget(self.zoom_slider)

        # + 버튼
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setFixedSize(25, 25)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        layout.addWidget(self.zoom_in_btn)

        # 100% 레이블 (클릭하면 초기화)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(45)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_label.setStyleSheet(
            """
            QLabel {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 3px;
                padding: 2px;
            }
            QLabel:hover {
                background-color: #e5f3ff;
                border: 1px solid #0078d4;
                cursor: pointer;
            }
            """
        )
        self.zoom_label.mousePressEvent = self.reset_zoom
        layout.addWidget(self.zoom_label)

        self.setLayout(layout)

        # 위젯 크기를 콘텐츠에 맞게 고정
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def zoom_in(self):
        """확대 (10% 증가)"""
        new_value = int(self.zoom_slider.value() * 1.1)
        self.zoom_slider.setValue(min(500, new_value))

    def zoom_out(self):
        """축소 (10% 감소)"""
        new_value = int(self.zoom_slider.value() / 1.1)
        self.zoom_slider.setValue(max(10, new_value))

    def reset_zoom(self, event=None):
        """배율 초기화 (100%)"""
        self.zoom_slider.setValue(100)

    def on_slider_changed(self, value):
        """슬라이더 값 변경 시"""
        zoom_factor = value / 100.0
        self.current_zoom = zoom_factor
        self.zoom_label.setText(f"{value}%")
        self.zoom_changed.emit(zoom_factor)

    def set_zoom(self, factor):
        """외부에서 배율 설정"""
        value = int(factor * 100)
        self.zoom_slider.blockSignals(True)  # 시그널 중복 방지
        self.zoom_slider.setValue(max(10, min(500, value)))
        self.zoom_slider.blockSignals(False)
        self.zoom_label.setText(f"{value}%")
        self.current_zoom = factor
