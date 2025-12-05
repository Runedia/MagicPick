import numpy as np
from PIL import Image
from PyQt5.QtCore import QRect, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QScrollArea

from config.translations import tr


class ImageViewer(QScrollArea):
    zoom_changed = pyqtSignal(float)  # 배율 변경 시그널
    crop_confirmed = pyqtSignal(tuple)  # 자르기 확정 (x, y, w, h)
    crop_cancelled = pyqtSignal()  # 자르기 취소
    crop_size_changed = pyqtSignal(int, int)  # 자르기 영역 크기 변경 (w, h)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.current_image = None
        self.zoom_factor = 1.0  # 확대/축소 배율

        # 패닝 관련 변수
        self.panning = False
        self.pan_start_pos = None

        # 자르기 모드
        self.crop_mode = False
        self.crop_overlay = None

        self.init_ui()

    def init_ui(self):
        self.setWidgetResizable(True)
        self.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color: #2b2b2b; }")
        self.image_label.setText(tr("status.no_image"))
        self.image_label.setStyleSheet(
            "QLabel { background-color: #2b2b2b; color: #888; font-size: 14pt; }"
        )

        self.setWidget(self.image_label)

        # 자르기 오버레이 초기화
        self._init_crop_overlay()

    def _init_crop_overlay(self):
        """자르기 오버레이 초기화"""
        from ui.widgets.crop_overlay import CropOverlay

        self.crop_overlay = CropOverlay(self.viewport())
        self.crop_overlay.crop_confirmed.connect(self._on_crop_confirmed)
        self.crop_overlay.crop_cancelled.connect(self._on_crop_cancelled)
        self.crop_overlay.crop_changed.connect(self._on_crop_changed)
        self.crop_overlay.hide()

    def set_image(self, image):
        if isinstance(image, Image.Image):
            self.current_image = image
            self.display_image()
        elif isinstance(image, np.ndarray):
            self.current_image = Image.fromarray(image)
            self.display_image()
        elif isinstance(image, str):
            try:
                self.current_image = Image.open(image)
                self.display_image()
            except Exception as e:
                print(f"이미지 열기 실패: {e}")

    def display_image(self):
        if self.current_image is None:
            return

        img = self.current_image.copy()

        viewer_width = self.viewport().width()
        viewer_height = self.viewport().height()

        img_width, img_height = img.size

        # 기본 맞춤 스케일 계산
        fit_scale = min(viewer_width / img_width, viewer_height / img_height, 1.0)

        # zoom_factor 적용
        final_scale = fit_scale * self.zoom_factor

        if final_scale != 1.0:
            new_width = int(img_width * final_scale * 0.95)
            new_height = int(img_height * final_scale * 0.95)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        if img.mode == "RGB":
            data = img.tobytes("raw", "RGB")
            qimage = QImage(
                data, img.width, img.height, img.width * 3, QImage.Format_RGB888
            )
        elif img.mode == "RGBA":
            data = img.tobytes("raw", "RGBA")
            qimage = QImage(
                data, img.width, img.height, img.width * 4, QImage.Format_RGBA8888
            )
        else:
            img = img.convert("RGB")
            data = img.tobytes("raw", "RGB")
            qimage = QImage(
                data, img.width, img.height, img.width * 3, QImage.Format_RGB888
            )

        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap)
        self.image_label.setStyleSheet("")

        # zoom_changed 시그널 발생
        if hasattr(self, "zoom_changed"):
            self.zoom_changed.emit(self.zoom_factor)

    def get_image(self):
        return self.current_image

    def clear_image(self):
        self.current_image = None
        self.image_label.clear()
        self.image_label.setText(tr("status.no_image"))
        self.image_label.setStyleSheet(
            "QLabel { background-color: #2b2b2b; color: #888; font-size: 14pt; }"
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_image:
            self.display_image()

        # 자르기 오버레이 크기 조정
        if self.crop_overlay:
            self.crop_overlay.setGeometry(self.viewport().rect())

    def wheelEvent(self, event):
        """CTRL+휠로 확대/축소"""
        # 자르기 모드에서는 줌 비활성화
        if self.crop_mode:
            event.accept()
            return

        if event.modifiers() == Qt.ControlModifier:
            # 휠 델타값 (보통 ±120)
            delta = event.angleDelta().y()

            if delta > 0:
                # 확대 (10% 증가)
                self.set_zoom(self.zoom_factor * 1.1)
            elif delta < 0:
                # 축소 (10% 감소)
                self.set_zoom(self.zoom_factor / 1.1)

            event.accept()
        else:
            super().wheelEvent(event)

    def set_zoom(self, factor):
        """배율 설정 (0.1 ~ 5.0 범위)"""
        self.zoom_factor = max(0.1, min(5.0, factor))
        if self.current_image:
            self.display_image()

    def reset_zoom(self):
        """배율 초기화 (100%)"""
        self.set_zoom(1.0)

    def get_zoom(self):
        """현재 배율 반환"""
        return self.zoom_factor

    def mousePressEvent(self, event):
        """마우스 누름 - 패닝 시작"""
        # 자르기 모드에서는 패닝 비활성화
        if self.crop_mode:
            event.ignore()
            return

        if event.button() == Qt.LeftButton:
            self.panning = True
            self.pan_start_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """마우스 이동 - 패닝 중"""
        # 자르기 모드에서는 패닝 비활성화
        if self.crop_mode:
            event.ignore()
            return

        if self.panning and self.pan_start_pos:
            # 이동 거리 계산
            delta = event.pos() - self.pan_start_pos
            self.pan_start_pos = event.pos()

            # 스크롤바 이동
            h_bar = self.horizontalScrollBar()
            v_bar = self.verticalScrollBar()

            h_bar.setValue(h_bar.value() - delta.x())
            v_bar.setValue(v_bar.value() - delta.y())

            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """마우스 놓음 - 패닝 종료"""
        # 자르기 모드에서는 패닝 비활성화
        if self.crop_mode:
            event.ignore()
            return

        if event.button() == Qt.LeftButton:
            self.panning = False
            self.pan_start_pos = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    # ===== 자르기 모드 메서드 =====

    def start_crop_mode(self):
        """자르기 모드 시작"""
        if self.current_image is None:
            return False

        self.crop_mode = True

        # 이미지가 표시되는 영역 계산
        image_rect = self._get_image_display_rect()
        image_size = self.current_image.size  # (width, height)

        # 오버레이 크기 설정 및 시작
        self.crop_overlay.setGeometry(self.viewport().rect())
        self.crop_overlay.start_crop(image_rect, image_size)
        self.crop_overlay.setFocus()

        return True

    def stop_crop_mode(self):
        """자르기 모드 종료"""
        self.crop_mode = False
        if self.crop_overlay:
            self.crop_overlay.stop_crop()

    def confirm_crop(self):
        """자르기 확정"""
        if self.crop_overlay:
            self.crop_overlay.confirm_crop()

    def cancel_crop(self):
        """자르기 취소"""
        if self.crop_overlay:
            self.crop_overlay.cancel_crop()

    def reset_crop(self):
        """자르기 영역 초기화"""
        if self.crop_overlay:
            self.crop_overlay.reset_crop()

    def is_crop_mode(self):
        """자르기 모드 여부"""
        return self.crop_mode

    def _get_image_display_rect(self) -> QRect:
        """이미지가 실제로 표시되는 영역 계산 (뷰포트 좌표)"""
        if self.current_image is None:
            return QRect()

        # 현재 표시된 pixmap 크기
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            return QRect()

        # 이미지 라벨 위치 (스크롤 영역 내)
        label_pos = self.image_label.pos()

        # 뷰포트 내에서의 이미지 위치 계산
        viewport_rect = self.viewport().rect()

        # 이미지 라벨은 중앙 정렬됨
        pixmap_size = pixmap.size()

        # 라벨 내에서 pixmap 위치 (중앙 정렬)
        label_size = self.image_label.size()
        offset_x = (label_size.width() - pixmap_size.width()) // 2
        offset_y = (label_size.height() - pixmap_size.height()) // 2

        # 스크롤 위치 고려
        scroll_x = self.horizontalScrollBar().value()
        scroll_y = self.verticalScrollBar().value()

        # 최종 뷰포트 내 위치
        x = label_pos.x() + offset_x - scroll_x
        y = label_pos.y() + offset_y - scroll_y

        return QRect(x, y, pixmap_size.width(), pixmap_size.height())

    def _on_crop_confirmed(self, crop_rect: tuple):
        """자르기 확정 시그널 처리"""
        self.crop_mode = False
        self.crop_confirmed.emit(crop_rect)

    def _on_crop_cancelled(self):
        """자르기 취소 시그널 처리"""
        self.crop_mode = False
        self.crop_cancelled.emit()

    def _on_crop_changed(self, width: int, height: int):
        """자르기 영역 크기 변경 시그널 처리"""
        self.crop_size_changed.emit(width, height)
