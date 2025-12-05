"""
자르기 오버레이 위젯

이미지 뷰어 위에 표시되며, PowerPoint 스타일의 자르기 기능을 제공합니다.
8개의 핸들(코너 4개 + 변 4개)을 드래그하여 자르기 영역을 조정합니다.
"""

from PyQt5.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QCursor, QPainter, QPen
from PyQt5.QtWidgets import QWidget


class CropOverlay(QWidget):
    """이미지 뷰어 위에 표시되는 자르기 오버레이"""

    # 시그널
    crop_confirmed = pyqtSignal(tuple)  # (x, y, width, height) 이미지 좌표
    crop_cancelled = pyqtSignal()
    crop_changed = pyqtSignal(int, int)  # (width, height) - 실시간 크기 변경

    # 상수
    HANDLE_SIZE = 10
    MIN_CROP_SIZE = 20

    # 핸들 위치 상수
    HANDLE_TL = 0  # Top-Left
    HANDLE_T = 1  # Top
    HANDLE_TR = 2  # Top-Right
    HANDLE_R = 3  # Right
    HANDLE_BR = 4  # Bottom-Right
    HANDLE_B = 5  # Bottom
    HANDLE_BL = 6  # Bottom-Left
    HANDLE_L = 7  # Left

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

        # 이미지 정보
        self.image_rect = QRect()  # 위젯 내 이미지 표시 영역
        self.image_size = (0, 0)  # 원본 이미지 크기 (width, height)

        # 자르기 영역 (위젯 좌표)
        self.crop_rect = QRect()

        # 핸들 영역
        self.handles = [QRect() for _ in range(8)]

        # 드래그 상태
        self.dragging = False
        self.drag_handle = -1
        self.drag_start = QPoint()
        self.drag_rect_start = QRect()

        # 초기에는 숨김
        self.hide()

    def start_crop(self, image_rect: QRect, image_size: tuple):
        """
        자르기 모드 시작

        Args:
            image_rect: 위젯 내 이미지가 표시되는 영역 (위젯 좌표)
            image_size: 원본 이미지 크기 (width, height)
        """
        self.image_rect = image_rect
        self.image_size = image_size

        # 초기 자르기 영역 = 전체 이미지 영역
        self.crop_rect = QRect(image_rect)

        self._update_handles()
        self.show()
        self.raise_()

        # 초기 크기 알림
        w, h = self._get_crop_size_in_image()
        self.crop_changed.emit(w, h)

    def stop_crop(self):
        """자르기 모드 종료"""
        self.hide()
        self.dragging = False
        self.drag_handle = -1

    def reset_crop(self):
        """자르기 영역 초기화"""
        self.crop_rect = QRect(self.image_rect)
        self._update_handles()
        self.update()

        w, h = self._get_crop_size_in_image()
        self.crop_changed.emit(w, h)

    def confirm_crop(self):
        """자르기 확정"""
        # 위젯 좌표를 이미지 좌표로 변환
        x, y, w, h = self._widget_to_image_rect()
        self.crop_confirmed.emit((x, y, w, h))
        self.stop_crop()

    def cancel_crop(self):
        """자르기 취소"""
        self.crop_cancelled.emit()
        self.stop_crop()

    def _get_crop_size_in_image(self) -> tuple:
        """현재 자르기 영역의 이미지 좌표 크기 반환"""
        if self.image_rect.width() == 0 or self.image_rect.height() == 0:
            return (0, 0)

        scale_x = self.image_size[0] / self.image_rect.width()
        scale_y = self.image_size[1] / self.image_rect.height()

        w = int(self.crop_rect.width() * scale_x)
        h = int(self.crop_rect.height() * scale_y)

        return (w, h)

    def _widget_to_image_rect(self) -> tuple:
        """위젯 좌표의 자르기 영역을 이미지 좌표로 변환"""
        if self.image_rect.width() == 0 or self.image_rect.height() == 0:
            return (0, 0, self.image_size[0], self.image_size[1])

        scale_x = self.image_size[0] / self.image_rect.width()
        scale_y = self.image_size[1] / self.image_rect.height()

        x = int((self.crop_rect.x() - self.image_rect.x()) * scale_x)
        y = int((self.crop_rect.y() - self.image_rect.y()) * scale_y)
        w = int(self.crop_rect.width() * scale_x)
        h = int(self.crop_rect.height() * scale_y)

        # 범위 제한
        x = max(0, min(x, self.image_size[0]))
        y = max(0, min(y, self.image_size[1]))
        w = max(1, min(w, self.image_size[0] - x))
        h = max(1, min(h, self.image_size[1] - y))

        return (x, y, w, h)

    def _update_handles(self):
        """핸들 위치 업데이트"""
        r = self.crop_rect
        hs = self.HANDLE_SIZE
        half = hs // 2

        # 코너 핸들
        self.handles[self.HANDLE_TL] = QRect(r.left() - half, r.top() - half, hs, hs)
        self.handles[self.HANDLE_TR] = QRect(r.right() - half, r.top() - half, hs, hs)
        self.handles[self.HANDLE_BL] = QRect(r.left() - half, r.bottom() - half, hs, hs)
        self.handles[self.HANDLE_BR] = QRect(
            r.right() - half, r.bottom() - half, hs, hs
        )

        # 변 핸들
        cx = r.center().x()
        cy = r.center().y()
        self.handles[self.HANDLE_T] = QRect(cx - half, r.top() - half, hs, hs)
        self.handles[self.HANDLE_B] = QRect(cx - half, r.bottom() - half, hs, hs)
        self.handles[self.HANDLE_L] = QRect(r.left() - half, cy - half, hs, hs)
        self.handles[self.HANDLE_R] = QRect(r.right() - half, cy - half, hs, hs)

    def _get_handle_at(self, pos: QPoint) -> int:
        """주어진 위치의 핸들 인덱스 반환 (-1이면 없음)"""
        for i, handle in enumerate(self.handles):
            if handle.contains(pos):
                return i
        return -1

    def _get_cursor_for_handle(self, handle_idx: int) -> QCursor:
        """핸들에 맞는 커서 반환"""
        cursors = {
            self.HANDLE_TL: Qt.SizeFDiagCursor,
            self.HANDLE_TR: Qt.SizeBDiagCursor,
            self.HANDLE_BL: Qt.SizeBDiagCursor,
            self.HANDLE_BR: Qt.SizeFDiagCursor,
            self.HANDLE_T: Qt.SizeVerCursor,
            self.HANDLE_B: Qt.SizeVerCursor,
            self.HANDLE_L: Qt.SizeHorCursor,
            self.HANDLE_R: Qt.SizeHorCursor,
        }
        return QCursor(cursors.get(handle_idx, Qt.ArrowCursor))

    def paintEvent(self, event):
        """그리기"""
        if not self.isVisible():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 1. 자르기 영역 외부 어둡게
        overlay_color = QColor(0, 0, 0, 150)

        # 전체 위젯 영역
        full_rect = self.rect()

        # 상단 영역
        if self.crop_rect.top() > full_rect.top():
            painter.fillRect(
                QRect(
                    full_rect.left(),
                    full_rect.top(),
                    full_rect.width(),
                    self.crop_rect.top() - full_rect.top(),
                ),
                overlay_color,
            )

        # 하단 영역
        if self.crop_rect.bottom() < full_rect.bottom():
            painter.fillRect(
                QRect(
                    full_rect.left(),
                    self.crop_rect.bottom(),
                    full_rect.width(),
                    full_rect.bottom() - self.crop_rect.bottom(),
                ),
                overlay_color,
            )

        # 좌측 영역
        painter.fillRect(
            QRect(
                full_rect.left(),
                self.crop_rect.top(),
                self.crop_rect.left() - full_rect.left(),
                self.crop_rect.height(),
            ),
            overlay_color,
        )

        # 우측 영역
        painter.fillRect(
            QRect(
                self.crop_rect.right(),
                self.crop_rect.top(),
                full_rect.right() - self.crop_rect.right(),
                self.crop_rect.height(),
            ),
            overlay_color,
        )

        # 2. 자르기 영역 테두리
        painter.setPen(QPen(Qt.white, 2, Qt.SolidLine))
        painter.drawRect(self.crop_rect)

        # 3. 3등분 가이드라인
        painter.setPen(QPen(QColor(255, 255, 255, 100), 1, Qt.DashLine))

        third_w = self.crop_rect.width() / 3
        for i in range(1, 3):
            x = self.crop_rect.left() + int(third_w * i)
            painter.drawLine(x, self.crop_rect.top(), x, self.crop_rect.bottom())

        third_h = self.crop_rect.height() / 3
        for i in range(1, 3):
            y = self.crop_rect.top() + int(third_h * i)
            painter.drawLine(self.crop_rect.left(), y, self.crop_rect.right(), y)

        # 4. 핸들 그리기
        self._update_handles()
        painter.setPen(QPen(Qt.white, 1))
        painter.setBrush(QColor(0, 120, 215))

        for handle in self.handles:
            painter.drawRect(handle)

    def mousePressEvent(self, event):
        """마우스 클릭"""
        if event.button() == Qt.LeftButton:
            handle = self._get_handle_at(event.pos())
            if handle >= 0:
                self.dragging = True
                self.drag_handle = handle
                self.drag_start = event.pos()
                self.drag_rect_start = QRect(self.crop_rect)
                event.accept()
                return

        event.ignore()

    def mouseMoveEvent(self, event):
        """마우스 이동"""
        if self.dragging and self.drag_handle >= 0:
            delta = event.pos() - self.drag_start
            self._apply_handle_drag(delta)
            self.update()

            w, h = self._get_crop_size_in_image()
            self.crop_changed.emit(w, h)
            event.accept()
        else:
            # 커서 변경
            handle = self._get_handle_at(event.pos())
            if handle >= 0:
                self.setCursor(self._get_cursor_for_handle(handle))
            else:
                self.setCursor(Qt.ArrowCursor)
            event.ignore()

    def mouseReleaseEvent(self, event):
        """마우스 릴리즈"""
        if event.button() == Qt.LeftButton and self.dragging:
            self.dragging = False
            self.drag_handle = -1
            event.accept()
        else:
            event.ignore()

    def _apply_handle_drag(self, delta: QPoint):
        """핸들 드래그 적용"""
        h = self.drag_handle
        new_rect = QRect(self.drag_rect_start)

        # 핸들 타입에 따라 영역 조정
        if h == self.HANDLE_TL:
            new_rect.setLeft(new_rect.left() + delta.x())
            new_rect.setTop(new_rect.top() + delta.y())
        elif h == self.HANDLE_T:
            new_rect.setTop(new_rect.top() + delta.y())
        elif h == self.HANDLE_TR:
            new_rect.setRight(new_rect.right() + delta.x())
            new_rect.setTop(new_rect.top() + delta.y())
        elif h == self.HANDLE_R:
            new_rect.setRight(new_rect.right() + delta.x())
        elif h == self.HANDLE_BR:
            new_rect.setRight(new_rect.right() + delta.x())
            new_rect.setBottom(new_rect.bottom() + delta.y())
        elif h == self.HANDLE_B:
            new_rect.setBottom(new_rect.bottom() + delta.y())
        elif h == self.HANDLE_BL:
            new_rect.setLeft(new_rect.left() + delta.x())
            new_rect.setBottom(new_rect.bottom() + delta.y())
        elif h == self.HANDLE_L:
            new_rect.setLeft(new_rect.left() + delta.x())

        # 정규화
        new_rect = new_rect.normalized()

        # 최소 크기 확인
        if new_rect.width() < self.MIN_CROP_SIZE:
            if h in (self.HANDLE_L, self.HANDLE_TL, self.HANDLE_BL):
                new_rect.setLeft(new_rect.right() - self.MIN_CROP_SIZE)
            else:
                new_rect.setRight(new_rect.left() + self.MIN_CROP_SIZE)

        if new_rect.height() < self.MIN_CROP_SIZE:
            if h in (self.HANDLE_T, self.HANDLE_TL, self.HANDLE_TR):
                new_rect.setTop(new_rect.bottom() - self.MIN_CROP_SIZE)
            else:
                new_rect.setBottom(new_rect.top() + self.MIN_CROP_SIZE)

        # 이미지 영역 내로 제한
        new_rect.setLeft(max(self.image_rect.left(), new_rect.left()))
        new_rect.setTop(max(self.image_rect.top(), new_rect.top()))
        new_rect.setRight(min(self.image_rect.right(), new_rect.right()))
        new_rect.setBottom(min(self.image_rect.bottom(), new_rect.bottom()))

        self.crop_rect = new_rect

    def keyPressEvent(self, event):
        """키 입력"""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.confirm_crop()
        elif event.key() == Qt.Key_Escape:
            self.cancel_crop()
        else:
            event.ignore()
