"""
히스토리 관리 모듈 (Undo/Redo 시스템)
"""

from typing import Optional

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal


class HistoryManager(QObject):
    """이미지 편집 히스토리를 관리하는 클래스"""

    # 시그널 정의
    state_changed = pyqtSignal()  # 히스토리 상태 변경 시 발생
    undo_available = pyqtSignal(bool)  # Undo 가능 여부
    redo_available = pyqtSignal(bool)  # Redo 가능 여부

    def __init__(self, max_history: int = 20):
        """
        Args:
            max_history: 최대 히스토리 개수
        """
        super().__init__()
        self.history = []
        self.current_index = -1
        self.max_history = max_history

    def add_state(self, image: np.ndarray, description: str = "") -> None:
        """
        새로운 상태를 히스토리에 추가

        Args:
            image: 저장할 이미지 (NumPy array)
            description: 작업 설명 (선택사항)
        """
        # 현재 위치 이후의 히스토리 삭제 (새 분기 시작)
        self.history = self.history[: self.current_index + 1]

        # 새 상태 추가 (이미지 복사본 저장)
        state = {"image": image.copy(), "description": description}
        self.history.append(state)
        self.current_index += 1

        # 최대 히스토리 초과 시 가장 오래된 것 삭제
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.current_index -= 1

        # 시그널 발생
        self._emit_state_signals()

    def undo(self) -> Optional[np.ndarray]:
        """
        이전 상태로 되돌리기

        Returns:
            이전 상태의 이미지, 없으면 None
        """
        if self.can_undo():
            self.current_index -= 1
            self._emit_state_signals()
            return self.history[self.current_index]["image"].copy()
        return None

    def redo(self) -> Optional[np.ndarray]:
        """
        다음 상태로 진행

        Returns:
            다음 상태의 이미지, 없으면 None
        """
        if self.can_redo():
            self.current_index += 1
            self._emit_state_signals()
            return self.history[self.current_index]["image"].copy()
        return None

    def can_undo(self) -> bool:
        """Undo 가능 여부 확인"""
        return self.current_index > 0

    def can_redo(self) -> bool:
        """Redo 가능 여부 확인"""
        return self.current_index < len(self.history) - 1

    def get_current_state(self) -> Optional[np.ndarray]:
        """
        현재 상태의 이미지 반환

        Returns:
            현재 이미지, 히스토리가 비어있으면 None
        """
        if self.current_index >= 0 and self.current_index < len(self.history):
            return self.history[self.current_index]["image"].copy()
        return None

    def get_current_description(self) -> str:
        """현재 상태의 설명 반환"""
        if self.current_index >= 0 and self.current_index < len(self.history):
            return self.history[self.current_index]["description"]
        return ""

    def get_undo_description(self) -> str:
        """Undo 시 돌아갈 상태의 설명 반환"""
        if self.can_undo():
            return self.history[self.current_index - 1]["description"]
        return ""

    def get_redo_description(self) -> str:
        """Redo 시 진행할 상태의 설명 반환"""
        if self.can_redo():
            return self.history[self.current_index + 1]["description"]
        return ""

    def clear(self) -> None:
        """히스토리 초기화"""
        self.history.clear()
        self.current_index = -1
        self._emit_state_signals()

    def get_history_size(self) -> int:
        """현재 히스토리 개수 반환"""
        return len(self.history)

    def get_current_index(self) -> int:
        """현재 히스토리 인덱스 반환"""
        return self.current_index

    def _emit_state_signals(self) -> None:
        """상태 변경 시그널 발생"""
        self.state_changed.emit()
        self.undo_available.emit(self.can_undo())
        self.redo_available.emit(self.can_redo())

    def get_memory_usage_mb(self) -> float:
        """
        현재 히스토리의 메모리 사용량 추정 (MB)

        Returns:
            메모리 사용량 (MB)
        """
        total_bytes = 0
        for state in self.history:
            image = state["image"]
            total_bytes += image.nbytes

        return total_bytes / (1024 * 1024)
