"""
프로세스 간 싱글톤 구현

QSharedMemory를 사용하여 MagicPick 인스턴스가 하나만 실행되도록 보장합니다.
"""

from PyQt5.QtCore import QSharedMemory


class SingletonGuard:
    """
    단일 인스턴스 실행을 보장하는 싱글톤 가드.

    QSharedMemory를 사용하여 프로세스 간 싱글톤을 구현합니다.
    첫 번째 인스턴스는 shared memory를 생성하고,
    후속 인스턴스는 기존 shared memory를 감지하여 중복 실행을 방지합니다.
    """

    def __init__(self):
        """SingletonGuard 초기화"""
        self._shared_memory = QSharedMemory("MagicPickSingleInstance")

    def is_already_running(self):
        """
        다른 인스턴스가 이미 실행 중인지 확인합니다.

        Returns:
            bool: 다른 인스턴스가 실행 중이면 True,
                  이것이 첫 인스턴스면 False
        """
        # 기존 shared memory에 attach 시도
        if self._shared_memory.attach():
            # 이미 존재 - 다른 인스턴스가 실행 중
            return True

        # 새로운 shared memory 생성 시도 (1 byte면 충분)
        if self._shared_memory.create(1):
            # 생성 성공 - 이것이 첫 인스턴스
            return False

        # 생성 실패 (정상적으로는 발생하지 않음)
        return False

    def __del__(self):
        """소멸자 - shared memory 정리"""
        try:
            if self._shared_memory.isAttached():
                self._shared_memory.detach()
        except RuntimeError:
            # Qt 객체가 이미 삭제된 경우 무시
            # (애플리케이션 종료 시 Qt의 소멸 순서 문제)
            pass
