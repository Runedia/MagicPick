import os
import sys

# 표준 입출력 인코딩을 utf-8로 강제 설정
os.environ["PYTHONIOENCODING"] = "utf-8"

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtGui import QFont, QIcon

from services import TrayService, SingletonGuard


def main():
    # 1단계: QApplication 생성 전 싱글톤 체크
    singleton = SingletonGuard()
    if singleton.is_already_running():
        # 다른 인스턴스 실행 중 - 경고 표시 후 종료
        app = QApplication(sys.argv)
        QMessageBox.warning(
            None, "MagicPick", "MagicPick이 이미 실행 중입니다.\n시스템 트레이를 확인하세요."
        )
        return

    # 2단계: QApplication 생성
    app = QApplication(sys.argv)
    app.setApplicationName("MagicPick")
    app.setOrganizationName("MagicPick")
    app.setWindowIcon(QIcon("assets/logo.png"))

    # 핵심: 마지막 창이 닫혀도 앱 종료 안함
    app.setQuitOnLastWindowClosed(False)

    # 3단계: 폰트 설정
    font = QFont("D2Coding", 10)
    app.setFont(font)

    # 4단계: 트레이 서비스 생성 (메인 프로세스)
    tray_service = TrayService(app)

    # GC 방지를 위해 참조 유지
    app._singleton = singleton
    app._tray_service = tray_service

    # 5단계: Qt 이벤트 루프 시작
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
