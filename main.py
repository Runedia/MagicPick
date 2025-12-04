import os
import sys

from PyQt5.QtGui import QFont, QFontDatabase, QIcon
from PyQt5.QtWidgets import QApplication

from services import SingletonGuard, TrayService

# 표준 입출력 인코딩을 utf-8로 강제 설정
os.environ["PYTHONIOENCODING"] = "utf-8"

# fmt: off
from rich.traceback import install

install(show_locals=True)  # 변수 값 표시 옵션 켜기
# fmt: on


def main():
    # 1단계: QApplication 생성 전 싱글톤 체크
    singleton = SingletonGuard()
    if singleton.is_already_running():
        # 다른 인스턴스 실행 중 - 경고 표시 후 종료
        # app = QApplication(sys.argv)
        # QMessageBox.warning(None, "MagicPick", "MagicPick이 이미 실행 중입니다.\n시스템 트레이를 확인하세요.", )
        return

    # 2단계: QApplication 생성
    app = QApplication(sys.argv)
    app.setApplicationName("MagicPick")
    app.setOrganizationName("MagicPick")
    app.setWindowIcon(QIcon("assets/logo.png"))

    # 핵심: 마지막 창이 닫혀도 앱 종료 안함
    app.setQuitOnLastWindowClosed(False)

    # 3단계: 폰트 로드 및 설정
    # Qt 리소스 파일에서 D2Coding 폰트 로드
    font_id = QFontDatabase.addApplicationFont(":/fonts/D2Coding.ttf")
    if font_id != -1:
        # 폰트 로드 성공 시 패밀리 이름 가져오기
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            # 로드된 폰트 패밀리로 애플리케이션 폰트 설정
            font = QFont(families[0], 10)
            app.setFont(font)
    else:
        # 폰트 로드 실패 시 시스템 폰트 사용
        print("Warning: D2Coding 폰트 로드 실패, 기본 폰트 사용")
        font = QFont("Malgun Gothic", 10)
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
