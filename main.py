import os
import sys

# 표준 입출력 인코딩을 utf-8로 강제 설정
os.environ["PYTHONIOENCODING"] = "utf-8"

# 소스 코드 인코딩 선언 (Python 3에서는 기본이지만 명시 권장)
# -*- coding: utf-8 -*-

from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import QApplication

from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("MagicPick")
    app.setOrganizationName("MagicPick")
    app.setWindowIcon(QIcon("assets/logo.png"))

    font = QFont("D2Coding", 10)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
