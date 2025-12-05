"""
시스템 트레이 서비스

MagicPick의 메인 백그라운드 서비스입니다.
시스템 트레이 아이콘, 전역 단축키, MainWindow 생명주기를 관리합니다.
"""

from PyQt5.QtCore import QObject
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction, QMenu, QSystemTrayIcon

from utils.global_hotkey import GlobalHotkeyManager


class TrayService(QObject):
    """
    MagicPick 시스템 트레이 서비스.

    시스템 트레이 아이콘 관리, 전역 단축키 처리, MainWindow 생명주기 관리를
    담당하는 메인 백그라운드 서비스입니다.

    MainWindow는 필요할 때만 생성되며 (lazy creation),
    닫힌 후에도 인스턴스를 유지하여 히스토리와 상태를 보존합니다.
    """

    def __init__(self, app):
        """
        TrayService 초기화

        Args:
            app: QApplication 인스턴스
        """
        super().__init__()
        self.app = app
        self._main_window = None  # MainWindow 인스턴스 (lazy creation)
        self._default_timer_delay = 0  # 창이 닫혔을 때의 타이머 설정 저장

        # 시스템 트레이 아이콘 생성
        self.tray_icon = QSystemTrayIcon(self)

        # 전역 단축키 관리자 생성
        self.hotkey_manager = GlobalHotkeyManager()

        # 초기화
        self.setup_tray_icon()
        self.connect_hotkey_signals()
        self.hotkey_manager.register_hotkeys()

        # 트레이 아이콘 표시
        self.tray_icon.show()

    def setup_tray_icon(self):
        """시스템 트레이 아이콘 및 컨텍스트 메뉴를 설정합니다."""
        from config.translations import tr
        from utils.resource_path import get_resource_path

        # 아이콘 설정
        icon = QIcon(get_resource_path("assets/logo.png"))
        self.tray_icon.setIcon(icon)
        self.tray_icon.setToolTip("MagicPick - 화면 캡처 및 이미지 편집")

        # 컨텍스트 메뉴 생성
        menu = QMenu()

        # 메뉴 항목: 창 열기
        open_action = QAction(tr("tray.show_main_window"), self)
        open_action.triggered.connect(self.show_main_window)
        menu.addAction(open_action)

        # 구분선
        menu.addSeparator()

        # 메뉴 항목: 종료
        quit_action = QAction(tr("tray.quit"), self)
        quit_action.triggered.connect(self.quit_application)
        menu.addAction(quit_action)

        self.tray_icon.setContextMenu(menu)

        # 더블 클릭 시 MainWindow 열기
        self.tray_icon.activated.connect(self.on_tray_activated)

    def on_tray_activated(self, reason):
        """
        트레이 아이콘 활성화 이벤트 처리.

        Args:
            reason: QSystemTrayIcon.ActivationReason
        """
        if reason == QSystemTrayIcon.DoubleClick:
            self.show_main_window()

    def connect_hotkey_signals(self):
        """전역 단축키 시그널을 캡처 핸들러에 연결합니다."""
        self.hotkey_manager.fullscreen_pressed.connect(self.handle_fullscreen_hotkey)
        self.hotkey_manager.region_pressed.connect(self.handle_region_hotkey)
        self.hotkey_manager.window_pressed.connect(self.handle_window_hotkey)
        self.hotkey_manager.monitor_pressed.connect(self.handle_monitor_hotkey)

    def get_or_create_main_window(self):
        """
        MainWindow 인스턴스를 가져오거나 생성합니다.

        첫 호출 시 MainWindow를 생성하고, 이후에는 동일한 인스턴스를 반환합니다.
        이를 통해 창을 닫았다가 다시 열어도 히스토리와 상태가 보존됩니다.

        Returns:
            MainWindow: 싱글톤 MainWindow 인스턴스
        """
        if self._main_window is None:
            from ui.main_window import MainWindow

            self._main_window = MainWindow()

            # 창 닫기 시그널 연결
            self._main_window.closing.connect(self.on_main_window_closing)

            # 캡처 완료/실패 시그널 연결 (디바운스 리셋용)
            self._main_window.screen_capture.capture_completed.connect(
                self.hotkey_manager.reset_capture_state
            )
            self._main_window.screen_capture.capture_failed.connect(
                self.hotkey_manager.reset_capture_state
            )

        return self._main_window

    def show_main_window(self):
        """MainWindow를 표시합니다 (필요시 생성)."""
        window = self.get_or_create_main_window()
        window.show()
        window.activateWindow()  # 포커스 이동
        window.raise_()  # 최상위로

    def on_main_window_closing(self):
        """
        MainWindow가 닫히기 직전 호출됩니다.

        현재 타이머 설정을 저장하여 다음 단축키 캡처에 사용할 수 있도록 합니다.
        """
        if self._main_window and self._main_window.toolbar:
            self._default_timer_delay = self._main_window.toolbar.get_timer_delay()

    def handle_fullscreen_hotkey(self):
        """Ctrl+Shift+F1 단축키 처리 - 전체 화면 캡처"""
        from capture import fullscreen

        window = self.get_or_create_main_window()

        # 자르기 모드/다이얼로그 정리
        window._prepare_for_capture()

        # 지연 시간 결정
        if window.isVisible():
            # 창이 열려있으면 toolbar 설정 사용
            delay = None
            current_delay = window.toolbar.get_timer_delay()
            if current_delay > 0:
                window.update_status(f"{current_delay}초 후 전체 화면 캡처 시작...")
            else:
                window.update_status("전체 화면 캡처 중...")
        else:
            # 창이 닫혀있으면 즉시 캡처
            delay = 0

        # 캡처 실행
        window.screen_capture.execute_capture(
            fullscreen.capture_fullscreen, external_delay=delay
        )

    def handle_region_hotkey(self):
        """Ctrl+Shift+F2 단축키 처리 - 영역 지정 캡처"""
        from capture import region

        window = self.get_or_create_main_window()

        # 자르기 모드/다이얼로그 정리
        window._prepare_for_capture()

        # 지연 시간 결정
        if window.isVisible():
            delay = None
            current_delay = window.toolbar.get_timer_delay()
            if current_delay > 0:
                window.update_status(f"{current_delay}초 후 영역 캡처 시작...")
            else:
                window.update_status("영역 캡처 중...")
        else:
            delay = 0

        # 캡처 실행
        window.screen_capture.execute_capture(
            region.capture_region, external_delay=delay
        )

    def handle_window_hotkey(self):
        """Ctrl+Shift+F3 단축키 처리 - 윈도우 캡처"""
        from capture import window as win_capture

        window = self.get_or_create_main_window()

        # 자르기 모드/다이얼로그 정리
        window._prepare_for_capture()

        # 지연 시간 결정
        if window.isVisible():
            delay = None
            current_delay = window.toolbar.get_timer_delay()
            if current_delay > 0:
                window.update_status(f"{current_delay}초 후 윈도우 캡처 시작...")
            else:
                window.update_status("윈도우 캡처 중...")
        else:
            delay = 0

        # 캡처 실행
        window.screen_capture.execute_capture(
            win_capture.capture_window, external_delay=delay
        )

    def handle_monitor_hotkey(self):
        """
        Ctrl+Shift+F4 단축키 처리 - 모니터 캡처

        모니터 캡처는 다이얼로그 선택이 필요하지만
        메인 윈도우는 표시하지 않습니다 (다이얼로그만 표시).
        """
        window = self.get_or_create_main_window()

        # 자르기 모드/다이얼로그 정리
        window._prepare_for_capture()

        # 모니터 캡처 메서드 호출 (다이얼로그만 표시, 메인 윈도우는 숨김)
        window.capture_monitor()

        # 디바운스 리셋
        self.hotkey_manager.reset_capture_state()

    def quit_application(self):
        """애플리케이션을 종료합니다."""
        # 단축키 등록 해제
        self.hotkey_manager.unregister_hotkeys()

        # 트레이 아이콘 숨김
        self.tray_icon.hide()

        # 애플리케이션 종료
        self.app.quit()
