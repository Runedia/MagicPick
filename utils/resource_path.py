"""
리소스 경로 헬퍼

PyInstaller 빌드된 exe에서도 리소스 파일을 올바르게 찾을 수 있도록 지원합니다.
"""

import os
import sys


def get_resource_path(relative_path: str) -> str:
    """
    읽기 전용 리소스 파일의 절대 경로를 반환합니다.

    PyInstaller로 빌드된 exe에서는 임시 폴더(_MEIPASS)를 기준으로,
    개발 환경에서는 프로젝트 루트를 기준으로 경로를 계산합니다.

    사용 예: 아이콘, 폰트 등 변경되지 않는 리소스

    Args:
        relative_path: 프로젝트 루트 기준 상대 경로 (예: "assets/logo.png")

    Returns:
        리소스 파일의 절대 경로
    """
    # PyInstaller가 생성한 임시 폴더 확인
    if hasattr(sys, "_MEIPASS"):
        # PyInstaller로 빌드된 exe 실행 중
        base_path = sys._MEIPASS
    else:
        # 개발 환경 - 스크립트 위치 기준으로 프로젝트 루트 찾기
        # main.py가 프로젝트 루트에 있다고 가정
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    return os.path.join(base_path, relative_path)


def get_user_data_path(relative_path: str) -> str:
    """
    사용자 데이터 파일의 절대 경로를 반환합니다.

    PyInstaller로 빌드된 exe에서는 exe가 있는 폴더를 기준으로,
    개발 환경에서는 프로젝트 루트를 기준으로 경로를 계산합니다.

    사용 예: 프리셋, 설정 파일 등 사용자가 저장/수정하는 데이터

    Args:
        relative_path: 프로젝트 루트 기준 상대 경로 (예: "config/reshade_presets")

    Returns:
        사용자 데이터 파일의 절대 경로
    """
    if hasattr(sys, "_MEIPASS"):
        # PyInstaller로 빌드된 exe 실행 중
        # sys.executable = exe 파일의 전체 경로
        base_path = os.path.dirname(sys.executable)
    else:
        # 개발 환경 - 프로젝트 루트
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    return os.path.join(base_path, relative_path)
