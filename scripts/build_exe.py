"""
PyInstaller 빌드 자동화 스크립트

사용법:
    python scripts/build_exe.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).parent.parent
    spec_file = project_root / "build_config.spec"

    print("=" * 80)
    print("PyInstaller 빌드 시작")
    print("=" * 80)

    if not spec_file.exists():
        print(f"오류: {spec_file} 파일을 찾을 수 없습니다.")
        sys.exit(1)

    # PyInstaller 실행
    cmd = ["pyinstaller", "--clean", str(spec_file)]

    print(f"\n실행 명령어: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, cwd=project_root, check=True)

        print("\n" + "=" * 80)
        print("빌드 완료!")
        print("=" * 80)
        print(f"\n실행 파일 위치: {project_root / 'dist' / 'ImageEditor.exe'}")

    except subprocess.CalledProcessError as e:
        print(f"\n빌드 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
