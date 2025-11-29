# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 빌드 설정 파일

사용법:
    pyinstaller build_config.spec
"""

import sys
from pathlib import Path

block_cipher = None

# 프로젝트 루트
project_root = Path.cwd()

# ReShade 필터 모듈 자동 수집
reshade_modules = []
reshade_dir = project_root / "filters" / "reshade"

if reshade_dir.exists():
    for py_file in reshade_dir.glob("*.py"):
        if py_file.stem not in ["__init__", "hlsl_helpers"]:
            module_name = f"filters.reshade.{py_file.stem}"
            reshade_modules.append(module_name)

# Hidden imports (PyInstaller가 자동으로 찾지 못하는 모듈)
hidden_imports = [
    "filters.reshade",
    "filters.reshade.hlsl_helpers",
] + reshade_modules

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=[
        ("assets", "assets"),  # 리소스 파일
        ("config", "config"),  # 설정 파일
    ],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="ImageEditor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI 모드
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="assets/app.ico" if (project_root / "assets" / "app.ico").exists() else None,
)
