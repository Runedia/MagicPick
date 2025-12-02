"""
MagicPick 백그라운드 서비스 컴포넌트
"""

from .singleton import SingletonGuard
from .tray_service import TrayService

__all__ = ["SingletonGuard", "TrayService"]
