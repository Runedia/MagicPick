"""
ReShade 동적 필터 로딩 테스트 스크립트

사용법:
    python scripts/test_reshade_filters.py
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from filters.reshade_filters import (
    get_filter_class,
    list_available_filters,
)
from filters.reshade_parser import ReShadeParser


def main():
    print("=" * 80)
    print("ReShade 동적 필터 로딩 시스템 테스트")
    print("=" * 80)

    # 1. 사용 가능한 필터 목록
    print("\n[1] 사용 가능한 필터 목록:")
    print("-" * 80)
    available = list_available_filters()
    print(f"총 {len(available)}개 필터 발견:\n")

    for i, filter_name in enumerate(available, 1):
        filter_class = get_filter_class(filter_name)
        if filter_class:
            instance = filter_class()
            print(f"{i:3d}. {filter_name:40s} -> {instance.description}")

    # 2. ReShadeParser 통합 테스트
    print("\n" + "=" * 80)
    print("[2] ReShadeParser 지원 효과 확인:")
    print("-" * 80)
    supported = ReShadeParser.get_supported_effects()
    print(f"총 {len(supported)}개 효과 지원\n")

    # 3. 샘플 필터 인스턴스 생성 테스트
    print("=" * 80)
    print("[3] 샘플 필터 인스턴스 생성 테스트:")
    print("-" * 80)

    test_filters = [
        "AdaptiveSharpen",
        "Bloom",
        "Vibrance",
        "Clarity",
        "FakeHDR",
        "PD80_03_Levels",
    ]

    for filter_name in test_filters:
        filter_class = get_filter_class(filter_name)
        if filter_class:
            try:
                instance = filter_class()
                print(f"✓ {filter_name:30s} - 생성 성공")
            except Exception as e:
                print(f"✗ {filter_name:30s} - 생성 실패: {e}")
        else:
            print(f"? {filter_name:30s} - 클래스 없음")

    print("\n" + "=" * 80)
    print("테스트 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()
