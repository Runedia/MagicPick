"""
필터 시스템 테스트
"""

import numpy as np
from filters.base_filter import BaseFilter, FilterManager, FilterPipeline
from filters.basic_filters import (
    GrayscaleFilter, SepiaFilter, InvertFilter,
    SoftFilter, SharpFilter, WarmFilter, CoolFilter, VignetteFilter
)


def create_test_image(width=100, height=100):
    """테스트용 이미지 생성"""
    # 그라데이션 이미지 생성
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            image[i, j] = [
                int(255 * j / width),
                int(255 * i / height),
                128
            ]
    return image


def test_grayscale_filter():
    """회색조 필터 테스트"""
    print("Testing GrayscaleFilter...")
    filter_obj = GrayscaleFilter()
    test_image = create_test_image()
    result = filter_obj.apply(test_image)
    
    # 회색조는 R=G=B 확인
    assert np.all(result[:, :, 0] == result[:, :, 1])
    assert np.all(result[:, :, 1] == result[:, :, 2])
    print("[PASS] GrayscaleFilter passed")


def test_sepia_filter():
    """세피아 필터 테스트"""
    print("Testing SepiaFilter...")
    filter_obj = SepiaFilter()
    test_image = create_test_image()
    result = filter_obj.apply(test_image)
    
    # 결과가 유효한 범위 내에 있는지 확인
    assert np.all(result >= 0) and np.all(result <= 255)
    print("[PASS] SepiaFilter passed")


def test_invert_filter():
    """반전 필터 테스트"""
    print("Testing InvertFilter...")
    filter_obj = InvertFilter()
    test_image = create_test_image()
    result = filter_obj.apply(test_image)
    
    # 반전 확인: original + inverted = 255
    assert np.allclose(test_image + result, 255)
    print("[PASS] InvertFilter passed")


def test_warm_cool_filters():
    """따뜻한/차가운 필터 테스트"""
    print("Testing Warm/Cool Filters...")
    warm_filter = WarmFilter()
    cool_filter = CoolFilter()
    test_image = create_test_image()
    
    warm_result = warm_filter.apply(test_image)
    cool_result = cool_filter.apply(test_image)
    
    # 따뜻한 필터는 빨강 채널 증가
    assert np.mean(warm_result[:, :, 0]) >= np.mean(test_image[:, :, 0])
    # 차가운 필터는 파랑 채널 증가
    assert np.mean(cool_result[:, :, 2]) >= np.mean(test_image[:, :, 2])
    
    print("[PASS] Warm/Cool Filters passed")


def test_filter_manager():
    """FilterManager 테스트"""
    print("Testing FilterManager...")
    manager = FilterManager()
    
    # 필터 등록
    manager.register_filter(GrayscaleFilter())
    manager.register_filter(SepiaFilter())
    
    # 등록 확인
    assert manager.get_filter('회색조') is not None
    assert manager.get_filter('세피아') is not None
    assert manager.get_filter('존재하지않는필터') is None
    
    # 필터 적용
    test_image = create_test_image()
    result = manager.apply_filter(test_image, '회색조')
    assert result is not None
    
    print("[PASS] FilterManager passed")


def test_filter_pipeline():
    """FilterPipeline 테스트"""
    print("Testing FilterPipeline...")
    pipeline = FilterPipeline()
    
    # 필터 추가
    pipeline.add_filter(GrayscaleFilter())
    pipeline.add_filter(InvertFilter())
    
    # 파이프라인 적용
    test_image = create_test_image()
    result = pipeline.apply_pipeline(test_image)
    
    assert result is not None
    # 회색조 + 반전 결과 확인
    assert np.all(result[:, :, 0] == result[:, :, 1])
    
    print("[PASS] FilterPipeline passed")


def run_all_tests():
    """모든 테스트 실행"""
    print("=" * 50)
    print("Running Filter System Tests")
    print("=" * 50)
    
    try:
        test_grayscale_filter()
        test_sepia_filter()
        test_invert_filter()
        test_warm_cool_filters()
        test_filter_manager()
        test_filter_pipeline()
        
        print("=" * 50)
        print("All tests passed!")
        print("=" * 50)
        return True
        
    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n[FAIL] Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    run_all_tests()
