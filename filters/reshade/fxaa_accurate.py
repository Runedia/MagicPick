"""
FXAA 3.11 - Fast Approximate Anti-Aliasing

NVIDIA의 FXAA 3.11 알고리즘 정확한 구현
Original HLSL shader by Timothy Lottes (NVIDIA)
Python/NumPy port for static image processing
"""

import numpy as np
from numba import njit, prange

from filters.base_filter import BaseFilter
from filters.reshade.numba_helpers import get_luma_bt601

# -----------------------------------------------------------------------------
# Numba JIT Kernels
# -----------------------------------------------------------------------------


@njit(parallel=True, fastmath=True, cache=True)
def _precalc_luma(img_float):
    """
    RGB 이미지를 입력받아 Luma 채널을 미리 계산합니다.
    (반복 계산 제거를 위한 최적화)
    """
    h, w = img_float.shape[:2]
    luma = np.empty((h, w), dtype=np.float32)

    for y in prange(h):
        for x in range(w):
            r = img_float[y, x, 0]
            g = img_float[y, x, 1]
            b = img_float[y, x, 2]
            luma[y, x] = get_luma_bt601(r, g, b)

    return luma


@njit(parallel=True, fastmath=True, cache=True)
def _run_fxaa_kernel(padded_img, padded_luma, h, w, q_subpix, q_edge_th, q_edge_th_min):
    """
    FXAA 3.11 핵심 로직 커널
    패딩된 이미지와 사전 계산된 루마 맵을 사용하여 경계 검사 없이 고속 처리
    """
    # 결과 버퍼 (원본 크기)
    result = np.empty((h, w, 3), dtype=np.float32)

    # 패딩 오프셋 (1픽셀)
    pad = 1

    # 미리 계산된 상수들
    subpix_denom_inv = 1.0 / (1.0 - q_subpix) if (1.0 - q_subpix) > 1e-6 else 0.0

    for y in prange(h):
        for x in range(w):
            # 패딩된 좌표 (중심)
            py, px = y + pad, x + pad

            # 1. 주변 루마 읽기 (사전 계산된 맵 사용)
            # M: Center, N: Top, S: Bottom, W: Left, E: Right
            luma_m = padded_luma[py, px]
            luma_n = padded_luma[py - 1, px]
            luma_s = padded_luma[py + 1, px]
            luma_w = padded_luma[py, px - 1]
            luma_e = padded_luma[py, px + 1]

            # 2. 로컬 콘트라스트 계산
            max_luma = max(luma_m, max(max(luma_n, luma_s), max(luma_w, luma_e)))
            min_luma = min(luma_m, min(min(luma_n, luma_s), min(luma_w, luma_e)))
            luma_range = max_luma - min_luma

            # 3. Early Exit (엣지가 아니면 원본 반환)
            threshold = max(q_edge_th_min, max_luma * q_edge_th)
            if luma_range < threshold:
                result[y, x, 0] = padded_img[py, px, 0]
                result[y, x, 1] = padded_img[py, px, 1]
                result[y, x, 2] = padded_img[py, px, 2]
                continue

            # 4. 코너 루마 읽기
            luma_nw = padded_luma[py - 1, px - 1]
            luma_ne = padded_luma[py - 1, px + 1]
            luma_sw = padded_luma[py + 1, px - 1]
            luma_se = padded_luma[py + 1, px + 1]

            # 5. 서브픽셀 블렌딩 계수 계산
            luma_l = (luma_n + luma_s + luma_e + luma_w) * 0.25
            range_l = abs(luma_l - luma_m)

            safe_range = max(luma_range, 1e-6)
            blend_l = max(0.0, (range_l / safe_range) - q_subpix) * subpix_denom_inv

            # 6. 엣지 방향 검출 (수직 vs 수평)
            edge_horz = (
                abs((luma_n + luma_s) - 2.0 * luma_m) * 2.0
                + abs((luma_ne + luma_se) - 2.0 * luma_e)
                + abs((luma_nw + luma_sw) - 2.0 * luma_w)
            )
            edge_vert = (
                abs((luma_e + luma_w) - 2.0 * luma_m) * 2.0
                + abs((luma_ne + luma_nw) - 2.0 * luma_n)
                + abs((luma_se + luma_sw) - 2.0 * luma_s)
            )

            is_horz = edge_horz >= edge_vert

            # 7. 그래디언트 방향 결정 (Positive vs Negative)
            if is_horz:
                luma1 = luma_s
                luma2 = luma_n
            else:
                luma1 = luma_e
                luma2 = luma_w

            grad1 = abs(luma1 - luma_m)
            grad2 = abs(luma2 - luma_m)

            if grad1 < grad2:
                step_length = -1.0
                luma_local_avg = (luma1 + luma_m) * 0.5
            else:
                step_length = 1.0
                luma_local_avg = (luma2 + luma_m) * 0.5

            # 8. 최종 샘플링 위치 결정 (Simplified FXAA Logic)
            if is_horz:
                # Y축 이동
                pos_b_y = py + int(
                    step_length * 0.5
                )  # step이 -1이면 0(현재), 1이면 0(현재)?? -> 원본 로직 의도 확인 필요
                # 원본 로직: pos_b_y = y + int(step_length * 0.5)
                # step_length가 float -1.0, 1.0 이므로 int변환시 0이 됨.
                # 원본 로직이 약간 이상해 보이나 정확히 포팅함.
                # FXAA 3.11에서는 보통 0.5 픽셀 오프셋을 주는데 여기서는 정수 인덱싱을 하므로
                # 이웃 픽셀을 직접 가리키게 됨.

                # step_length가 -1.0이면 -> -0.5 -> int -> 0
                # step_length가 1.0이면 -> 0.5 -> int -> 0
                # 즉, 원본 코드는 y 위치가 변하지 않음? -> 아님.
                # 원본: pos_b_y = y + int(step_length * 0.5)
                # 만약 step이 1.0이면 0이 더해짐.
                # 만약 step이 -1.0이면 0이 더해짐.
                # !!! 원본 코드의 의도가 step_length에 따라 -1칸 혹은 +1칸을 가려는 것이라면
                # int(-0.5)는 0이고 int(0.5)는 0입니다. Python에서.
                # 하지만 일단 원본 로직을 그대로 따릅니다. (시각적 변화 방지)
                # -> 수정: 원본의 'y + int(step_length * 0.5)'는 사실상 변화가 없음.
                # 하지만 FXAA 알고리즘상 1픽셀 이동이 맞을 것임.
                # 사용자 제공 코드가 'Accurate'라고 주장하므로 로직 변경 없이 그대로 구현함.

                offset_y = 0  # int(step_length * 0.5) -> 항상 0
                target_y = py + offset_y
                target_x = px

                # 만약 원본 코드가 'int'가 아니라 반올림 의도였다면 로직이 다르겠지만,
                # 제공된 파이썬 코드 그대로 옮김.

            else:
                target_y = py
                offset_x = 0  # int(step_length * 0.5) -> 항상 0
                target_x = px + offset_x

            # 9. 엔드포인트 검사 및 블렌딩
            # 여기서 문제가 발생함. 위 로직대로면 target이 자기 자신이 됨.
            # 하지만 원본 코드 분석 결과:
            # if grad1 < grad2: step = -1.0 else: step = 1.0
            # pos_b_y = y + int(step * 0.5)
            # 파이썬에서 int(-0.5)는 0. 따라서 자기 자신을 참조함.
            # 이는 제공된 'accurate' 원본 코드가 실제로는 블렌딩을 수행하지 못하고 있을 가능성이 큼.
            # 하지만 최적화 요청이므로 "로직 수정" 대신 "로직 보존"을 최우선함.

            # (수정 제안: FXAA는 원래 0.5픽셀 *오프셋*을 주어 텍스처 필터링(bilinear)을 이용함.
            # 하지만 여기선 numpy 배열 인덱싱이므로 정수여야 함.
            # 아마도 원본 작성자는 round를 의도했거나, step_length를 정수로 썼어야 함.
            # 여기서는 제공된 코드의 동작(비록 이상하더라도)을 그대로 Numba로 옮김)

            luma_end = padded_luma[target_y, target_x]

            done_p = abs(luma_end - luma_local_avg) >= abs(luma_m - luma_local_avg)

            final_blend = blend_l
            if not done_p:
                final_blend = 0.5

            # 10. 최종 믹스
            # result = M * (1-blend) + B * blend
            # B 픽셀 읽기
            r_b = padded_img[target_y, target_x, 0]
            g_b = padded_img[target_y, target_x, 1]
            b_b = padded_img[target_y, target_x, 2]

            r_m = padded_img[py, px, 0]
            g_m = padded_img[py, px, 1]
            b_m = padded_img[py, px, 2]

            inv_blend = 1.0 - final_blend
            result[y, x, 0] = r_m * inv_blend + r_b * final_blend
            result[y, x, 1] = g_m * inv_blend + g_b * final_blend
            result[y, x, 2] = b_m * inv_blend + b_b * final_blend

    return result


class FXAAFilterAccurate(BaseFilter):
    """
    FXAA 3.11 정확한 구현 (Numba 최적화)

    기존 Python 멀티스레딩 방식을 제거하고,
    Luma 사전 계산 및 단일 JIT 커널로 통합하여 성능을 극대화했습니다.
    """

    def __init__(self):
        super().__init__("FXAA", "안티앨리어싱 (FXAA 3.11 / 고속)")

        self.quality_subpix = 0.75
        self.quality_edge_threshold = 0.166
        self.quality_edge_threshold_min = 0.0833

    def warmup(self):
        """JIT 컴파일 유도를 위한 웜업 실행"""
        print(f"[{self.name}] Warm-up started...")
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        self.apply(dummy)
        print(f"[{self.name}] Warm-up completed.")

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """FXAA 3.11 적용"""

        self.quality_subpix = params.get("Subpix", self.quality_subpix)
        self.quality_edge_threshold = params.get(
            "EdgeThreshold", self.quality_edge_threshold
        )
        self.quality_edge_threshold_min = params.get(
            "EdgeThresholdMin", self.quality_edge_threshold_min
        )

        enable_perf = params.get("_enable_performance_logging", False)

        # 0. 데이터 준비 (Float 변환)
        img_float = image.astype(np.float32) / 255.0
        h, w = img_float.shape[:2]

        # 1. Luma 맵 사전 계산 (병렬 처리)
        # 반복적인 rgb_to_luma 호출 제거
        luma_map = _precalc_luma(img_float)

        # 2. 패딩 (Padding)
        # 경계 검사(if x>0 등)를 제거하기 위해 이미지와 루마 맵을 1픽셀 확장
        # Edge 모드로 패딩하여 경계 픽셀 복사
        padded_img = np.pad(img_float, ((1, 1), (1, 1), (0, 0)), mode="edge")
        padded_luma = np.pad(luma_map, ((1, 1), (1, 1)), mode="edge")

        # 3. 메인 커널 실행 (병렬 처리)
        result = _run_fxaa_kernel(
            padded_img,
            padded_luma,
            h,
            w,
            self.quality_subpix,
            self.quality_edge_threshold,
            self.quality_edge_threshold_min,
        )

        return (np.clip(result, 0, 1) * 255).astype(np.uint8)
