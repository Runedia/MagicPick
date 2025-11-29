import numpy as np
from numba import njit, prange

from filters.base_filter import BaseFilter

# -----------------------------------------------------------------------------
# Numba JIT Kernels
# -----------------------------------------------------------------------------


@njit(fastmath=True, inline="always")
def _saturate(x):
    """값을 [0, 1] 범위로 클램핑"""
    return min(max(x, 0.0), 1.0)


@njit(parallel=True, fastmath=True)
def _luma_sharpen_kernel(
    padded_img,
    h,
    w,
    pad_y,
    pad_x,
    offsets_y,
    offsets_x,
    w_vec,
    sharp_clamp,
    show_sharpen,
):
    """
    LumaSharpen의 핵심 로직을 병렬 처리하는 Numba 커널

    Args:
        padded_img: 패딩된 입력 이미지 (H', W', 3)
        h, w: 원본 이미지 크기
        pad_y, pad_x: 패딩 두께
        offsets_y, offsets_x: 패턴에 따른 Y, X 오프셋 배열 (정수형)
        w_vec: 미리 계산된 가중치 벡터 [R_w, G_w, B_w]
        sharp_clamp: 클램핑 강도
        show_sharpen: 샤프닝 마스크 보기 모드 여부
    """
    out = np.empty((h, w, 3), dtype=np.uint8)

    # 가중치 언패킹
    wr, wg, wb = w_vec[0], w_vec[1], w_vec[2]

    # 샘플 개수 및 정규화 계수
    num_samples = len(offsets_y)
    inv_samples = 1.0 / num_samples if num_samples > 0 else 0.0

    # 병렬 루프 실행
    for y in prange(h):
        for x in range(w):
            # 패딩된 이미지 기준 좌표
            py, px = y + pad_y, x + pad_x

            # 1. 원본 픽셀 읽기
            ori_r = padded_img[py, px, 0]
            ori_g = padded_img[py, px, 1]
            ori_b = padded_img[py, px, 2]

            # 2. 블러 픽셀 계산 (패턴 오프셋 평균)
            if num_samples == 0:
                blur_r, blur_g, blur_b = ori_r, ori_g, ori_b
            else:
                acc_r, acc_g, acc_b = 0.0, 0.0, 0.0
                for i in range(num_samples):
                    sy = py + offsets_y[i]
                    sx = px + offsets_x[i]
                    acc_r += padded_img[sy, sx, 0]
                    acc_g += padded_img[sy, sx, 1]
                    acc_b += padded_img[sy, sx, 2]

                blur_r = acc_r * inv_samples
                blur_g = acc_g * inv_samples
                blur_b = acc_b * inv_samples

            # 3. 샤프닝 신호 추출 (원본 - 블러)
            diff_r = ori_r - blur_r
            diff_g = ori_g - blur_g
            diff_b = ori_b - blur_b

            # 4. 루마 내적 및 클램핑 계수 적용
            # sharp_luma = saturate(dot(sharp, strength_luma_clamp) + 0.5)
            # w_vec은 이미 (coef * strength * pattern_factor * (0.5/clamp))가 계산된 값임
            dot_val = (diff_r * wr + diff_g * wg + diff_b * wb) + 0.5

            # [0, 1] 범위로 saturate
            sharp_luma_01 = _saturate(dot_val)

            # 5. 최종 스케일링
            # sharp_luma = (sharp_clamp * 2.0) * sharp_luma - sharp_clamp
            sharp_luma_scaled = (sharp_clamp * 2.0) * sharp_luma_01 - sharp_clamp

            # 6. 결과 출력
            if show_sharpen:
                # 디버그 모드: saturate(0.5 + (sharp_luma * 4.0))
                val = 0.5 + sharp_luma_scaled * 4.0
                gray = _saturate(val)
                out[y, x, 0] = int(gray * 255)
                out[y, x, 1] = int(gray * 255)
                out[y, x, 2] = int(gray * 255)
            else:
                # 일반 모드: ori + sharp_luma
                out[y, x, 0] = int(_saturate(ori_r + sharp_luma_scaled) * 255)
                out[y, x, 1] = int(_saturate(ori_g + sharp_luma_scaled) * 255)
                out[y, x, 2] = int(_saturate(ori_b + sharp_luma_scaled) * 255)

    return out


class LumaSharpenFilterAccurate(BaseFilter):
    """
    LumaSharpen 정확한 구현 (Numba 최적화 버전)

    기존 HLSL 셰이더 로직을 100% 유지하면서,
    Numba JIT 컴파일을 통해 CPU 연산 속도를 획기적으로 개선했습니다.
    """

    def __init__(self):
        super().__init__("LumaSharpen", "루마 기반 언샤프 마스크 (정확/고속)")

        self.sharp_strength = 0.65
        self.sharp_clamp = 0.035
        self.pattern = 1
        self.offset_bias = 1.0
        self.show_sharpen = False

        self.coef_luma = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """LumaSharpen 효과 적용"""
        # 파라미터 업데이트
        self.sharp_strength = params.get("sharp_strength", self.sharp_strength)
        self.sharp_clamp = params.get("sharp_clamp", self.sharp_clamp)
        self.pattern = int(params.get("pattern", self.pattern))
        self.offset_bias = params.get("offset_bias", self.offset_bias)
        self.show_sharpen = params.get("show_sharpen", self.show_sharpen)

        # 0. 데이터 준비
        img_float = image.astype(np.float32) / 255.0
        h, w = img_float.shape[:2]

        # 1. 패턴별 오프셋 정의 (상대 좌표)
        # (offset_x, offset_y)
        if self.pattern == 0:  # Fast
            raw_offsets = [(1.0 / 3.0, -1.0 / 3.0), (-1.0 / 3.0, 1.0 / 3.0)]
            pattern_factor = 1.5
        elif self.pattern == 1:  # Normal
            raw_offsets = [(0.5, -0.5), (-0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
            pattern_factor = 1.0  # (se+sw+ne+nw)*0.25 -> average
        elif self.pattern == 2:  # Wider
            raw_offsets = [(0.4, -1.2), (-1.2, -0.4), (1.2, 0.4), (-0.4, 1.2)]
            pattern_factor = 0.51
        elif self.pattern == 3:  # Pyramid
            raw_offsets = [(0.5, -1.0), (-1.0, -0.5), (1.0, 0.5), (-0.5, 1.0)]
            pattern_factor = 0.666
        else:
            # 패턴이 유효하지 않으면 원본 반환 (혹은 offsets 비우기)
            raw_offsets = []
            pattern_factor = 0.0

        # 2. 오프셋 정수화 및 패딩 계산
        offsets_y = []
        offsets_x = []
        max_off_y = 0
        max_off_x = 0

        for ox, oy in raw_offsets:
            # int(round(...))는 원본 로직과 동일
            idx = int(round(ox * self.offset_bias))
            idy = int(round(oy * self.offset_bias))
            offsets_x.append(idx)
            offsets_y.append(idy)
            max_off_x = max(max_off_x, abs(idx))
            max_off_y = max(max_off_y, abs(idy))

        # Numba로 넘길 때는 NumPy 배열이어야 함
        np_offsets_y = np.array(offsets_y, dtype=np.int32)
        np_offsets_x = np.array(offsets_x, dtype=np.int32)

        # 3. 이미지 패딩 (Zero Padding = mode='constant', value=0)
        # 최소 1픽셀은 패딩해야 안전 (오프셋이 0이어도)
        pad_y = max(max_off_y, 1)
        pad_x = max(max_off_x, 1)

        padded_img = np.pad(
            img_float, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="constant"
        )

        # 4. 가중치 벡터 사전 계산 (Vector Optimization)
        # Numba 내부 연산을 줄이기 위해 상수항들을 미리 곱해서 넘깁니다.
        # Original: dot(sharp, coef * strength) * factor * (0.5 / clamp)
        # Combined: sharp * [ coef * strength * factor * (0.5 / clamp) ]

        safe_clamp = max(self.sharp_clamp, 1e-6)  # 0 나누기 방지
        w_factor = (self.sharp_strength * pattern_factor) * (0.5 / safe_clamp)
        w_vec = self.coef_luma * w_factor

        # 5. JIT 커널 실행
        result = _luma_sharpen_kernel(
            padded_img,
            h,
            w,
            pad_y,
            pad_x,
            np_offsets_y,
            np_offsets_x,
            w_vec,
            safe_clamp,
            self.show_sharpen,
        )

        return result
