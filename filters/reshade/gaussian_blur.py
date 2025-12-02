"""
Gaussian Blur 필터

2-pass 분리형 가우시안 블러를 구현합니다.
5가지 반경(radius) 옵션으로 다양한 블러 강도를 지원합니다.
"""

import numpy as np
from numba import njit, prange

from filters.base_filter import BaseFilter


@njit(parallel=True, fastmath=True, cache=True)
def _gaussian_pass_kernel(image, offsets, weights, offset_scale, is_horizontal):
    """
    가우시안 블러 1D 패스 (Numba 가속)
    """
    h, w, c = image.shape
    result = np.zeros((h, w, c), dtype=np.float32)

    num_samples = len(offsets)

    # 방향 벡터
    dx = 1 if is_horizontal else 0
    dy = 0 if is_horizontal else 1

    for y in prange(h):
        for x in range(w):
            # 중심 픽셀
            r_acc = image[y, x, 0] * weights[0]
            g_acc = image[y, x, 1] * weights[0]
            b_acc = image[y, x, 2] * weights[0]

            for i in range(1, num_samples):
                off_val = offsets[i] * offset_scale

                # 정수 오프셋 (반올림)
                # Numba에서는 round가 float를 반환하므로 int로 캐스팅
                off_px = int(round(off_val))

                weight = weights[i]

                # Positive direction
                px_p = x + off_px * dx
                py_p = y + off_px * dy

                # 클램핑 (Edge 모드)
                px_p = min(max(px_p, 0), w - 1)
                py_p = min(max(py_p, 0), h - 1)

                r_acc += image[py_p, px_p, 0] * weight
                g_acc += image[py_p, px_p, 1] * weight
                b_acc += image[py_p, px_p, 2] * weight

                # Negative direction
                px_n = x - off_px * dx
                py_n = y - off_px * dy

                # 클램핑
                px_n = min(max(px_n, 0), w - 1)
                py_n = min(max(py_n, 0), h - 1)

                r_acc += image[py_n, px_n, 0] * weight
                g_acc += image[py_n, px_n, 1] * weight
                b_acc += image[py_n, px_n, 2] * weight

            result[y, x, 0] = r_acc
            result[y, x, 1] = g_acc
            result[y, x, 2] = b_acc

    return result


class GaussianBlurFilter(BaseFilter):
    """가우시안 블러 필터 (2-pass separable, Numba Accelerated)"""

    # 반경별 오프셋 및 가중치 프리셋
    BLUR_PRESETS = {
        0: {
            "offsets": [0.0, 1.1824255238, 3.0293122308, 5.0040701377],
            "weights": [
                0.39894,
                0.2959599993,
                0.0045656525,
                0.00000149278686458842,
            ],
        },
        1: {
            "offsets": [
                0.0,
                1.4584295168,
                3.40398480678,
                5.3518057801,
                7.302940716,
                9.2581597095,
            ],
            "weights": [
                0.13298,
                0.23227575,
                0.1353261595,
                0.0511557427,
                0.01253922,
                0.0019913644,
            ],
        },
        2: {
            "offsets": [
                0.0,
                1.4895848401,
                3.4757135714,
                5.4618796741,
                7.4481042327,
                9.4344079746,
                11.420811147,
                13.4073334,
                15.3939936778,
                17.3808101174,
                19.3677999584,
            ],
            "weights": [
                0.06649,
                0.1284697563,
                0.111918249,
                0.0873132676,
                0.0610011113,
                0.0381655709,
                0.0213835661,
                0.0107290241,
                0.0048206869,
                0.0019396469,
                0.0006988718,
            ],
        },
        3: {
            "offsets": [
                0.0,
                1.4953705027,
                3.4891992113,
                5.4830312105,
                7.4768683759,
                9.4707125766,
                11.4645656736,
                13.4584295168,
                15.4523059431,
                17.4461967743,
                19.4401038149,
                21.43402885,
                23.4279736431,
                25.4219399344,
                27.4159294386,
            ],
            "weights": [
                0.0443266667,
                0.0872994708,
                0.0820892038,
                0.0734818355,
                0.0626171681,
                0.0507956191,
                0.0392263968,
                0.0288369812,
                0.0201808877,
                0.0134446557,
                0.0085266392,
                0.0051478359,
                0.0029586248,
                0.0016187257,
                0.0008430913,
            ],
        },
        4: {
            "offsets": [
                0.0,
                1.4953705027,
                3.4891992113,
                5.4830312105,
                7.4768683759,
                9.4707125766,
                11.4645656736,
                13.4584295168,
                15.4523059431,
                17.4461967743,
                19.4661974725,
                21.4627427973,
                23.4592916956,
                25.455844494,
                27.4524015179,
                29.4489630909,
                31.445529535,
                33.4421011704,
            ],
            "weights": [
                0.033245,
                0.0659162217,
                0.0636705814,
                0.0598194658,
                0.0546642566,
                0.0485871646,
                0.0420045997,
                0.0353207015,
                0.0288880982,
                0.0229808311,
                0.0177815511,
                0.013382297,
                0.0097960001,
                0.0069746748,
                0.0048301008,
                0.0032534598,
                0.0021315311,
                0.0013582974,
            ],
        },
    }

    def __init__(self):
        super().__init__("GaussianBlur", "가우시안 블러")
        self.radius = 1  # 0 ~ 4
        self.offset = 1.0  # 0.0 ~ 1.0 (추가 반경 조정)
        self.strength = 0.3  # 0.0 ~ 1.0

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """Gaussian Blur 필터 적용 (2-pass)"""
        # 파라미터 업데이트
        self.radius = int(params.get("radius", self.radius))
        self.offset = params.get("offset", self.offset)
        self.strength = params.get("strength", self.strength)

        # radius 범위 제한
        self.radius = max(0, min(4, self.radius))

        img_float = image.astype(np.float32) / 255.0

        # 프리셋 가져오기
        preset = self.BLUR_PRESETS[self.radius]
        # Numba로 넘기기 위해 NumPy 배열로 변환
        offsets = np.array(preset["offsets"], dtype=np.float32)
        weights = np.array(preset["weights"], dtype=np.float32)

        # Pass 1: 수평 블러
        blur_h = _gaussian_pass_kernel(img_float, offsets, weights, self.offset, True)

        # Pass 2: 수직 블러
        blur_final = _gaussian_pass_kernel(blur_h, offsets, weights, self.offset, False)

        # 강도 조정 (Original과 Blend)
        result = img_float + (blur_final - img_float) * self.strength
        result = np.clip(result, 0.0, 1.0)

        return (result * 255).astype(np.uint8)
