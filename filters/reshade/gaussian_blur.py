"""
Gaussian Blur 필터

2-pass 분리형 가우시안 블러를 구현합니다.
5가지 반경(radius) 옵션으로 다양한 블러 강도를 지원합니다.
"""

import numpy as np

from filters.base_filter import BaseFilter

from .hlsl_helpers import lerp, saturate


class GaussianBlurFilter(BaseFilter):
    """가우시안 블러 필터 (2-pass separable)"""

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

    def _apply_gaussian_1d(self, image, offsets, weights, horizontal=True):
        """1D 가우시안 블러 적용 (수평 또는 수직)"""
        h, w = image.shape[:2]
        result = np.zeros_like(image, dtype=np.float32)

        # 중심 가중치
        result = image * weights[0]

        # 양방향 샘플링
        for i in range(1, len(offsets)):
            offset_scaled = offsets[i] * self.offset

            if horizontal:
                # 수평 블러
                offset_px = int(round(offset_scaled))
                if offset_px > 0:
                    # +방향
                    shifted_pos = np.zeros_like(image)
                    shifted_pos[:, : w - offset_px] = image[:, offset_px:]
                    result += shifted_pos * weights[i]

                    # -방향
                    shifted_neg = np.zeros_like(image)
                    shifted_neg[:, offset_px:] = image[:, : w - offset_px]
                    result += shifted_neg * weights[i]
            else:
                # 수직 블러
                offset_px = int(round(offset_scaled))
                if offset_px > 0:
                    # +방향
                    shifted_pos = np.zeros_like(image)
                    shifted_pos[: h - offset_px, :] = image[offset_px:, :]
                    result += shifted_pos * weights[i]

                    # -방향
                    shifted_neg = np.zeros_like(image)
                    shifted_neg[offset_px:, :] = image[: h - offset_px, :]
                    result += shifted_neg * weights[i]

        return result

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """Gaussian Blur 필터 적용 (2-pass)"""
        # 파라미터 업데이트
        self.radius = int(params.get("radius", self.radius))
        self.offset = params.get("offset", self.offset)
        self.strength = params.get("strength", self.strength)

        # radius 범위 제한
        self.radius = max(0, min(4, self.radius))

        img_float = image.astype(np.float32) / 255.0
        orig = img_float.copy()

        # 프리셋 가져오기
        preset = self.BLUR_PRESETS[self.radius]
        offsets = preset["offsets"]
        weights = preset["weights"]

        # Pass 1: 수평 블러
        blur_h = self._apply_gaussian_1d(img_float, offsets, weights, horizontal=True)

        # Pass 2: 수직 블러
        blur_final = self._apply_gaussian_1d(blur_h, offsets, weights, horizontal=False)

        # 강도 조정
        result = lerp(orig, blur_final, self.strength)
        result = saturate(result)

        return (result * 255).astype(np.uint8)
