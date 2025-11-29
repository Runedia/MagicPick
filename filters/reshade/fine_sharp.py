"""
ReShade FineSharp 필터
"""

import cv2
import numpy as np

from ..base_filter import BaseFilter
from .hlsl_helpers import (
    lerp,
    rgb_to_yuv,
    saturate,
    yuv_to_rgb,
)


class FineSharpFilter(BaseFilter):
    def __init__(self):
        super().__init__(
            "FineSharp", "ReShade의 FineSharp 효과를 적용하여 이미지를 선명하게 합니다."
        )
        self.set_default_params(
            {
                "sstr": 2.0,  # Sharpening Strength
                "cstr": 0.9,  # Equalization Strength
                "xstr": 0.19,  # XSharpen-style final sharpening
                "xrep": 0.25,  # Repair artefacts
                "lstr": 1.49,  # Modifier for non-linear sharpening
                "pstr": 1.272,  # Exponent for non-linear sharpening
                "mode": 1,  # Technique mode
            }
        )

    def _sharp_diff(self, c, sstr, lstr, pstr):
        ldmp = sstr + 0.1
        t = c[:, :, 3] - c[:, :, 0]  # alpha (original Y) - x (processed Y)

        sign_t = np.sign(t)
        abs_t = np.abs(t)

        term1 = sstr / 255.0
        term2 = np.power(abs_t / (lstr / 255.0), 1.0 / pstr)
        term3 = (t * t) / ((t * t) + (ldmp / 65025.0))

        return sign_t * term1 * term2 * term3

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        params = self.validate_params(params)
        sstr = float(params["sstr"])
        cstr = float(params["cstr"])
        xstr = float(params["xstr"])
        xrep = float(params["xrep"])
        lstr = float(params["lstr"])
        pstr = float(params["pstr"])
        mode = int(params["mode"])

        img_float = image.astype(np.float32)

        # Pass 0: ToYUV
        # yuv.x = Y, yuv.y = U, yuv.z = V
        # o.a = original Y
        yuv = rgb_to_yuv(img_float)
        pass0_out = np.dstack((yuv, yuv[:, :, 0]))  # R,G,B,A = Y,U,V,Y_original

        # Grain removal passes
        if mode == 1:
            # Pass 1: RemoveGrain11 (Box Blur 3x3)
            pass1_in = pass0_out[:, :, 0]  # Y channel
            pass1_out_y = cv2.boxFilter(pass1_in, -1, (3, 3))
            pass1_out = pass0_out.copy()
            pass1_out[:, :, 0] = pass1_out_y

            # Pass 2: RemoveGrain4 (Median Blur 3x3)
            pass2_in = pass1_out[:, :, 0]
            pass2_out_y = cv2.medianBlur(pass2_in, 3)
            pass2_out = pass1_out.copy()
            pass2_out[:, :, 0] = pass2_out_y

            finesharp_in = pass2_out

        elif mode == 2:
            # Pass 2: RemoveGrain4 (Median Blur 3x3)
            pass2_in = pass0_out[:, :, 0]
            pass2_out_y = cv2.medianBlur(pass2_in, 3)
            pass2_out = pass0_out.copy()
            pass2_out[:, :, 0] = pass2_out_y

            # Pass 1: RemoveGrain11 (Box Blur 3x3)
            pass1_in = pass2_out[:, :, 0]
            pass1_out_y = cv2.boxFilter(pass1_in, -1, (3, 3))
            pass1_out = pass2_out.copy()
            pass1_out[:, :, 0] = pass1_out_y

            finesharp_in = pass1_out

        else:  # Mode 3
            # Pass 2: RemoveGrain4 (Median Blur 3x3)
            pass2_in_1 = pass0_out[:, :, 0]
            pass2_out_y_1 = cv2.medianBlur(pass2_in_1, 3)
            pass2_out_1 = pass0_out.copy()
            pass2_out_1[:, :, 0] = pass2_out_y_1

            # Pass 1: RemoveGrain11 (Box Blur 3x3)
            pass1_in = pass2_out_1[:, :, 0]
            pass1_out_y = cv2.boxFilter(pass1_in, -1, (3, 3))
            pass1_out = pass2_out_1.copy()
            pass1_out[:, :, 0] = pass1_out_y

            # Pass 2 again
            pass2_in_2 = pass1_out[:, :, 0]
            pass2_out_y_2 = cv2.medianBlur(pass2_in_2, 3)
            pass2_out_2 = pass1_out.copy()
            pass2_out_2[:, :, 0] = pass2_out_y_2

            finesharp_in = pass2_out_2

        # Pass 3: FineSharpA
        pass3_in = finesharp_in

        # _sharp_diff를 전체 이미지에 대해 계산
        sd = self._sharp_diff(pass3_in, sstr, lstr, pstr)

        # HLSL 코드의 가중 평균을 커널 필터로 구현
        # o.x += o.x; (x2)
        # o.x += add(neighbors);
        # o.x += o.x; (x2)
        # o.x += add(diag_neighbors);
        # o.x *= 0.0625; (/16)
        # 이는 [[1, 2, 1], [2, 4, 2], [1, 2, 1]] / 16 가중치 커널과 동일
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0

        # 각 픽셀의 sd 값에 대해 주변 sd 값의 가중 평균을 계산
        sd_wa = cv2.filter2D(sd, -1, kernel)

        pass3_out = pass3_in.copy()
        # o.x = o.a + sd;
        pass3_out[:, :, 0] = pass3_in[:, :, 3] + sd
        # o.x -= cstr * sd_wa;
        pass3_out[:, :, 0] -= cstr * sd_wa
        # o.a = o.x
        pass3_out[:, :, 3] = pass3_out[:, :, 0]

        # Pass 4: FineSharpB
        pass4_in = pass3_out
        pass4_in_y = pass4_in[:, :, 3]  # .a channel (Y from Pass 3)

        # Unsharp mask: o.x = mad(9.9, (o.a - o.x), o.a)
        # o.x is a 3x3 blur of o.a
        blurred_y = cv2.blur(pass4_in_y, (3, 3))
        unsharp_y = (pass4_in_y - blurred_y) * 9.9 + pass4_in_y

        # Clamp using 2nd and 8th value in 3x3 neighborhood
        # This requires getting all 9 neighbors and sorting
        h, w = pass4_in_y.shape
        neighbors = []
        for j in range(-1, 2):
            for i in range(-1, 2):
                # Use np.roll for simplicity to get neighbors
                rolled = np.roll(pass4_in_y, shift=(j, i), axis=(0, 1))
                neighbors.append(rolled)

        neighborhood = np.stack(neighbors, axis=-1)
        neighborhood.sort(axis=-1)

        t2 = neighborhood[:, :, 1]  # 2nd smallest value
        t8 = neighborhood[:, :, 7]  # 8th smallest value (2nd largest)

        # o.x = max(o.x, min(t2, o.a));
        # o.x = min(o.x, max(t8, o.a));
        lower_bound = np.minimum(t2, pass4_in_y)
        upper_bound = np.maximum(t8, pass4_in_y)

        pass4_out_y = np.clip(unsharp_y, lower_bound, upper_bound)

        pass4_out = pass4_in.copy()
        pass4_out[:, :, 0] = pass4_out_y
        # pass4_out[:,:,3] is not modified in this pass, it carries over from pass3

        # Pass 5: FineSharpC
        pass5_in = pass4_out
        pass5_in_x = pass5_in[:, :, 0]
        pass5_in_a = pass5_in[:, :, 3]  # Original Y from pass 3

        # Simplified edge detection
        laplacian = cv2.Laplacian(pass5_in_x, cv2.CV_32F, ksize=1)
        edge = np.abs(laplacian)

        pass5_out_y = lerp(pass5_in_a, pass5_in_x, xstr * (1.0 - saturate(edge * xrep)))

        pass5_out = pass5_in.copy()
        pass5_out[:, :, 0] = pass5_out_y

        # Pass 6: ToRGB
        pass6_in_yuv = pass5_out[:, :, :3]
        result_rgb = yuv_to_rgb(pass6_in_yuv)

        return np.clip(result_rgb, 0, 255).astype(np.uint8)
