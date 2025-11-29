"""
AdaptiveSharpen - 정확한 2-pass 구현

Original HLSL shader by bacondither
Python/NumPy port for static image processing
"""

import numpy as np
from numba import njit, prange

from filters.base_filter import BaseFilter

# -----------------------------------------------------------------------------
# Numba JIT Kernels (Scalar Math Helpers)
# -----------------------------------------------------------------------------


@njit(fastmath=True, inline="always", cache=True)
def _n_saturate(x):
    return min(max(x, 0.0), 1.0)


@njit(fastmath=True, inline="always", cache=True)
def _n_lerp(a, b, t):
    return a + t * (b - a)


@njit(fastmath=True, inline="always", cache=True)
def _n_sqr(x):
    return x * x


@njit(fastmath=True, inline="always", cache=True)
def _n_clamp(x, min_val, max_val):
    return min(max(x, min_val), max_val)


@njit(fastmath=True, inline="always", cache=True)
def _n_smoothstep(edge0, edge1, x):
    t = _n_saturate((x - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)


@njit(fastmath=True, inline="always", cache=True)
def _n_pow_safe(x, y):
    return abs(x) ** y


@njit(fastmath=True, inline="always", cache=True)
def _n_soft_lim_tanh_approx(v, s):
    if s == 0:
        return 0.0
    ratio = v / s
    ratio_sq = ratio * ratio
    return _n_saturate(abs(ratio) * (27.0 + ratio_sq) / (27.0 + 9.0 * ratio_sq)) * s


@njit(fastmath=True, inline="always", cache=True)
def _n_wpmean(a, b, w, pm_p):
    term_a = abs(w) * _n_pow_safe(a, pm_p)
    term_b = abs(1.0 - w) * _n_pow_safe(b, pm_p)
    return _n_pow_safe(term_a + term_b, 1.0 / pm_p)


# -----------------------------------------------------------------------------
# Numba JIT Kernels (Main Passes)
# -----------------------------------------------------------------------------


@njit(parallel=True, fastmath=True, cache=True)
def _run_pass0(padded_img, h, w, out_edge, out_luma):
    # Padding offset is (2, 2) based on max offset
    pad_y, pad_x = 2, 2

    # Luma coefficients
    cr, cg, cb = 0.2558, 0.6511, 0.0931

    # Pre-defined offsets for Pass 0
    # (dy, dx)
    offsets_y = np.array([0, -1, 0, 1, -1, 1, -1, 0, 1, 0, -2, 2, 0], dtype=np.int8)
    offsets_x = np.array([0, -1, -1, -1, 0, 0, 1, 1, 1, -2, 0, 0, 2], dtype=np.int8)

    for y in prange(h):
        for x in range(w):
            py, px = y + pad_y, x + pad_x

            # Read 13 neighbors & Calculate Luma
            # Instead of storing arrays, we compute accumulators on the fly where possible
            # But logic requires diffs, so we store lumas and colors

            # c[0] is center
            c0_r = padded_img[py, px, 0]
            c0_g = padded_img[py, px, 1]
            c0_b = padded_img[py, px, 2]

            # Center Luma
            luma_sq = c0_r * c0_r * cr + c0_g * c0_g * cg + c0_b * c0_b * cb
            luma_val = np.sqrt(luma_sq)
            out_luma[y, x] = luma_val  # Store Luma result

            # Read neighbors into a temporary local array for blur calculation
            # c_vals shape: (13, 3)
            c_vals = np.zeros((13, 3), dtype=np.float32)

            for i in range(13):
                oy, ox = offsets_y[i], offsets_x[i]
                c_vals[i, 0] = padded_img[py + oy, px + ox, 0]
                c_vals[i, 1] = padded_img[py + oy, px + ox, 1]
                c_vals[i, 2] = padded_img[py + oy, px + ox, 2]

            # Blur calculation
            # blur = (2*(c[2]+c[4]+c[5]+c[7]) + (c[1]+c[3]+c[6]+c[8]) + 4*c[0]) / 16.0
            blur_r = (
                2 * (c_vals[2, 0] + c_vals[4, 0] + c_vals[5, 0] + c_vals[7, 0])
                + (c_vals[1, 0] + c_vals[3, 0] + c_vals[6, 0] + c_vals[8, 0])
                + 4 * c_vals[0, 0]
            ) * 0.0625
            blur_g = (
                2 * (c_vals[2, 1] + c_vals[4, 1] + c_vals[5, 1] + c_vals[7, 1])
                + (c_vals[1, 1] + c_vals[3, 1] + c_vals[6, 1] + c_vals[8, 1])
                + 4 * c_vals[0, 1]
            ) * 0.0625
            blur_b = (
                2 * (c_vals[2, 2] + c_vals[4, 2] + c_vals[5, 2] + c_vals[7, 2])
                + (c_vals[1, 2] + c_vals[3, 2] + c_vals[6, 2] + c_vals[8, 2])
                + 4 * c_vals[0, 2]
            ) * 0.0625

            # c_comp calculation
            # exp2(sum(blur * -37/15)) -> exp2(sum * -2.4666)
            blur_sum = blur_r + blur_g + blur_b
            c_comp = _n_saturate(0.2666 + 0.9 * (2.0 ** (blur_sum * -2.4666)))

            # Edge calculation
            # b_diff = abs(blur - c[i])
            # Weighted sum of squares
            edge_accum = 0.0
            for k in range(3):  # RGB loop
                b_val = blur_r if k == 0 else (blur_g if k == 1 else blur_b)

                diff0 = abs(b_val - c_vals[0, k])
                diff1 = abs(b_val - c_vals[1, k])
                diff2 = abs(b_val - c_vals[2, k])
                diff3 = abs(b_val - c_vals[3, k])
                diff4 = abs(b_val - c_vals[4, k])
                diff5 = abs(b_val - c_vals[5, k])
                diff6 = abs(b_val - c_vals[6, k])
                diff7 = abs(b_val - c_vals[7, k])
                diff8 = abs(b_val - c_vals[8, k])
                diff9 = abs(b_val - c_vals[9, k])
                diff10 = abs(b_val - c_vals[10, k])
                diff11 = abs(b_val - c_vals[11, k])
                diff12 = abs(b_val - c_vals[12, k])

                term = (
                    1.38 * diff0
                    + 1.15 * (diff2 + diff4 + diff5 + diff7)
                    + 0.92 * (diff1 + diff3 + diff6 + diff8)
                    + 0.23 * (diff9 + diff10 + diff11 + diff12)
                )
                edge_accum += term * term

            edge_val = np.sqrt(edge_accum)
            out_edge[y, x] = edge_val * c_comp


@njit(parallel=True, fastmath=True, cache=True)
def _run_pass1(
    padded_img,
    padded_edge,
    padded_luma,
    h,
    w,
    curve_height,
    curveslope,
    L_overshoot,
    L_compr_low,
    L_compr_high,
    D_overshoot,
    D_compr_low,
    D_compr_high,
    scale_lim,
    scale_cs,
    pm_p,
):
    pad_y, pad_x = 3, 3  # Max offset is 3 in Pass 1

    offsets_y = np.array(
        [
            0,
            -1,
            0,
            1,
            -1,
            1,
            -1,
            0,
            1,
            0,
            -2,
            2,
            0,
            0,
            1,
            -1,
            3,
            2,
            2,
            -3,
            -2,
            -2,
            0,
            1,
            -1,
        ],
        dtype=np.int8,
    )
    offsets_x = np.array(
        [
            0,
            -1,
            -1,
            -1,
            0,
            0,
            1,
            1,
            1,
            -2,
            0,
            0,
            2,
            3,
            2,
            2,
            0,
            1,
            -1,
            0,
            1,
            -1,
            -3,
            -2,
            -2,
        ],
        dtype=np.int8,
    )

    # Weights constants
    W1_0, W1_1, W1_2 = 0.5, 1.0, 1.41421356
    W2_0, W2_1, W2_2 = 0.86602540, 1.0, 0.54772255

    for y in prange(h):
        for x in range(w):
            py, px = y + pad_y, x + pad_x

            # Read center pixel from original image
            img_r = padded_img[py, px, 0]
            img_g = padded_img[py, px, 1]
            img_b = padded_img[py, px, 2]

            # Read edges and lumas (25 neighbors)
            e = np.zeros(25, dtype=np.float32)
            l = np.zeros(25, dtype=np.float32)

            for i in range(25):
                oy, ox = offsets_y[i], offsets_x[i]
                e[i] = padded_edge[py + oy, px + ox]
                l[i] = padded_luma[py + oy, px + ox]

            # Max Edge calc
            # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0]
            maxedge = e[0]
            for i in range(1, 13):
                if e[i] > maxedge:
                    maxedge = e[i]

            # SBE Calculation
            # soft_if inline
            rcp_me = 1.0 / (abs(maxedge) + 0.03)

            def soft_if_val(idx1, idx2, idx3):
                return _n_saturate(
                    (e[idx1] + e[idx2] + e[idx3] + 0.056) * rcp_me - 0.85
                )

            sbe = (
                soft_if_val(2, 9, 22) * soft_if_val(7, 12, 13)
                + soft_if_val(4, 10, 19) * soft_if_val(5, 11, 16)
                + soft_if_val(1, 24, 21) * soft_if_val(8, 14, 17)
                + soft_if_val(3, 23, 18) * soft_if_val(6, 20, 15)
            )

            # Compression Factors (fast_ops = 1 fixed)
            sbe_lerp = _n_saturate(1.091 * sbe - 2.282)
            cs_L = _n_lerp(L_compr_low, L_compr_high, sbe_lerp)
            cs_D = _n_lerp(D_compr_low, D_compr_high, sbe_lerp)

            # dW calculation
            dw_lerp = _n_saturate(2.4 * e[0] - 0.82)
            dw0 = _n_sqr(_n_lerp(W1_0, W2_0, dw_lerp))
            dw1 = _n_sqr(_n_lerp(W1_1, W2_1, dw_lerp))
            dw2 = _n_sqr(_n_lerp(W1_2, W2_2, dw_lerp))

            # Mdiff calculation
            # mdiff_c0
            sum_l0 = (
                abs(l[0] - l[2])
                + abs(l[0] - l[4])
                + abs(l[0] - l[5])
                + abs(l[0] - l[7])
            )
            sum_l0_q = (
                abs(l[0] - l[1])
                + abs(l[0] - l[3])
                + abs(l[0] - l[6])
                + abs(l[0] - l[8])
            )
            mdiff_c0 = 0.02 + 3.0 * (sum_l0 + 0.25 * sum_l0_q)

            # Weights array (12)
            w_val = np.zeros(12, dtype=np.float32)

            # Helper for mdiff denominator
            # (g is center, others are neighbors)
            # abs(l[g]-l[a]) + ... + 0.5(...)
            def get_mdiff_denom(g, a, b, c, d, e_, f_):
                val = (
                    abs(l[g] - l[a])
                    + abs(l[g] - l[b])
                    + abs(l[g] - l[c])
                    + abs(l[g] - l[d])
                    + 0.5 * (abs(l[g] - l[e_]) + abs(l[g] - l[f_]))
                )
                return val + 1e-8

            w_val[0] = min(mdiff_c0 / get_mdiff_denom(24, 21, 2, 4, 9, 10, 1), dw1)
            w_val[1] = dw0
            w_val[2] = min(mdiff_c0 / get_mdiff_denom(23, 18, 5, 2, 9, 11, 3), dw1)
            w_val[3] = dw0
            w_val[4] = dw0
            w_val[5] = min(mdiff_c0 / get_mdiff_denom(4, 20, 15, 7, 10, 12, 6), dw1)
            w_val[6] = dw0
            w_val[7] = min(mdiff_c0 / get_mdiff_denom(5, 7, 17, 14, 12, 11, 8), dw1)
            w_val[8] = min(mdiff_c0 / get_mdiff_denom(2, 24, 23, 22, 1, 3, 9), dw2)
            w_val[9] = min(mdiff_c0 / get_mdiff_denom(20, 19, 21, 4, 1, 6, 10), dw2)
            w_val[10] = min(mdiff_c0 / get_mdiff_denom(17, 5, 18, 16, 3, 8, 11), dw2)
            w_val[11] = min(mdiff_c0 / get_mdiff_denom(13, 15, 7, 14, 6, 8, 12), dw2)

            # Smoothing weights
            w_val[0] = (
                max(max((w_val[8] + w_val[9]) * 0.25, w_val[0]), 0.25) + w_val[0]
            ) * 0.5
            w_val[2] = (
                max(max((w_val[8] + w_val[10]) * 0.25, w_val[2]), 0.25) + w_val[2]
            ) * 0.5
            w_val[5] = (
                max(max((w_val[9] + w_val[11]) * 0.25, w_val[5]), 0.25) + w_val[5]
            ) * 0.5
            w_val[7] = (
                max(max((w_val[10] + w_val[11]) * 0.25, w_val[7]), 0.25) + w_val[7]
            ) * 0.5

            # Laplace loop
            lowthrsum = 0.0
            weightsum = 0.0
            neg_laplace = 0.0

            for pix in range(12):
                lowthr = _n_clamp((13.2 * e[pix + 1] - 0.221), 0.01, 1.0)
                term = w_val[pix] * lowthr
                neg_laplace += (l[pix + 1] * l[pix + 1]) * term
                weightsum += term
                lowthrsum += lowthr

            lowthrsum /= 12.0
            neg_laplace = np.sqrt(neg_laplace / (weightsum + 1e-8))

            # Sharpen value
            sharpen_val = curve_height / (
                curve_height * curveslope * _n_pow_safe(e[0], 3.5) + 0.625
            )

            sharpdiff = (l[0] - neg_laplace) * (lowthrsum * sharpen_val + 0.01)

            # Masking
            min_overshoot = min(abs(L_overshoot), abs(D_overshoot))
            fskip_th = 0.114 * (min_overshoot**0.676) + 3.20e-4

            if abs(sharpdiff) <= fskip_th:
                # No sharpening needed
                padded_img[py, px, 0] = img_r
                padded_img[py, px, 1] = img_g
                padded_img[py, px, 2] = img_b
                continue

            # Sorting luma (Heavy operation)
            # Numba efficient sorting of small array
            l_sorted = np.sort(l)  # Copies and sorts

            nmax = (max(l_sorted[23], l[0]) * 2 + l_sorted[24]) / 3.0
            nmin = (min(l_sorted[1], l[0]) * 2 + l_sorted[0]) / 3.0

            min_dist = min(abs(nmax - l[0]), abs(l[0] - nmin))
            pos_scale = min_dist + L_overshoot
            neg_scale = min_dist + D_overshoot

            pos_scale = min(
                pos_scale, scale_lim * (1 - scale_cs) + pos_scale * scale_cs
            )
            neg_scale = min(
                neg_scale, scale_lim * (1 - scale_cs) + neg_scale * scale_cs
            )

            sharpdiff_pos = max(sharpdiff, 0.0)
            sharpdiff_neg = min(sharpdiff, 0.0)

            term1 = _n_wpmean(
                sharpdiff_pos,
                _n_soft_lim_tanh_approx(sharpdiff_pos, pos_scale),
                cs_L,
                pm_p,
            )
            term2 = _n_wpmean(
                sharpdiff_neg,
                _n_soft_lim_tanh_approx(sharpdiff_neg, neg_scale),
                cs_D,
                pm_p,
            )

            sharpdiff_limited = term1 - term2

            sharpdiff_lim = _n_saturate(l[0] + sharpdiff_limited) - l[0]
            satmul = (l[0] + max(sharpdiff_lim * 0.9, sharpdiff_lim) * 1.03 + 0.03) / (
                l[0] + 0.03
            )

            # Final Result Luma
            res_luma = l[0] + (sharpdiff_lim * 3 + sharpdiff_limited) * 0.25

            # Apply back to RGB
            # res = luma + (orig - luma) * satmul
            # But here res is the NEW luma. The formula in original code:
            # res = luma[0] + ... + (origsat - luma[0]) * satmul
            # Wait, origsat in original code is saturate(img).

            # We modify original image channels directly
            # r_new = res_luma + (img_r - l[0]) * satmul

            padded_img[py, px, 0] = _n_saturate(res_luma + (img_r - l[0]) * satmul)
            padded_img[py, px, 1] = _n_saturate(res_luma + (img_g - l[0]) * satmul)
            padded_img[py, px, 2] = _n_saturate(res_luma + (img_b - l[0]) * satmul)


class AdaptiveSharpenFilterAccurate(BaseFilter):
    """
    AdaptiveSharpen (Numba Accelerated)

    JIT 컴파일을 사용하여 Python 오버헤드를 제거하고 병렬 처리를 수행합니다.
    최초 실행 시 컴파일 시간(약 1~2초)이 소요되지만 이후 매우 빠릅니다.
    """

    def __init__(self):
        super().__init__("AdaptiveSharpen", "적응형 샤프닝")

        self.curve_height = 1.0
        self.curveslope = 0.5
        self.L_overshoot = 0.003
        self.L_compr_low = 0.167
        self.L_compr_high = 0.334
        self.D_overshoot = 0.009
        self.D_compr_low = 0.250
        self.D_compr_high = 0.500
        self.scale_lim = 0.1
        self.scale_cs = 0.056
        self.pm_p = 0.7

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        # Params update
        self.curve_height = params.get("curve_height", self.curve_height)
        self.curveslope = params.get("curveslope", self.curveslope)
        self.L_overshoot = params.get("L_overshoot", self.L_overshoot)
        self.L_compr_low = params.get("L_compr_low", self.L_compr_low)
        self.L_compr_high = params.get("L_compr_high", self.L_compr_high)
        self.D_overshoot = params.get("D_overshoot", self.D_overshoot)
        self.D_compr_low = params.get("D_compr_low", self.D_compr_low)
        self.D_compr_high = params.get("D_compr_high", self.D_compr_high)
        self.scale_lim = params.get("scale_lim", self.scale_lim)
        self.scale_cs = params.get("scale_cs", self.scale_cs)
        self.pm_p = params.get("pm_p", self.pm_p)

        # 0. Prepare Data
        img_float = image.astype(np.float32) / 255.0
        h, w = img_float.shape[:2]

        # 1. Pass 0 Setup (Pad and Execute)
        # Pass 0 needs 2px padding
        pad_0 = 2
        img_padded_0 = np.pad(
            img_float, ((pad_0, pad_0), (pad_0, pad_0), (0, 0)), mode="constant"
        )

        # Outputs for Pass 0
        out_edge = np.zeros((h, w), dtype=np.float32)
        out_luma = np.zeros((h, w), dtype=np.float32)

        # Run JIT Kernel 0
        _run_pass0(img_padded_0, h, w, out_edge, out_luma)

        # 2. Pass 1 Setup
        # Pass 1 needs 3px padding.
        # We need to pad the *Original Image* (for RGB access) and the *Edge/Luma* (from Pass 0)
        pad_1 = 3
        # Padding original image for Pass 1 (Reuse float image, just bigger pad)
        img_padded_1 = np.pad(
            img_float, ((pad_1, pad_1), (pad_1, pad_1), (0, 0)), mode="constant"
        )

        edge_padded = np.pad(out_edge, pad_1, mode="constant")
        luma_padded = np.pad(out_luma, pad_1, mode="constant")

        # Run JIT Kernel 1 (In-place modification of img_padded_1)
        _run_pass1(
            img_padded_1,
            edge_padded,
            luma_padded,
            h,
            w,
            self.curve_height,
            self.curveslope,
            self.L_overshoot,
            self.L_compr_low,
            self.L_compr_high,
            self.D_overshoot,
            self.D_compr_low,
            self.D_compr_high,
            self.scale_lim,
            self.scale_cs,
            self.pm_p,
        )

        # Crop result
        result = img_padded_1[pad_1 : pad_1 + h, pad_1 : pad_1 + w]

        return (np.clip(result, 0, 1) * 255).astype(np.uint8)
