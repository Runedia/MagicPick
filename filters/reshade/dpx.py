import numpy as np
from numba import njit, prange

from filters.base_filter import BaseFilter
from filters.reshade.numba_helpers import lerp, saturate


@njit(parallel=True, fastmath=True, cache=True)
def _dpx_kernel(
    img,
    h,
    w,
    rgb_curve,
    rgb_c,
    contrast,
    saturation,
    colorfulness,
    strength,
    curve_scale,
    curve_offset,
    xyz_mat,
    rgb_mat,
):
    """
    DPX 필터 핵심 커널
    모든 중간 배열 생성을 제거하고 픽셀당 한 번의 패스만 수행
    """

    # 출력 버퍼
    out = np.empty((h, w, 3), dtype=np.uint8)

    # 상수 사전 계산 (Loop Invariant)
    inv_colorfulness = 1.0 / colorfulness

    # Contrast Pre-calc
    contrast_mul = 1.0 - contrast
    contrast_add = 0.5 * contrast

    # Luma Coefs (from original code)
    lw_r, lw_g, lw_b = 0.30, 0.59, 0.11

    # Unpack Curve Params (Vector to Scalar for register usage)
    curve_r, curve_g, curve_b = rgb_curve[0], rgb_curve[1], rgb_curve[2]
    c_r, c_g, c_b = rgb_c[0], rgb_c[1], rgb_c[2]
    scale_r, scale_g, scale_b = curve_scale[0], curve_scale[1], curve_scale[2]
    off_r, off_g, off_b = curve_offset[0], curve_offset[1], curve_offset[2]

    # Unpack Matrix (XYZ) - Transposed in original code logic
    # Original: dot(v, M.T) -> Row vector multiplication
    xyz_00, xyz_01, xyz_02 = xyz_mat[0, 0], xyz_mat[0, 1], xyz_mat[0, 2]
    xyz_10, xyz_11, xyz_12 = xyz_mat[1, 0], xyz_mat[1, 1], xyz_mat[1, 2]
    xyz_20, xyz_21, xyz_22 = xyz_mat[2, 0], xyz_mat[2, 1], xyz_mat[2, 2]

    # Unpack Matrix (RGB) - Transposed
    rgb_00, rgb_01, rgb_02 = rgb_mat[0, 0], rgb_mat[0, 1], rgb_mat[0, 2]
    rgb_10, rgb_11, rgb_12 = rgb_mat[1, 0], rgb_mat[1, 1], rgb_mat[1, 2]
    rgb_20, rgb_21, rgb_22 = rgb_mat[2, 0], rgb_mat[2, 1], rgb_mat[2, 2]

    for y in prange(h):
        for x in range(w):
            # 1. Load & Normalize
            r_in = img[y, x, 0] * (1.0 / 255.0)
            g_in = img[y, x, 1] * (1.0 / 255.0)
            b_in = img[y, x, 2] * (1.0 / 255.0)

            # 2. Contrast
            b_r = r_in * contrast_mul + contrast_add
            b_g = g_in * contrast_mul + contrast_add
            b_b = b_in * contrast_mul + contrast_add

            # 3. Curve (Sigmoid-like)
            # term1 = 1.0 / (1.0 + exp(-curve * (B - c)))
            # B = (term1 / scale) + offset
            # fastmath=True handles exp efficiently
            term_r = 1.0 / (1.0 + np.exp(-curve_r * (b_r - c_r)))
            b_r = (term_r / scale_r) + off_r

            term_g = 1.0 / (1.0 + np.exp(-curve_g * (b_g - c_g)))
            b_g = (term_g / scale_g) + off_g

            term_b = 1.0 / (1.0 + np.exp(-curve_b * (b_b - c_b)))
            b_b = (term_b / scale_b) + off_b

            # 4. Colorfulness
            # value = max(r, g, b)
            val = max(b_r, max(b_g, b_b))

            # Div by zero check
            if val < 1e-8:
                c0_r, c0_g, c0_b = 0.0, 0.0, 0.0
            else:
                # color = B / value
                col_r = b_r / val
                col_g = b_g / val
                col_b = b_b / val

                # color = pow(color, 1/colorfulness)
                col_r = col_r**inv_colorfulness
                col_g = col_g**inv_colorfulness
                col_b = col_b**inv_colorfulness

                # c0 = color * value
                c0_r = col_r * val
                c0_g = col_g * val
                c0_b = col_b * val

            # 5. Matrix Transform 1 (RGB -> XYZ equivalent)
            # dot(c0, XYZ_matrix.T)
            trans_r = c0_r * xyz_00 + c0_g * xyz_01 + c0_b * xyz_02
            trans_g = c0_r * xyz_10 + c0_g * xyz_11 + c0_b * xyz_12
            trans_b = c0_r * xyz_20 + c0_g * xyz_21 + c0_b * xyz_22

            # 6. Luma & Saturation
            luma = trans_r * lw_r + trans_g * lw_g + trans_b * lw_b

            # lerp(luma, trans, saturation)
            trans_r = luma + saturation * (trans_r - luma)
            trans_g = luma + saturation * (trans_g - luma)
            trans_b = luma + saturation * (trans_b - luma)

            # 7. Matrix Transform 2 (XYZ -> RGB equivalent)
            # dot(trans, RGB_matrix.T)
            res_r = trans_r * rgb_00 + trans_g * rgb_01 + trans_b * rgb_02
            res_g = trans_r * rgb_10 + trans_g * rgb_11 + trans_b * rgb_12
            res_b = trans_r * rgb_20 + trans_g * rgb_21 + trans_b * rgb_22

            # 8. Blend with Original
            final_r = lerp(r_in, res_r, strength)
            final_g = lerp(g_in, res_g, strength)
            final_b = lerp(b_in, res_b, strength)

            # 9. Store
            out[y, x, 0] = int(saturate(final_r) * 255)
            out[y, x, 1] = int(saturate(final_g) * 255)
            out[y, x, 2] = int(saturate(final_b) * 255)

    return out


class DPXFilterAccurate(BaseFilter):
    """
    DPX/Cineon 시네마틱 색상 그레이딩 (Numba 최적화)

    Cineon Log 변환 및 매트릭스 연산을 Numba JIT를 통해
    단일 패스로 처리하여 CPU 병목을 제거했습니다.
    """

    def __init__(self):
        super().__init__("DPX", "DPX (Cineon Look)")

        self.rgb_curve = np.array([8.0, 8.0, 8.0], dtype=np.float32)
        self.rgb_c = np.array([0.36, 0.36, 0.34], dtype=np.float32)
        self.contrast = 0.1
        self.saturation = 3.0
        self.colorfulness = 2.5
        self.strength = 0.20

        self.RGB_matrix = np.array(
            [
                [2.6714711726599600, -1.2672360578624100, -0.4109956021722270],
                [-1.0251070293466400, 1.9840911624108900, 0.0439502493584124],
                [0.0610009456429445, -0.2236707508128630, 1.1590210416706100],
            ],
            dtype=np.float32,
        )

        self.XYZ_matrix = np.array(
            [
                [0.5003033835433160, 0.3380975732227390, 0.1645897795458570],
                [0.2579688942747580, 0.6761952591447060, 0.0658358459823868],
                [0.0234517888692628, 0.1126992737203000, 0.8668396731242010],
            ],
            dtype=np.float32,
        )

    def warmup(self):
        """JIT 컴파일 유도"""
        print(f"[{self.name}] Warm-up started...")
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        self.apply(dummy)
        print(f"[{self.name}] Warm-up completed.")

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        # 파라미터 추출
        rgb_curve = np.array(
            params.get("RGB_Curve", tuple(self.rgb_curve)), dtype=np.float32
        )
        rgb_c = np.array(params.get("RGB_C", tuple(self.rgb_c)), dtype=np.float32)

        contrast = float(params.get("Contrast", self.contrast))
        saturation = float(params.get("Saturation", self.saturation))
        colorfulness = float(params.get("Colorfulness", self.colorfulness))
        strength = float(params.get("Strength", self.strength))

        # --- 상수 Pre-calculation (Python 단에서 처리) ---
        # 반복문 내부의 부하를 줄이기 위해 커브 관련 상수를 미리 계산하여 전달
        # B_temp = 1.0 / (1.0 + np.exp(rgb_curve / 2.0))
        # scale = -2.0 * B_temp + 1.0
        # offset = -B_temp / scale

        b_temp = 1.0 / (1.0 + np.exp(rgb_curve / 2.0))
        curve_scale = -2.0 * b_temp + 1.0
        curve_offset = -b_temp / (curve_scale + 1e-8)  # 0나누기 방지

        h, w = image.shape[:2]

        # JIT 커널 실행
        result = _dpx_kernel(
            image,
            h,
            w,
            rgb_curve,
            rgb_c,
            contrast,
            saturation,
            colorfulness,
            strength,
            curve_scale,
            curve_offset,
            self.XYZ_matrix,
            self.RGB_matrix,
        )

        return result
