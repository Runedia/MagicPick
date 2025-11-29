import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import dot, lerp, saturate


def get_lum(x):
    return dot(x, np.array([0.212656, 0.715158, 0.072186]))


def sat(res, x):
    # x range [-1, 1]
    lum = get_lum(res)
    lum = lum[:, :, np.newaxis]
    return saturate(lerp(lum, res, x + 1.0))


def vib(res, x):
    # x range [-1, 1]
    min_v = np.min(res, axis=2)
    max_v = np.max(res, axis=2)
    chroma = max_v - min_v
    lum = get_lum(res)
    lum = lum[:, :, np.newaxis]
    chroma = chroma[:, :, np.newaxis]

    return saturate(lerp(lum, res, 1.0 + (x * (1.0 - chroma))))


def adjust_color(scale, colorvalue, adjust, bk, method):
    """
    scale: calculated scale factor (sRGB, sCMY, etc) (H, W)
    colorvalue: current channel value (H, W)
    adjust: adjustment amount (float)
    bk: black adjustment (float)
    method: 0 (Absolute) or 1 (Relative)
    """
    # method comes in as 0 or 1
    # In shader: ( 1.0f - colorvalue * method )
    # If method=0 (Abs): 1.0
    # If method=1 (Rel): 1.0 - colorvalue

    factor = 1.0 - colorvalue * float(method)
    term = ((-1.0 - adjust) * bk - adjust) * factor

    # clamp(term, -colorvalue, 1.0 - colorvalue) * scale

    # We need to broadcast shapes
    if np.isscalar(adjust):
        # adjust and bk are scalars
        pass

    clamped = np.clip(term, -colorvalue, 1.0 - colorvalue)
    return clamped * scale


class PD80_SelectiveColorFilter(BaseFilter):
    """
    PD80 Selective Color 필터

    특정 색상 범위(빨강, 노랑, 초록, 청록, 파랑, 자홍, 흰색, 중성색, 검은색)를
    선택적으로 조정합니다.
    """

    def __init__(self):
        super().__init__("PD80_SelectiveColor", "PD80 선택적 색상 보정")
        self.corr_method = 1  # 0: Absolute, 1: Relative
        self.corr_method2 = 1  # Saturation method

        # Define all parameters
        # Red, Yellow, Green, Cyan, Blue, Magenta, White, Neutral, Black
        # Each has cya, mag, yel, bla, sat, vib
        colors = ["r", "y", "g", "c", "b", "m", "w", "n", "bk"]
        attribs = ["cya", "mag", "yel", "bla", "sat", "vib"]

        for c in colors:
            for a in attribs:
                setattr(self, f"{c}_adj_{a}", 0.0)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        # Update parameters
        self.corr_method = params.get("corr_method", self.corr_method)
        self.corr_method2 = params.get("corr_method2", self.corr_method2)

        colors = ["r", "y", "g", "c", "b", "m", "w", "n", "bk"]
        attribs = ["cya", "mag", "yel", "bla", "sat", "vib"]
        for c in colors:
            for a in attribs:
                key = f"{c}_adj_{a}"
                setattr(self, key, params.get(key, getattr(self, key)))

        img_float = image.astype(np.float32) / 255.0

        # Calculate min, max, mid
        min_val = np.min(img_float, axis=2)
        max_val = np.max(img_float, axis=2)
        sum_val = np.sum(img_float, axis=2)
        mid_val = sum_val - min_val - max_val

        orig_r = img_float[:, :, 0]
        orig_g = img_float[:, :, 1]
        orig_b = img_float[:, :, 2]

        # Scales
        sRGB = max_val - mid_val
        sCMY = mid_val - min_val
        sNeutrals = 1.0 - (np.abs(max_val - 0.5) + np.abs(min_val - 0.5))
        sWhites = (min_val - 0.5) * 2.0
        sBlacks = (0.5 - max_val) * 2.0

        # Relative saturation deltas
        r_d_m = orig_r - orig_b
        r_d_y = orig_r - orig_g
        y_d = mid_val - orig_b
        g_d_y = orig_g - orig_r
        g_d_c = orig_g - orig_b
        c_d = mid_val - orig_r
        b_d_c = orig_b - orig_g
        b_d_m = orig_b - orig_r
        m_d = mid_val - orig_g

        # Initialize deltas to 1.0
        r_delta = np.ones_like(min_val)
        y_delta = np.ones_like(min_val)
        g_delta = np.ones_like(min_val)
        c_delta = np.ones_like(min_val)
        b_delta = np.ones_like(min_val)
        m_delta = np.ones_like(min_val)

        if self.corr_method2:
            r_delta = np.minimum(r_d_m, r_d_y)
            y_delta = y_d
            g_delta = np.minimum(g_d_y, g_d_c)
            c_delta = c_d
            b_delta = np.minimum(b_d_c, b_d_m)
            m_delta = m_d

        # Helper to apply adjustments
        def apply_group(mask, scale, adj_c, adj_m, adj_y, adj_k, adj_s, adj_v, delta):
            if not np.any(mask):
                return

            # Extract masked values
            masked_scale = scale[mask]
            masked_img = img_float[mask]
            masked_delta = delta[mask]

            # Adjust RGB channels
            # Cyan adj affects Red channel?
            # In CMYK, Cyan controls Red ink? No, Cyan removes Red.
            # In shader: color.x = color.x + adjustcolor(..., adj_c, adj_k, ...)
            # color.x is Red. So adj_c controls Red.
            # Wait, standard selective color: "Cyan" slider changes Cyan component.
            # Changing Cyan component in Red color means changing amount of Cyan ink?
            # In RGB, Red = 1 - Cyan.
            # The shader implementation:
            # color.x (Red) += adjustcolor(..., adj_c, ...)
            # So increasing "Cyan" slider adds to Red channel?
            # Let's check `adjustcolor` sign.
            # `term = ((-1.0 - adjust) * bk - adjust)`
            # If adjust > 0 (adding Cyan), term becomes negative.
            # Adding negative to Red channel -> Reduces Red -> Increases Cyan. Correct.

            c_adj = adjust_color(
                masked_scale, masked_img[:, 0], adj_c, adj_k, self.corr_method
            )
            m_adj = adjust_color(
                masked_scale, masked_img[:, 1], adj_m, adj_k, self.corr_method
            )
            y_adj = adjust_color(
                masked_scale, masked_img[:, 2], adj_y, adj_k, self.corr_method
            )

            masked_img[:, 0] += c_adj
            masked_img[:, 1] += m_adj
            masked_img[:, 2] += y_adj

            # Sat / Vib
            masked_img = sat(masked_img, adj_s * masked_delta)
            masked_img = vib(masked_img, adj_v * masked_delta)

            img_float[mask] = masked_img

        # Apply groups
        # Red: max == r
        apply_group(
            max_val == orig_r,
            sRGB,
            self.r_adj_cya,
            self.r_adj_mag,
            self.r_adj_yel,
            self.r_adj_bla,
            self.r_adj_sat,
            self.r_adj_vib,
            r_delta,
        )

        # Yellow: min == b
        apply_group(
            min_val == orig_b,
            sCMY,
            self.y_adj_cya,
            self.y_adj_mag,
            self.y_adj_yel,
            self.y_adj_bla,
            self.y_adj_sat,
            self.y_adj_vib,
            y_delta,
        )

        # Green: max == g
        apply_group(
            max_val == orig_g,
            sRGB,
            self.g_adj_cya,
            self.g_adj_mag,
            self.g_adj_yel,
            self.g_adj_bla,
            self.g_adj_sat,
            self.g_adj_vib,
            g_delta,
        )

        # Cyan: min == r
        apply_group(
            min_val == orig_r,
            sCMY,
            self.c_adj_cya,
            self.c_adj_mag,
            self.c_adj_yel,
            self.c_adj_bla,
            self.c_adj_sat,
            self.c_adj_vib,
            c_delta,
        )

        # Blue: max == b
        apply_group(
            max_val == orig_b,
            sRGB,
            self.b_adj_cya,
            self.b_adj_mag,
            self.b_adj_yel,
            self.b_adj_bla,
            self.b_adj_sat,
            self.b_adj_vib,
            b_delta,
        )

        # Magenta: min == g
        apply_group(
            min_val == orig_g,
            sCMY,
            self.m_adj_cya,
            self.m_adj_mag,
            self.m_adj_yel,
            self.m_adj_bla,
            self.m_adj_sat,
            self.m_adj_vib,
            m_delta,
        )

        # Whites: min >= 0.5
        # Scale factor is sWhites, but also smoothstep(0.5, 1.0, min_value) for sat/vib
        mask_w = min_val >= 0.5
        if np.any(mask_w):
            # Special handling because sat/vib scaling is different
            masked_scale = sWhites[mask_w]
            masked_img = img_float[mask_w]
            min_v_masked = min_val[mask_w]

            c_adj = adjust_color(
                masked_scale,
                masked_img[:, 0],
                self.w_adj_cya,
                self.w_adj_bla,
                self.corr_method,
            )
            m_adj = adjust_color(
                masked_scale,
                masked_img[:, 1],
                self.w_adj_mag,
                self.w_adj_bla,
                self.corr_method,
            )
            y_adj = adjust_color(
                masked_scale,
                masked_img[:, 2],
                self.w_adj_yel,
                self.w_adj_bla,
                self.corr_method,
            )

            masked_img[:, 0] += c_adj
            masked_img[:, 1] += m_adj
            masked_img[:, 2] += y_adj

            # smoothstep(0.5, 1.0, min_value)
            t = saturate((min_v_masked - 0.5) / (1.0 - 0.5))
            ss = t * t * (3.0 - 2.0 * t)

            masked_img = sat(masked_img, self.w_adj_sat * ss)
            masked_img = vib(masked_img, self.w_adj_vib * ss)

            img_float[mask_w] = masked_img

        # Neutrals: max > 0.0 && min < 1.0
        mask_n = (max_val > 0.0) & (min_val < 1.0)
        if np.any(mask_n):
            # Simple group application logic works here but without delta?
            # Shader uses sat(..., n_adj_sat) directly (no delta or smoothstep)
            masked_scale = sNeutrals[mask_n]
            masked_img = img_float[mask_n]

            c_adj = adjust_color(
                masked_scale,
                masked_img[:, 0],
                self.n_adj_cya,
                self.n_adj_bla,
                self.corr_method,
            )
            m_adj = adjust_color(
                masked_scale,
                masked_img[:, 1],
                self.n_adj_mag,
                self.n_adj_bla,
                self.corr_method,
            )
            y_adj = adjust_color(
                masked_scale,
                masked_img[:, 2],
                self.n_adj_yel,
                self.n_adj_bla,
                self.corr_method,
            )

            masked_img[:, 0] += c_adj
            masked_img[:, 1] += m_adj
            masked_img[:, 2] += y_adj

            masked_img = sat(masked_img, self.n_adj_sat)
            masked_img = vib(masked_img, self.n_adj_vib)

            img_float[mask_n] = masked_img

        # Blacks: max < 0.5
        mask_bk = max_val < 0.5
        if np.any(mask_bk):
            masked_scale = sBlacks[mask_bk]
            masked_img = img_float[mask_bk]
            max_v_masked = max_val[mask_bk]

            c_adj = adjust_color(
                masked_scale,
                masked_img[:, 0],
                self.bk_adj_cya,
                self.bk_adj_bla,
                self.corr_method,
            )
            m_adj = adjust_color(
                masked_scale,
                masked_img[:, 1],
                self.bk_adj_mag,
                self.bk_adj_bla,
                self.corr_method,
            )
            y_adj = adjust_color(
                masked_scale,
                masked_img[:, 2],
                self.bk_adj_yel,
                self.bk_adj_bla,
                self.corr_method,
            )

            masked_img[:, 0] += c_adj
            masked_img[:, 1] += m_adj
            masked_img[:, 2] += y_adj

            # smoothstep(0.5, 0.0, max_value) = smoothstep(edge0=0.5, edge1=0.0, x=max_value)
            # smoothstep implementation: t = clamp((x - edge0) / (edge1 - edge0))
            t = saturate((max_v_masked - 0.5) / (0.0 - 0.5))
            ss = t * t * (3.0 - 2.0 * t)

            masked_img = sat(masked_img, self.bk_adj_sat * ss)
            masked_img = vib(masked_img, self.bk_adj_vib * ss)

            img_float[mask_bk] = masked_img

        return (saturate(img_float) * 255).astype(np.uint8)
