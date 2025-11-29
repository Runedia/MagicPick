import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import saturate


class PD80_ColorSpaceCurves(BaseFilter):
    """
    PD80_03_Color_Space_Curves.fx implementation
    Author: prod80
    Python implementation for Gemini Image Editor
    """

    def __init__(self):
        super().__init__("PD80ColorSpaceCurves", "색공간 커브")

        # Default parameters
        self.color_space = 1  # 0: RGB-W, 1: L*a*b*, 2: HSL, 3: HSV
        self.enable_dither = True
        self.dither_strength = 1.0

        # Curves
        self.pos0_toe_grey = 0.2
        self.pos1_toe_grey = 0.2
        self.pos0_shoulder_grey = 0.8
        self.pos1_shoulder_grey = 0.8

        # Saturation
        self.colorsat = 0.0

        if params:
            for k, v in params.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def _rgb_to_hsl(self, rgb):
        # rgb: (H, W, 3) in [0, 1]
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        mx = np.max(rgb, axis=2)
        mn = np.min(rgb, axis=2)
        df = mx - mn
        h = np.zeros_like(mx)
        s = np.zeros_like(mx)
        l = (mx + mn) / 2

        # Saturation
        # if mx != mn:
        #   if l < 0.5: s = df / (mx + mn)
        #   else:       s = df / (2.0 - mx - mn)

        mask_diff = df > 1e-7

        s[mask_diff] = np.where(
            l[mask_diff] < 0.5,
            df[mask_diff] / (mx[mask_diff] + mn[mask_diff]),
            df[mask_diff] / (2.0 - mx[mask_diff] - mn[mask_diff]),
        )

        # Hue
        # if mx == r: h = (g - b) / df
        # if mx == g: h = 2.0 + (b - r) / df
        # if mx == b: h = 4.0 + (r - g) / df

        mask_r = (mx == r) & mask_diff
        mask_g = (mx == g) & mask_diff
        mask_b = (mx == b) & mask_diff

        h[mask_r] = (g[mask_r] - b[mask_r]) / df[mask_r]
        h[mask_g] = 2.0 + (b[mask_g] - r[mask_g]) / df[mask_g]
        h[mask_b] = 4.0 + (r[mask_b] - g[mask_b]) / df[mask_b]

        h = (h / 6.0) % 1.0

        return np.stack([h, s, l], axis=2)

    def _hsl_to_rgb(self, hsl):
        # hsl: (H, W, 3) in [0, 1]
        h, s, l = hsl[:, :, 0], hsl[:, :, 1], hsl[:, :, 2]

        def hue_to_rgb(p, q, t):
            t = np.where(t < 0, t + 1, t)
            t = np.where(t > 1, t - 1, t)

            cond1 = t < 1 / 6
            cond2 = t < 1 / 2
            cond3 = t < 2 / 3

            return np.where(
                cond1,
                p + (q - p) * 6 * t,
                np.where(cond2, q, np.where(cond3, p + (q - p) * (2 / 3 - t) * 6, p)),
            )

        q = np.where(l < 0.5, l * (1 + s), l + s - l * s)
        p = 2 * l - q

        r = hue_to_rgb(p, q, h + 1 / 3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1 / 3)

        return np.stack([r, g, b], axis=2)

    def _rgb_to_hsv(self, rgb):
        # rgb: (H, W, 3)
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        mx = np.max(rgb, axis=2)
        mn = np.min(rgb, axis=2)
        df = mx - mn

        h = np.zeros_like(mx)
        s = np.zeros_like(mx)
        v = mx

        mask_diff = df > 1e-7
        mask_mx = mx > 0

        s[mask_mx] = df[mask_mx] / mx[mask_mx]

        mask_r = (mx == r) & mask_diff
        mask_g = (mx == g) & mask_diff
        mask_b = (mx == b) & mask_diff

        h[mask_r] = (g[mask_r] - b[mask_r]) / df[mask_r]
        h[mask_g] = 2.0 + (b[mask_g] - r[mask_g]) / df[mask_g]
        h[mask_b] = 4.0 + (r[mask_b] - g[mask_b]) / df[mask_b]

        h = (h / 6.0) % 1.0

        return np.stack([h, s, v], axis=2)

    def _hsv_to_rgb(self, hsv):
        # hsv: (H, W, 3)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        i = (h * 6).astype(int)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)

        i = i % 6

        conditions = [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5]

        r = np.select(conditions, [v, q, p, p, t, v], default=v)
        g = np.select(conditions, [t, v, v, q, p, p], default=p)
        b = np.select(conditions, [p, p, t, v, v, q], default=q)

        return np.stack([r, g, b], axis=2)

    def _srgb_to_lab(self, srgb):
        # srgb -> linear -> xyz -> lab
        # Simplified: using OpenCV if available would be faster, but sticking to pure NumPy for now
        # or mirroring HLSL

        # 1. sRGB to Linear (Approx 2.2 gamma or exact sRGB curve)
        # Using simpler Gamma 2.2 for speed in Python, or exact if critical.
        # Let's use exact sRGB curve for consistency with color gamut filter.

        mask = srgb < 0.04045
        linear = np.where(mask, srgb / 12.92, np.power((srgb + 0.055) / 1.055, 2.4))

        # 2. Linear to XYZ (D65)
        # Matrix
        M = np.array(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ]
        )
        xyz = np.tensordot(linear, M.T, axes=1)

        # 3. XYZ to Lab
        # D65 Ref White
        Xn, Yn, Zn = 0.95047, 1.00000, 1.08883

        x = xyz[:, :, 0] / Xn
        y = xyz[:, :, 1] / Yn
        z = xyz[:, :, 2] / Zn

        # f(t) = t > (6/29)^3 ? t^(1/3) : (1/3)*(29/6)^2 * t + 4/29
        epsilon = 216.0 / 24389.0
        kappa = 24389.0 / 27.0

        mask_x = x > epsilon
        mask_y = y > epsilon
        mask_z = z > epsilon

        fx = np.where(mask_x, np.cbrt(x), (kappa * x + 16.0) / 116.0)
        fy = np.where(mask_y, np.cbrt(y), (kappa * y + 16.0) / 116.0)
        fz = np.where(mask_z, np.cbrt(z), (kappa * z + 16.0) / 116.0)

        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)

        # Normalize L to [0, 1], a, b are usually [-128, 127] approx.
        # But for curves, L is usually processed in [0, 100] or [0, 1].
        # Shader code: `lab_color.x = Tonemap(...)`. Input to tonemap is expected [0, 1] usually.
        # Let's check shader `pd80_srgb_to_lab`.
        # Usually L is 0..100.
        # But if Tonemap is used, it likely expects 0..1 range for L.
        # If shader implementation normalizes L to 0..1, we should too.
        # ReShade standard is often L 0..1.

        return np.stack([L / 100.0, a, b], axis=2)

    def _lab_to_srgb(self, lab):
        L = lab[:, :, 0] * 100.0
        a = lab[:, :, 1]
        b = lab[:, :, 2]

        epsilon = 216.0 / 24389.0
        kappa = 24389.0 / 27.0

        fy = (L + 16.0) / 116.0
        fx = a / 500.0 + fy
        fz = fy - b / 200.0

        fx3 = fx * fx * fx
        fz3 = fz * fz * fz

        xr = np.where(fx3 > epsilon, fx3, (116.0 * fx - 16.0) / kappa)
        yr = np.where(L > kappa * epsilon, np.power((L + 16.0) / 116.0, 3), L / kappa)
        zr = np.where(fz3 > epsilon, fz3, (116.0 * fz - 16.0) / kappa)

        Xn, Yn, Zn = 0.95047, 1.00000, 1.08883

        xyz = np.stack([xr * Xn, yr * Yn, zr * Zn], axis=2)

        # XYZ to Linear
        M_inv = np.array(
            [
                [3.2404542, -1.5371385, -0.4985314],
                [-0.9692660, 1.8760108, 0.0415560],
                [0.0556434, -0.2040259, 1.0572252],
            ]
        )
        linear = np.tensordot(xyz, M_inv.T, axes=1)

        # Linear to sRGB
        mask = linear < 0.0031308
        srgb = np.where(
            mask,
            linear * 12.92,
            1.055 * np.power(np.maximum(linear, 0), 1.0 / 2.4) - 0.055,
        )

        return srgb

    def _prepare_tonemap_params(self, p1, p2, p3):
        # p1: toe start (x, y)
        # p2: shoulder start (x, y)
        # p3: end (1.0, 1.0)

        # Calculate slope between toe and shoulder (linear section)
        denom = p2[0] - p1[0]
        denom = denom if abs(denom) > 1e-5 else 1e-5
        slope = (p2[1] - p1[1]) / denom

        # Mid
        mMid_x = slope
        mMid_y = p1[1] - slope * p1[0]

        # Toe
        denom_toe = p1[1] - slope * p1[0]
        denom_toe = denom_toe if abs(denom_toe) > 1e-5 else 1e-5

        mToe_x = slope * p1[0] ** 2 * p1[1] ** 2 / (denom_toe**2)
        mToe_y = slope * p1[0] ** 2 / denom_toe
        mToe_z = p1[1] ** 2 / denom_toe

        # Shoulder
        denom_sh = slope * (p2[0] - p3[0]) - p2[1] + p3[1]
        denom_sh = denom_sh if abs(denom_sh) > 1e-5 else 1e-5

        mShoulder_x = (
            slope * (p2[0] - p3[0]) ** 2 * (p2[1] - p3[1]) ** 2 / (denom_sh**2)
        )
        mShoulder_y = (
            slope * p2[0] * (p3[0] - p2[0]) + p3[0] * (p2[1] - p3[1])
        ) / denom_sh
        mShoulder_z = (
            -(p2[1] ** 2) + p3[1] * (slope * (p2[0] - p3[0]) + p2[1])
        ) / denom_sh

        return {
            "mToe": (mToe_x, mToe_y, mToe_z),
            "mMid": (mMid_x, mMid_y),
            "mShoulder": (mShoulder_x, mShoulder_y, mShoulder_z),
            "mBx": (p1[0], p2[0]),
        }

    def _tonemap(self, params, x):
        # Vectorized tonemap

        # Toe section
        toe = -params["mToe"][0] / (x + params["mToe"][1]) + params["mToe"][2]

        # Mid section
        mid = params["mMid"][0] * x + params["mMid"][1]

        # Shoulder section
        shoulder = (
            -params["mShoulder"][0] / (x + params["mShoulder"][1])
            + params["mShoulder"][2]
        )

        # Combine
        # result = ( x >= tc.mBx.x ) ? mid : toe;
        # result = ( x >= tc.mBx.y ) ? shoulder : result;

        res = np.where(x >= params["mBx"][0], mid, toe)
        res = np.where(x >= params["mBx"][1], shoulder, res)

        return res

    def apply(self, image, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        img_float = image.astype(np.float32) / 255.0

        # Dither
        if self.enable_dither and self.dither_strength > 0:
            noise = np.random.normal(
                0, 0.001 * self.dither_strength, img_float.shape
            ).astype(np.float32)
            img_float = saturate(img_float + noise)

        # Prepare Curves
        # Ensure boundaries: tx <= sx, ty <= sy
        tx = min(self.pos0_toe_grey, self.pos0_shoulder_grey)
        ty = min(self.pos1_toe_grey, self.pos1_shoulder_grey)
        sx = max(self.pos0_toe_grey, self.pos0_shoulder_grey)
        sy = max(self.pos1_toe_grey, self.pos1_shoulder_grey)

        tc = self._prepare_tonemap_params((tx, ty), (sx, sy), (1.0, 1.0))

        # Color conversion and Tone mapping
        rgb = img_float[:, :, ::-1]  # OpenCV BGR -> RGB

        final_rgb = rgb.copy()

        if self.color_space == 0:  # RGB-W
            rgb_luma = np.min(rgb, axis=2)
            rgb_chroma = rgb - rgb_luma[..., np.newaxis]

            luma_mapped = self._tonemap(tc, rgb_luma)
            chroma_mapped = rgb_chroma * (self.colorsat + 1.0)

            final_rgb = saturate(chroma_mapped + luma_mapped[..., np.newaxis])

        elif self.color_space == 1:  # L*a*b*
            lab = self._srgb_to_lab(rgb)
            L = lab[:, :, 0]

            L_mapped = self._tonemap(tc, L)
            lab[:, :, 0] = L_mapped
            lab[:, :, 1:] *= self.colorsat + 1.0

            final_rgb = self._lab_to_srgb(lab)

        elif self.color_space == 2:  # HSL
            hsl = self._rgb_to_hsl(rgb)
            L = hsl[:, :, 2]

            L_mapped = self._tonemap(tc, L)
            hsl[:, :, 2] = L_mapped
            hsl[:, :, 1] *= self.colorsat + 1.0

            final_rgb = self._hsl_to_rgb(saturate(hsl))

        elif self.color_space == 3:  # HSV
            hsv = self._rgb_to_hsv(rgb)
            V = hsv[:, :, 2]

            V_mapped = self._tonemap(tc, V)
            hsv[:, :, 2] = V_mapped
            hsv[:, :, 1] *= self.colorsat + 1.0

            final_rgb = self._hsv_to_rgb(saturate(hsv))

        # Apply dither again? Shader does it at the end too.
        if self.enable_dither and self.dither_strength > 0:
            noise = np.random.normal(
                0, 0.001 * self.dither_strength, final_rgb.shape
            ).astype(np.float32)
            final_rgb = saturate(final_rgb + noise)

        result_bgr = final_rgb[:, :, ::-1]
        return (result_bgr * 255).astype(np.uint8)
