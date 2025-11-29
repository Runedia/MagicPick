import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import dot, lerp, saturate


class FilmicPassFilter(BaseFilter):
    """
    FilmicPass.fx implementation
    Applies some common color adjustments to mimic a more cinema-like look.
    """

    def __init__(self):
        super().__init__("FilmicPass", "시네마틱 패스")

        # Default parameters
        self.Strength = 0.85
        self.Fade = 0.4
        self.Contrast = 1.0
        self.Linearization = 0.5
        self.Bleach = 0.0
        self.Saturation = -0.15

        self.RedCurve = 1.0
        self.GreenCurve = 1.0
        self.BlueCurve = 1.0
        self.BaseCurve = 1.5

        self.BaseGamma = 1.0
        self.EffectGamma = 0.65
        self.EffectGammaR = 1.0
        self.EffectGammaG = 1.0
        self.EffectGammaB = 1.0

        self.LumCoeff = np.array([0.212656, 0.715158, 0.072186], dtype=np.float32)

    def _calculate_curve(self, value, curve_param):
        """
        Helper for the sigmoid curve calculation:
        val = (1.0 / (1.0 + exp(-curve * (value - 0.5))) - y) / (1.0 - 2.0 * y)
        where y = 1.0 / (1.0 + exp(curve / 2.0))
        """
        # Avoid division by zero or singular points if curve_param is 0
        # If curve_param is very small, it's linear? No, check formula.
        # If curve_param is 0, exp(0)=1, y=0.5. 1-2y = 0. Division by zero.

        # Handle small curve_param by clustering
        curve = np.maximum(curve_param, 1e-5)

        y = 1.0 / (1.0 + np.exp(curve / 2.0))

        # denom = 1.0 - 2.0 * y
        # If curve is small, y -> 0.5, denom -> 0.
        # But reshade slider min is 0.0.

        term1 = 1.0 / (1.0 + np.exp(-curve * (value - 0.5)))
        res = (term1 - y) / (1.0 - 2.0 * y)

        return res

    def apply(self, image, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        # Convert to float
        img_float = image.astype(np.float32) / 255.0

        # OpenCV BGR to RGB
        B = img_float[:, :, ::-1].copy()

        # B = saturate(B)
        B = saturate(B)

        # B = pow(B, Linearization)
        B = np.power(B, self.Linearization)

        # B = lerp(H, B, Contrast) with H = 0.01
        H = 0.01
        B = lerp(H, B, self.Contrast)

        # float A = dot(B.rgb, LumCoeff)
        # float3 D = A
        A = dot(B, self.LumCoeff)
        # D is essentially grayscale version of B (broadcasted later)
        D = A.copy()  # (H, W, 1)

        # B = pow(abs(B), 1.0 / BaseGamma)
        B = np.power(np.abs(B), 1.0 / self.BaseGamma)

        # Curves
        # a, b, c, d = RedCurve, GreenCurve, BlueCurve, BaseCurve
        # y, z, w, v constants

        def get_curve_constants(curve_val):
            curve_val = max(curve_val, 1e-5)
            y = 1.0 / (1.0 + np.exp(curve_val / 2.0))
            denom = 1.0 - 2.0 * y
            return curve_val, y, denom

        a, y_a, denom_a = get_curve_constants(self.RedCurve)
        b, y_b, denom_b = get_curve_constants(self.GreenCurve)
        c, y_c, denom_c = get_curve_constants(self.BlueCurve)
        d, y_d, denom_d = get_curve_constants(self.BaseCurve)

        # D.r calculation (D is single channel A, but logic applies per channel if we treated D as RGB)
        # In shader: float3 D = A; which means D.r = A, D.g = A, D.b = A initially.
        # Then D.r = func(D.r), D.g = func(D.g)...
        # So we calculate 3 different D channels from the same A input.

        D_r = (1.0 / (1.0 + np.exp(-a * (D - 0.5))) - y_a) / denom_a
        D_g = (1.0 / (1.0 + np.exp(-b * (D - 0.5))) - y_b) / denom_b
        D_b = (1.0 / (1.0 + np.exp(-c * (D - 0.5))) - y_c) / denom_c

        # Stack back to 3 channels
        D_vec = np.concatenate([D_r, D_g, D_b], axis=2)

        # D = pow(abs(D), 1.0 / EffectGamma)
        D_vec = np.power(np.abs(D_vec), 1.0 / self.EffectGamma)

        # float3 Di = 1.0 - D
        Di = 1.0 - D_vec

        # D = lerp(D, Di, Bleach)
        D_vec = lerp(D_vec, Di, self.Bleach)

        # D.r = pow(abs(D.r), 1.0 / EffectGammaR) ...
        D_vec[:, :, 0] = np.power(np.abs(D_vec[:, :, 0]), 1.0 / self.EffectGammaR)
        D_vec[:, :, 1] = np.power(np.abs(D_vec[:, :, 1]), 1.0 / self.EffectGammaG)
        D_vec[:, :, 2] = np.power(np.abs(D_vec[:, :, 2]), 1.0 / self.EffectGammaB)

        # Overlay Logic
        # if (D < 0.5) C = (2*D - 1)*(B - B*B) + B
        # else         C = (2*D - 1)*(sqrt(B) - B) + B

        # Vectorized Overlay
        mask = D_vec < 0.5

        # Term common: (2.0 * D - 1.0)
        term_d = 2.0 * D_vec - 1.0

        # Case 1: D < 0.5
        # (2*D - 1) * (B - B*B) + B
        C1 = term_d * (B - B * B) + B

        # Case 2: D >= 0.5
        # (2*D - 1) * (sqrt(B) - B) + B
        C2 = term_d * (np.sqrt(np.maximum(B, 0)) - B) + B

        C = np.where(mask, C1, C2)

        # float3 F = lerp(B, C, Strength)
        F = lerp(B, C, self.Strength)

        # F = (1.0 / (1.0 + exp(-d * (F - 0.5))) - v) / (1.0 - 2.0 * v);
        # Using BaseCurve (d)
        F = (1.0 / (1.0 + np.exp(-d * (F - 0.5))) - y_d) / denom_d

        # Color Matrix (Saturation / Fade)
        r2R = 1.0 - self.Saturation
        g2R = 0.0 + self.Saturation
        b2R = 0.0 + self.Saturation

        r2G = 0.0 + self.Saturation
        g2G = (1.0 - self.Fade) - self.Saturation
        b2G = (0.0 + self.Fade) + self.Saturation

        r2B = 0.0 + self.Saturation
        g2B = (0.0 + self.Fade) + self.Saturation
        b2B = (1.0 - self.Fade) - self.Saturation

        # iF = F
        iF = F.copy()

        F_r = iF[:, :, 0] * r2R + iF[:, :, 1] * g2R + iF[:, :, 2] * b2R
        F_g = iF[:, :, 0] * r2G + iF[:, :, 1] * g2G + iF[:, :, 2] * b2G
        F_b = iF[:, :, 0] * r2B + iF[:, :, 1] * g2B + iF[:, :, 2] * b2B

        F = np.stack([F_r, F_g, F_b], axis=2)

        # float N = dot(F.rgb, LumCoeff)
        N = dot(F, self.LumCoeff)  # (H, W, 1)

        # float3 Cn = F;
        # Overlay again with N
        # if (N < 0.5) ...

        mask_n = N < 0.5
        term_n = 2.0 * N - 1.0

        # Case 1: N < 0.5 -> (2*N - 1)*(F - F*F) + F
        Cn1 = term_n * (F - F * F) + F

        # Case 2: N >= 0.5 -> (2*N - 1)*(sqrt(F) - F) + F
        Cn2 = term_n * (np.sqrt(np.maximum(F, 0)) - F) + F

        Cn = np.where(mask_n, Cn1, Cn2)

        # Cn = pow(max(Cn,0), 1.0 / Linearization)
        Cn = np.power(np.maximum(Cn, 0), 1.0 / self.Linearization)

        # float3 Fn = lerp(B, Cn, Strength)
        # Here B is the one from earlier step: B = pow(abs(B), 1.0 / BaseGamma)
        # As analyzed, we use the current state of B.

        Fn = lerp(B, Cn, self.Strength)

        # Clip and convert back to BGR
        Fn = saturate(Fn)
        result_bgr = Fn[:, :, ::-1]

        return (result_bgr * 255).astype(np.uint8)
