import cv2
import numpy as np

from filters.base_filter import BaseFilter


class SmartSharpFilter(BaseFilter):
    """
    Smart_Sharp.fx 구현

    Depth Based Unsharp Mask Bilateral Contrast Adaptive Sharpening
    Bilateral Filter를 사용하여 엣지를 보존하면서 블러링한 후(Low Variance Blur),
    원본과의 차이를 이용하여 샤프닝을 수행합니다.
    """

    def __init__(self):
        super().__init__("SmartSharp", "스마트 샤프닝")
        self.sharpness = 0.625
        self.b_grounding = 0.0  # Coarseness (0.0 - 1.0)
        self.d_sigma = 0.25  # Bilateral Sigma Color (BSIGMA)
        self.s_sigma = 15.0  # Bilateral Sigma Space (SIGMA) (Not directly used in cv2 as sigmaSpace, maybe scale?)

        # Advanced options
        self.ca_mask_boost = False
        self.ca_removal = False
        self.clamp_sharp = True

    def _rgb_to_ycbcr(self, img):
        # ReShade RGBtoYCbCr
        # Y  =  .299 * rgb.x + .587 * rgb.y + .114 * rgb.z;
        # Cb = -.169 * rgb.x - .331 * rgb.y + .500 * rgb.z;
        # Cr =  .500 * rgb.x - .419 * rgb.y - .081 * rgb.z;
        # return float3(Y,Cb + 128./255.,Cr + 128./255.);

        M = np.array(
            [[0.299, 0.587, 0.114], [-0.169, -0.331, 0.500], [0.500, -0.419, -0.081]]
        )

        ycbcr = np.dot(img, M.T)
        ycbcr[:, :, 1:] += 128.0 / 255.0
        return ycbcr

    def _ycbcr_to_rgb(self, img):
        # ReShade YCbCrtoRGB
        # float3 c = ycc - float3(0., 128./255., 128./255.);
        # float R = c.x + 1.400 * c.z;
        # float G = c.x - 0.343 * c.y - 0.711 * c.z;
        # float B = c.x + 1.765 * c.y;

        c = img.copy()
        c[:, :, 1:] -= 128.0 / 255.0

        M_inv = np.array([[1.0, 0.0, 1.400], [1.0, -0.343, -0.711], [1.0, 1.765, 0.0]])

        return np.dot(c, M_inv.T)

    def _get_luma(self, rgb):
        return np.dot(rgb, [0.2126, 0.7152, 0.0722])

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        if "sharpness" in params:
            self.sharpness = float(params["sharpness"])
        if "b_grounding" in params:
            self.b_grounding = float(params["b_grounding"])

        # Setup Bilateral Filter parameters
        # Kernel size d=5 (Quality 2 default MSIZE=5)
        d = 5

        # Sigma Color
        sigma_color = (
            self.d_sigma
        )  # 0.25 range 0-1? OpenCV uses 0-255 range usually for 8bit?
        # If input is float 0-1, sigmaColor should be small.
        # OpenCV bilateralFilter: if src is float32, sigmaColor is in value domain.

        # Sigma Space
        # Shader uses SIGMA=15, but also scales offset by rcp(kSize * GT()).
        # GT() = lerp(1.0, 0.5, B_Grounding)
        # If B_Grounding = 0, GT=1. Offset scale = 1/2.
        # If B_Grounding = 1, GT=0.5. Offset scale = 1/1 = 1. (Wider)

        # We can adjust sigmaSpace to mimic this.
        # Standard sigmaSpace for d=5 is usually around d.
        sigma_space = self.s_sigma * (1.0 + self.b_grounding)  # Heuristic

        img_float = image.astype(np.float32) / 255.0

        # Bilateral Filter
        # cv2.bilateralFilter expects float32 image.
        lvb = cv2.bilateralFilter(
            img_float, d, sigmaColor=sigma_color, sigmaSpace=sigma_space
        )

        # CAM (Contrast Adaptive Masking)
        # mnRGB = min( min( LI(c), LI(final_color)), LI(cc));
        # mxRGB = max( max( LI(c), LI(final_color)), LI(cc));
        # Ideally we should sample neighborhood min/max Luma.
        # For simplicity, we can use min/max of (Original, LVB).
        # Or implement a local min/max filter.
        # Let's use min/max of Original vs LVB for performance, effectively 1x1 neighborhood + blur.
        # Actually, true CAS uses 3x3 neighbors.
        # Smart Sharp shader uses 'cc' in the loop to find min/max.
        # This means min/max over the kernel window.

        # Approximating min/max over window using dilate/erode on Luma
        luma_src = self._get_luma(img_float)
        # luma_lvb = self._get_luma(lvb)

        kernel = np.ones((3, 3), np.uint8)
        luma_min = cv2.erode(luma_src, kernel)
        luma_max = cv2.dilate(luma_src, kernel)

        # Include LVB in min/max?
        # The shader includes 'final_color' (LVB) in min/max calculation.
        luma_lvb = self._get_luma(lvb)
        luma_min = np.minimum(luma_min, luma_lvb)
        luma_max = np.maximum(luma_max, luma_lvb)

        # Calculate CAS Mask
        # rcpMRGB = rcp(mxRGB), RGB_D = saturate(min(mnRGB, 1.0 - mxRGB) * rcpMRGB);
        rcp_mx = 1.0 / np.maximum(luma_max, 1e-6)
        rgb_d = np.clip(np.minimum(luma_min, 1.0 - luma_max) * rcp_mx, 0.0, 1.0)

        cas_mask = rgb_d

        # CA Boost
        if self.ca_mask_boost:
            # CAS_Mask = lerp(CAS_Mask,CAS_Mask * CAS_Mask,saturate(Sharp * 0.5));
            sharp_scaled = min(self.sharpness * 0.5, 1.0)
            cas_mask = (
                cas_mask * (1.0 - sharp_scaled) + (cas_mask * cas_mask) * sharp_scaled
            )

        if self.ca_removal:
            cas_mask = 1.0

        # Sharpening
        # Sharpen = RGBtoYCbCr(tex2D(BackBuffer,texcoord).rgb - LVB);
        diff = img_float - lvb
        diff_ycbcr = self._rgb_to_ycbcr(
            diff
        )  # Wait, diff can be negative. RGBtoYCbCr usually expects 0-1.

        # Shader logic: RGBtoYCbCr( rgb - lvb )
        # Y = .299*R + ...
        # Cb = ... + 0.5
        # This +0.5 offset in CbCr is for centering 0 at 0.5.
        # If we pass difference, we might want delta YCbCr without offset?
        # Shader `RGBtoYCbCr` adds 128/255.
        # `Sharpen.x *= Sharpen_Power` (Y channel)
        # `Sharpen = YCbCrtoRGB(Sharpen)` subtracts 128/255.
        # So the offset cancels out for Cb/Cr, but for Y it's just linear scale.
        # Actually, if input is difference, R,G,B can be negative.
        # Y will be difference in Luma.
        # Cb/Cr will be difference + 0.5.
        # Scaling Y (Luma diff) works.
        # Scaling Cb/Cr? No, only Sharpen.x (Y) is scaled.
        # So Cb/Cr difference is preserved (offset included).

        # Let's follow shader exactly.
        # But my _rgb_to_ycbcr adds offset.
        # Img A: RGB -> YCbCr (0.5 offset)
        # Img B: RGB -> YCbCr (0.5 offset)
        # Diff: A - B -> Y diff, Cb diff, Cr diff.
        # Shader computes RGB diff FIRST, then converts.
        # Diff RGB -> YCbCr.
        # Y = dot(DiffRGB, Coeff) (Delta Luma)
        # Cb = dot(DiffRGB, Coeff) + 0.5
        # Scale Y.
        # Convert back:
        # R = scaledY + 1.4*(Cb - 0.5) ...
        # Cb - 0.5 removes the offset we added.
        # So it works fine.

        sharpen_ycbcr = self._rgb_to_ycbcr(diff)

        # Apply Mask and Power
        # Sharpen_Power = Sharpness
        # Out = lerp(Color, Sharpen, CAM * saturate(Sharpen_Power)) ?
        # No, Sharpen_Out func returns:
        # Sharpen.rgb = color.rgb + Sharpen;
        # where Sharpen is the scaled delta.
        # And mask is applied?
        # `return float4(lerp(..., Sharpen, CAM * saturate(Sharpen_Power)), ...)`?
        # Wait, `Sharpen` in return statement is `color.rgb + Sharpen_Delta`.
        # So `lerp(color, color+delta, mask)` = `color + delta * mask`.

        sharpen_power = (
            self.sharpness * 3.1
        )  # Shader multiplies by 3.1 inside Sharpen_Out

        # Scale Y
        sharpen_ycbcr[:, :, 0] *= sharpen_power

        # Convert back to RGB delta
        sharpen_delta_rgb = self._ycbcr_to_rgb(sharpen_ycbcr)

        # Apply mask
        # CAM is cas_mask

        # Delta * Mask
        final_delta = sharpen_delta_rgb * np.expand_dims(cas_mask, axis=2)

        result = img_float + final_delta

        if self.clamp_sharp:
            result = np.clip(result, 0.0, 1.0)

        return (result * 255).astype(np.uint8)
