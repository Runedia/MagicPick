import cv2
import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import saturate


class PD80SharpeningFilter(BaseFilter):
    """
    PD80_05_Sharpening.fx 구현

    Luma Sharpening (Unsharp Mask 변형) 필터입니다.
    가우시안 블러를 이용해 엣지를 추출하고 선명도를 향상시킵니다.
    """

    def __init__(self):
        super().__init__("PD80Sharpening", "PD80 샤프닝")
        self.blur_sigma = 0.45
        self.sharpening = 1.7
        self.threshold = 0.0
        self.limiter = 0.03
        self.show_edges = False

    def _get_avg_color(self, col):
        # dot( col.xyz, float3( 0.333333f, 0.333334f, 0.333333f ));
        # Assuming col is (H, W, 3) or (3,)
        return np.dot(col, np.array([0.333333, 0.333334, 0.333333], dtype=np.float32))

    def _clip_color(self, color):
        lum = self._get_avg_color(color)  # shape (H, W)

        # color is (H, W, 3)
        # min/max per pixel
        min_col = np.min(color, axis=2)
        max_col = np.max(color, axis=2)

        # ( mincol < 0.0f ) ? lum + (( color.xyz - lum ) * lum ) / ( lum - mincol ) : color.xyz;
        # Note: lum, min_col are 2D, color is 3D. Need broadcasting.

        # Broadcasting preparation
        lum_3d = np.expand_dims(lum, axis=2)
        min_col_3d = np.expand_dims(min_col, axis=2)
        max_col_3d = np.expand_dims(max_col, axis=2)

        # Case 1: min < 0
        mask_min = min_col < 0.0
        denom_min = lum - min_col
        # Avoid division by zero (though if min < 0 and lum >= 0, denom > 0)
        denom_min = np.maximum(denom_min, 1e-6)

        res_min = lum_3d + ((color - lum_3d) * lum_3d) / np.expand_dims(
            denom_min, axis=2
        )

        # Apply mask
        color = np.where(np.expand_dims(mask_min, axis=2), res_min, color)

        # Case 2: max > 1
        mask_max = max_col > 1.0
        denom_max = max_col - lum
        denom_max = np.maximum(denom_max, 1e-6)

        res_max = lum_3d + ((color - lum_3d) * (1.0 - lum_3d)) / np.expand_dims(
            denom_max, axis=2
        )

        # Apply mask
        color = np.where(np.expand_dims(mask_max, axis=2), res_max, color)

        return color

    def _blend_luma(self, base, blend):
        lum_base = self._get_avg_color(base)
        lum_blend = self._get_avg_color(blend)
        l_diff = lum_blend - lum_base

        col = base + np.expand_dims(l_diff, axis=2)
        return self._clip_color(col)

    def _screen(self, c, b):
        return 1.0 - (1.0 - c) * (1.0 - b)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        if "blur_sigma" in params:
            self.blur_sigma = float(params["blur_sigma"])
        if "sharpening" in params:
            self.sharpening = float(params["sharpening"])
        if "threshold" in params:
            self.threshold = float(params["threshold"])
        if "limiter" in params:
            self.limiter = float(params["limiter"])
        if "show_edges" in params:
            self.show_edges = bool(params["show_edges"])

        img_float = image.astype(np.float32) / 255.0
        height, width = img_float.shape[:2]

        # Gaussian Math
        # float bSigma = BlurSigma * ( max( BUFFER_WIDTH, BUFFER_HEIGHT ) / 1920.0f );
        b_sigma = self.blur_sigma * (max(width, height) / 1920.0)

        # OpenCV GaussianBlur takes kernel size (0,0 means auto from sigma) and sigma
        # Sigma in shader is standard deviation.
        gaussian = cv2.GaussianBlur(img_float, (0, 0), sigmaX=b_sigma, sigmaY=b_sigma)

        # Logic:
        # float3 edges = max( saturate( orig.xyz - gaussian.xyz ) - Threshold, 0.0f );
        edges = np.maximum(saturate(img_float - gaussian) - self.threshold, 0.0)

        # float3 invGauss = saturate( 1.0f - gaussian.xyz );
        inv_gauss = saturate(1.0 - gaussian)

        # float3 oInvGauss = saturate( orig.xyz + invGauss.xyz );
        o_inv_gauss = saturate(img_float + inv_gauss)

        # float3 invOGauss = max( saturate( 1.0f - oInvGauss.xyz ) - Threshold, 0.0f );
        inv_o_gauss = np.maximum(saturate(1.0 - o_inv_gauss) - self.threshold, 0.0)

        # edges = max(( saturate( Sharpening * edges.xyz )) - ( saturate( Sharpening * invOGauss.xyz )), 0.0f );
        edges = np.maximum(
            saturate(self.sharpening * edges) - saturate(self.sharpening * inv_o_gauss),
            0.0,
        )

        # float3 blend = screen( orig.xyz, lerp( min( edges.xyz, limiter ), 0.0f, enable_depth * depth ));
        # Assuming depth disabled (0.0) -> lerp returns min(edges, limiter)

        to_blend = np.minimum(edges, self.limiter)
        blend = self._screen(img_float, to_blend)

        # float3 color = blendLuma( orig.xyz, blend.xyz );
        color = self._blend_luma(img_float, blend)

        if self.show_edges:
            # color.xyz = min( edges.xyz, limiter )
            color = to_blend

        return (np.clip(color, 0.0, 1.0) * 255).astype(np.uint8)
