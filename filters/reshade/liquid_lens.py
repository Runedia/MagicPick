import numpy as np
from numba import njit, prange
from PIL import Image
from scipy.ndimage import gaussian_filter

from filters.base_filter import BaseFilter
from filters.reshade.numba_helpers import lerp, saturate


@njit(fastmath=True, inline="always", cache=True)
def _scale_coord(u, v, scale, pivot_x=0.5, pivot_y=0.5):
    su = (u - pivot_x) * scale + pivot_x
    sv = (v - pivot_y) * scale + pivot_y
    return su, sv

@njit(fastmath=True, inline="always", cache=True)
def _fisheye_lens(u, v, amount, zoom):
    # Normalize to -1..1
    u = u * 2.0 - 1.0
    v = v * 2.0 - 1.0
    
    # Aspect ratio correction omitted for simplicity (assuming square or handled by caller)
    # If needed, we can pass aspect ratio.
    
    # fishUv = uv (simplified)
    dist_sq = u*u + v*v
    distort = np.sqrt(max(1.0 - dist_sq, 0.0))
    
    factor = lerp(1.0, distort * zoom, amount)
    
    u *= factor
    v *= factor
    
    # Back to 0..1
    u = (u + 1.0) * 0.5
    v = (v + 1.0) * 0.5
    
    return u, v

@njit(parallel=True, fastmath=True, cache=True)
def _liquid_lens_composite(img, flare_tex, h, w, fh, fw, fisheye_amount, fisheye_zoom, tint_amount, 
                           scales, tints_r, tints_g, tints_b, tints_a):
    out = np.empty((h, w, 3), dtype=np.uint8)
    
    # Scales and Tints are arrays of 7 elements
    
    for y in prange(h):
        for x in range(w):
            # Normalized UV
            u = x / w
            v = y / h
            
            # Apply Fisheye to UV
            fu, fv = _fisheye_lens(u, v, fisheye_amount * 10.0, fisheye_zoom * 3.0)
            
            # Accumulate flares
            flare_r = 0.0
            flare_g = 0.0
            flare_b = 0.0
            
            for i in range(7):
                # Scale UV for this flare layer
                # Shader: ScaleCoord(uv, Scale##id * BaseFlareDownscale)
                # BaseFlareDownscale is 4.0.
                scale_val = scales[i] * 4.0
                su, sv = _scale_coord(fu, fv, scale_val)
                
                # Sample flare texture (bilinear)
                sx = su * fw
                sy = sv * fh
                
                # Clamp/Wrap? Shader uses BORDER.
                if sx < 0 or sx >= fw - 1 or sy < 0 or sy >= fh - 1:
                    # Border (black?)
                    samp_r, samp_g, samp_b = 0.0, 0.0, 0.0
                else:
                    # Bilinear interpolation
                    x0 = int(sx)
                    y0 = int(sy)
                    x1 = min(x0 + 1, fw - 1)
                    y1 = min(y0 + 1, fh - 1)
                    
                    dx = sx - x0
                    dy = sy - y0
                    
                    # Sample 4 neighbors
                    c00 = flare_tex[y0, x0]
                    c10 = flare_tex[y0, x1]
                    c01 = flare_tex[y1, x0]
                    c11 = flare_tex[y1, x1]
                    
                    # Interpolate
                    c0 = c00 * (1 - dx) + c10 * dx
                    c1 = c01 * (1 - dx) + c11 * dx
                    c = c0 * (1 - dy) + c1 * dy
                    
                    samp_r, samp_g, samp_b = c[0], c[1], c[2]
                
                # Apply Tint
                # _GET_TINT(id) float4(lerp(1.0, Tint##id##.rgb * Tint##id##.a, TintAmount), 1.0)
                tr = lerp(1.0, tints_r[i] * tints_a[i], tint_amount)
                tg = lerp(1.0, tints_g[i] * tints_a[i], tint_amount)
                tb = lerp(1.0, tints_b[i] * tints_a[i], tint_amount)
                
                flare_r += samp_r * tr
                flare_g += samp_g * tg
                flare_b += samp_b * tb
            
            flare_r /= 7.0
            flare_g /= 7.0
            flare_b /= 7.0
            
            # Add to original
            orig_r = img[y, x, 0] / 255.0
            orig_g = img[y, x, 1] / 255.0
            orig_b = img[y, x, 2] / 255.0
            
            # Tone mapping (simplified: just add and saturate)
            # Shader does Inverse Tonemap -> Add -> Apply Tonemap.
            # Assuming standard range, we just add.
            
            final_r = orig_r + flare_r
            final_g = orig_g + flare_g
            final_b = orig_b + flare_b
            
            out[y, x, 0] = saturate(final_r) * 255
            out[y, x, 1] = saturate(final_g) * 255
            out[y, x, 2] = saturate(final_b) * 255
            
    return out

class LiquidLensFilter(BaseFilter):
    def __init__(self):
        super().__init__("LiquidLens", "리퀴드 렌즈")
        self.brightness = 0.1
        self.saturation = 0.7
        self.threshold = 0.95
        self.blur_size = 1.0
        self.blur_sigma = 0.5
        self.fisheye_amount = -0.1
        self.fisheye_zoom = 1.0
        self.tint_amount = 1.0
        
        # Tints (RGBA)
        self.tints = [
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.5, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0]
        ]
        
        # Scales
        self.scales = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
        
    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        self.brightness = params.get("brightness", self.brightness)
        self.saturation = params.get("saturation", self.saturation)
        self.threshold = params.get("threshold", self.threshold)
        self.blur_size = params.get("blur_size", self.blur_size)
        self.blur_sigma = params.get("blur_sigma", self.blur_sigma)
        self.fisheye_amount = params.get("fisheye_amount", self.fisheye_amount)
        self.fisheye_zoom = params.get("fisheye_zoom", self.fisheye_zoom)
        self.tint_amount = params.get("tint_amount", self.tint_amount)
        
        # Prepare Flare Source
        img_float = image.astype(np.float32) / 255.0
        
        # Downscale (1/4)
        h, w = img_float.shape[:2]
        small_h, small_w = h // 4, w // 4
        if small_h < 1 or small_w < 1:
            small_h, small_w = h, w
            
        # Use PIL for resize (fast and good quality)
        pil_img = Image.fromarray(image)
        small_pil = pil_img.resize((small_w, small_h), Image.BILINEAR)
        small_img = np.array(small_pil).astype(np.float32) / 255.0
        
        # Apply Threshold/Saturation/Brightness
        # Saturation
        luma = np.sum(small_img * [0.2126, 0.7152, 0.0722], axis=2, keepdims=True)
        small_img = luma + (small_img - luma) * self.saturation
        
        # Threshold
        mask = (small_img >= self.threshold).astype(np.float32)
        small_img *= mask
        
        # Brightness
        small_img *= self.brightness
        
        # Blur
        # Shader: sigma = sqrt(BlurSamples) * BlurSigma; (Samples=9 -> sqrt=3)
        sigma = 3.0 * self.blur_sigma * self.blur_size # * Downscale? Shader says "dir *= BlurSize * Downscale"
        # GaussianBlur1D uses sigma.
        # Let's approximate sigma.
        
        blurred_flare = np.zeros_like(small_img)
        for c in range(3):
            blurred_flare[:, :, c] = gaussian_filter(small_img[:, :, c], sigma=sigma)
            
        # Composite
        tints_r = np.array([t[0] for t in self.tints], dtype=np.float32)
        tints_g = np.array([t[1] for t in self.tints], dtype=np.float32)
        tints_b = np.array([t[2] for t in self.tints], dtype=np.float32)
        tints_a = np.array([t[3] for t in self.tints], dtype=np.float32)
        scales = np.array(self.scales, dtype=np.float32)
        
        result = _liquid_lens_composite(
            image, blurred_flare, h, w, small_h, small_w,
            self.fisheye_amount, self.fisheye_zoom, self.tint_amount,
            scales, tints_r, tints_g, tints_b, tints_a
        )
        
        return result
