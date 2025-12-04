"""
NeoBloom.fx 정확한 구현

Multi-layer bloom with atlas-based rendering
Original: FXShaders by luluco250
"""

import numpy as np
from numba import njit, prange

from filters.base_filter import BaseFilter

# Helper functions from FXShaders library

def scale_coord(uv, scale, center=0.5):
    """Scale UV coordinates around a center point"""
    if isinstance(center, (int, float)):
        center = np.array([center, center])
    return (uv - center) * scale + center


@njit(fastmath=True, cache=True)
def normal_distribution(x, mean, variance):
    """Gaussian/Normal distribution function"""
    return np.exp(-((x - mean) ** 2) / (2 * variance * variance))


# Tonemap functions (Reinhard)

@njit(fastmath=True, cache=True)
def reinhard_tonemap(color):
    """Reinhard tonemap"""
    return color / (1.0 + color)


@njit(fastmath=True, cache=True)
def reinhard_inverse(color, white_point=1.0):
    """Reinhard inverse tonemap (for HDR extraction)"""
    return white_point * color / (white_point - color + 1e-6)


@njit(fastmath=True, cache=True)
def reinhard_inverse_lum(color, white_point=1.0):
    """Reinhard inverse tonemap preserving luminance"""
    luma = np.dot(color, np.array([0.2126, 0.7152, 0.0722]))
    if luma < 1e-6:
        return color
    inv_luma = white_point * luma / (white_point - luma + 1e-6)
    return color * (inv_luma / (luma + 1e-6))


def apply_saturation(color, saturation):
    """Apply saturation to RGB color"""
    luma = np.dot(color, [0.2126, 0.7152, 0.0722])
    return np.clip(luma + (color - luma) * saturation, 0, None)


def blend_screen(a, b, weight):
    """Screen blend mode"""
    return 1.0 - (1.0 - a) * (1.0 - b * weight)


@njit(parallel=True, fastmath=True, cache=True)
def _gaussian_blur_1d(img, h, w, direction, sigma, samples):
    """
    1D Gaussian blur
    
    Args:
        img: Input image
        direction: (dx, dy) blur direction
        sigma: Gaussian sigma
        samples: Number of samples
    """
    out = np.zeros((h, w, 4), dtype=np.float32)
    dx, dy = direction
    
    # Calculate Gaussian weights
    weights = np.zeros(samples, dtype=np.float32)
    total_weight = 0.0
    half_samples = samples // 2
    
    for i in range(samples):
        offset = i - half_samples
        weight = np.exp(-(offset * offset) / (2.0 * sigma * sigma))
        weights[i] = weight
        total_weight += weight
    
    # Normalize weights
    weights /= total_weight
    
    for y in prange(h):
        for x in range(w):
            r_sum = 0.0
            g_sum = 0.0
            b_sum = 0.0
            a_sum = 0.0
            
            for i in range(samples):
                offset = i - half_samples
                ny = int(np.clip(y + dy * offset, 0, h - 1))
                nx = int(np.clip(x + dx * offset, 0, w - 1))
                
                weight = weights[i]
                r_sum += img[ny, nx, 0] * weight
                g_sum += img[ny, nx, 1] * weight
                b_sum += img[ny, nx, 2] * weight
                a_sum += img[ny, nx, 3] * weight
            
            out[y, x, 0] = r_sum
            out[y, x, 1] = g_sum
            out[y, x, 2] = b_sum
            out[y, x, 3] = a_sum
    
    return out


class NeoBloomFilter(BaseFilter):
    """
    NeoBloom - Advanced multi-layer bloom
    
    5-layer bloom atlas with Gaussian blur and normal distribution blending
    """
    
    def __init__(self):
        super().__init__("NeoBloom", "네오 블룸")
        
        # Bloom levels: (x, y, scale, miplevel)
        # These define the position and size of each bloom layer in the atlas
        self.bloom_levels = [
            (0.0, 0.5, 0.5, 1),      # Layer 0: top-left
            (0.5, 0.0, 0.25, 2),     # Layer 1: top-right
            (0.75, 0.875, 0.125, 3), # Layer 2: bottom-right small
            (0.875, 0.0, 0.0625, 5), # Layer 3: top-right tiny
            (0.0, 0.0, 0.03, 7)      # Layer 4: top-left micro
        ]
        self.bloom_count = 5
        
        # Basic parameters
        self.intensity = 1.0
        self.saturation = 1.0
        self.color_filter = np.array([1.0, 1.0, 1.0])
        
        # Blend mode: 0=Mix, 1=Addition, 2=Screen
        self.blend_mode = 1
        
        # Blending parameters
        self.mean = 0.0  # Bias between bloom layers
        self.variance = 5.0  # Contrast in bloom sizes
        
        # HDR parameters
        self.max_brightness = 100.0
        self.normalize_brightness = True
        self.magic_mode = False
        
        # Blur parameters
        self.sigma = 4.0
        self.padding = 0.1
        self.blur_samples = 27
        
        # Internal
        self.down_scale = 2
    
    def _inverse_tonemap_bloom(self, color):
        """Extract HDR information for bloom"""
        if self.magic_mode:
            return np.power(np.abs(color), self.max_brightness * 0.01)
        return reinhard_inverse_lum(color, 1.0 / self.max_brightness)
    
    def _inverse_tonemap(self, color):
        """Extract HDR information"""
        if self.magic_mode:
            return color
        return reinhard_inverse(color, 1.0 / self.max_brightness)
    
    def _tonemap(self, color):
        """Apply tonemap"""
        if self.magic_mode:
            return color
        return reinhard_tonemap(color)
    
    def _blend_bloom(self, color, bloom):
        """Blend bloom with scene color"""
        if self.normalize_brightness:
            w = self.intensity / self.max_brightness
        else:
            w = self.intensity
        
        if self.blend_mode == 0:  # Mix
            return np.clip(
                color * (1.0 - np.log2(w + 1.0)) + bloom * np.log2(w + 1.0),
                0, None
            )
        elif self.blend_mode == 1:  # Addition
            return color + bloom * w * 3.0
        elif self.blend_mode == 2:  # Screen
            return blend_screen(color, bloom, w)
        return color
    
    def _downsample_preprocess(self, img):
        """Downsample and preprocess image for bloom extraction"""
        h, w = img.shape[:2]
        new_h = max(h // self.down_scale, 1)
        new_w = max(w // self.down_scale, 1)
        
        # Simple box downsample
        result = np.zeros((new_h, new_w, 3), dtype=np.float32)
        
        for y in range(new_h):
            for x in range(new_w):
                sy = y * self.down_scale
                sx = x * self.down_scale
                
                # Average pixels in block
                count = 0
                r_sum, g_sum, b_sum = 0.0, 0.0, 0.0
                
                for dy in range(self.down_scale):
                    for dx in range(self.down_scale):
                        ny = min(sy + dy, h - 1)
                        nx = min(sx + dx, w - 1)
                        r_sum += img[ny, nx, 0]
                        g_sum += img[ny, nx, 1]
                        b_sum += img[ny, nx, 2]
                        count += 1
                
                result[y, x, 0] = r_sum / count
                result[y, x, 1] = g_sum / count
                result[y, x, 2] = b_sum / count
        
        # Apply saturation
        for y in range(new_h):
            for x in range(new_w):
                result[y, x] = apply_saturation(
                    result[y, x], self.saturation
                )
        
        # Apply color filter
        result *= self.color_filter
        
        # Inverse tonemap for HDR
        for y in range(new_h):
            for x in range(new_w):
                result[y, x] = self._inverse_tonemap_bloom(result[y, x])
        
        return result
    
    def _create_bloom_atlas(self, downsampled):
        """Create bloom atlas with all layers"""
        ds_h, ds_w = downsampled.shape[:2]
        
        # Atlas size (same as downsampled)
        atlas = np.zeros((ds_h, ds_w, 4), dtype=np.float32)
        
        # Split into bloom layers
        for i, (rect_x, rect_y, rect_scale, mip_level) in enumerate(
            self.bloom_levels
        ):
            # Calculate padding for this layer
            layer_padding = self.padding * (i + 1)
            
            # Size of this bloom rect in atlas
            rect_w = int(ds_w * rect_scale)
            rect_h = int(ds_h * rect_scale)
            
            if rect_w < 1 or rect_h < 1:
                continue
            
            # Position in atlas
            atlas_x = int(rect_x * ds_w)
            atlas_y = int(rect_y * ds_h)
            
            # Downsample further for higher mip levels
            mip_source = downsampled.copy()
            for _ in range(mip_level):
                sh, sw = mip_source.shape[:2]
                if sh <= 1 or sw <= 1:
                    break
                mip_source = mip_source[::2, ::2]
            
            # Resize to rect size with padding
            padded_scale = 1.0 + layer_padding
            target_h = max(int(rect_h / padded_scale), 1)
            target_w = max(int(rect_w / padded_scale), 1)
            
            # Simple resize (nearest neighbor for speed)
            mh, mw = mip_source.shape[:2]
            resized = np.zeros((target_h, target_w, 3), dtype=np.float32)
            
            for y in range(target_h):
                for x in range(target_w):
                    sy = int(y * mh / target_h)
                    sx = int(x * mw / target_w)
                    sy = min(sy, mh - 1)
                    sx = min(sx, mw - 1)
                    resized[y, x] = mip_source[sy, sx]
            
            # Place in atlas with padding offset
            pad_offset_y = int((rect_h - target_h) / 2)
            pad_offset_x = int((rect_w - target_w) / 2)
            
            for y in range(target_h):
                for x in range(target_w):
                    ay = atlas_y + pad_offset_y + y
                    ax = atlas_x + pad_offset_x + x
                    
                    if 0 <= ay < ds_h and 0 <= ax < ds_w:
                        atlas[ay, ax, :3] = resized[y, x]
                        atlas[ay, ax, 3] = 1.0  # Alpha
        
        return atlas
    
    def _blur_atlas(self, atlas):
        """Apply separable Gaussian blur to atlas"""
        h, w = atlas.shape[:2]
        
        # Horizontal blur
        blurred_h = _gaussian_blur_1d(
            atlas, h, w, (1.0, 0.0), self.sigma, self.blur_samples
        )
        
        # Vertical blur
        blurred_v = _gaussian_blur_1d(
            blurred_h, h, w, (0.0, 1.0), self.sigma, self.blur_samples
        )
        
        return blurred_v
    
    def _join_blooms(self, atlas, atlas_h, atlas_w):
        """Join bloom layers with normal distribution weighting"""
        result = np.zeros((atlas_h, atlas_w, 3), dtype=np.float32)
        
        for y in range(atlas_h):
            for x in range(atlas_w):
                uv_y = y / atlas_h
                uv_x = x / atlas_w
                
                bloom_sum = np.array([0.0, 0.0, 0.0])
                weight_sum = 0.0
                
                # Sample each bloom layer with normal distribution weight
                for i, (rect_x, rect_y, rect_scale, _) in enumerate(
                    self.bloom_levels
                ):
                    # Calculate weight for this layer
                    weight = normal_distribution(i, self.mean, self.variance)
                    
                    # Calculate sampling position with padding
                    layer_padding = self.padding * (i + 1)
                    padded_scale = 1.0 + layer_padding
                    
                    # Transform UV to rect space
                    rect_uv_x = (uv_x - 0.5) / padded_scale + 0.5
                    rect_uv_y = (uv_y - 0.5) / padded_scale + 0.5
                    
                    # Transform to atlas position
                    sample_y = (rect_uv_y + rect_y / rect_scale) * rect_scale
                    sample_x = (rect_uv_x + rect_x / rect_scale) * rect_scale
                    
                    # Sample atlas
                    sy = int(sample_y * atlas_h)
                    sx = int(sample_x * atlas_w)
                    
                    if 0 <= sy < atlas_h and 0 <= sx < atlas_w:
                        bloom_sum += atlas[sy, sx, :3] * weight
                        weight_sum += weight
                
                if weight_sum > 0:
                    result[y, x] = bloom_sum / weight_sum
        
        return result
    
    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Apply NeoBloom filter
        
        Args:
            image: RGB image (uint8, 0-255)
            **params: Filter parameters
        
        Returns:
            Bloomed image (uint8, 0-255)
        """
        # Update parameters
        self.intensity = params.get("intensity", self.intensity)
        self.saturation = params.get("saturation", self.saturation)
        self.mean = params.get("mean", self.mean)
        self.variance = params.get("variance", self.variance)
        self.max_brightness = params.get("max_brightness", self.max_brightness)
        self.sigma = params.get("sigma", self.sigma)
        self.blend_mode = params.get("blend_mode", self.blend_mode)
        
        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        orig_h, orig_w = img_float.shape[:2]
        
        # 1. Downsample and preprocess
        downsampled = self._downsample_preprocess(img_float)
        
        # 2. Create bloom atlas
        atlas = self._create_bloom_atlas(downsampled)
        
        # 3. Blur atlas
        blurred_atlas = self._blur_atlas(atlas)
        
        # 4. Join blooms with weights
        atlas_h, atlas_w = blurred_atlas.shape[:2]
        bloom = self._join_blooms(blurred_atlas, atlas_h, atlas_w)
        
        # 5. Upsample bloom to original size
        bloom_full = np.zeros((orig_h, orig_w, 3), dtype=np.float32)
        for y in range(orig_h):
            for x in range(orig_w):
                sy = int(y * atlas_h / orig_h)
                sx = int(x * atlas_w / orig_w)
                sy = min(sy, atlas_h - 1)
                sx = min(sx, atlas_w - 1)
                bloom_full[y, x] = bloom[sy, sx]
        
        # 6. Inverse tonemap original image
        color_hdr = np.zeros_like(img_float)
        for y in range(orig_h):
            for x in range(orig_w):
                color_hdr[y, x] = self._inverse_tonemap(img_float[y, x])
        
        # 7. Blend bloom with scene
        result = np.zeros_like(img_float)
        for y in range(orig_h):
            for x in range(orig_w):
                result[y, x] = self._blend_bloom(
                    color_hdr[y, x], bloom_full[y, x]
                )
        
        # 8. Tonemap final result
        for y in range(orig_h):
            for x in range(orig_w):
                result[y, x] = self._tonemap(result[y, x])
        
        # Clamp and convert
        result = np.clip(result, 0, 1)
        return (result * 255).astype(np.uint8)
