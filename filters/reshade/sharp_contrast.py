import numpy as np
from numba import njit, prange

from filters.base_filter import BaseFilter
from filters.reshade.numba_helpers import clamp, saturate


@njit(parallel=True, fastmath=True, cache=True)
def _sharp_contrast_kernel(img, h, w, strength, edge_bias, edge_floor, gamma_correct, size):
    out = np.empty((h, w, 3), dtype=np.float32)
    
    # Constants
    PI = 3.1415926536
    
    # Determine SECTOR_COUNT based on size (matching HLSL logic)
    sector_count = 4 # Default fallback
    if size <= 2: sector_count = 2
    elif size == 3: sector_count = 3
    elif size == 4: sector_count = 3
    elif size == 5: sector_count = 4
    elif size == 6: sector_count = 6
    elif size == 7: sector_count = 7
    elif size == 8: sector_count = 8
    elif size == 9: sector_count = 9
    elif size >= 10: sector_count = 13 # HLSL says 13 for 10, 15 for >= 11
    if size >= 11: sector_count = 15

    radius = size // 2
    
    sharpness_multiplier = max(1023 * ((2 * edge_bias / 3) + 0.333333)**4, 1e-10)
    
    # Pre-calculate luma for the whole image to avoid re-calculating in the loop
    # Using BT.601 coefficients as in the shader: 0.299, 0.587, 0.114
    luma_map = np.empty((h, w), dtype=np.float32)
    for y in prange(h):
        for x in range(w):
            l = 0.299 * img[y, x, 0] + 0.587 * img[y, x, 1] + 0.114 * img[y, x, 2]
            if gamma_correct:
                l = l * l
            luma_map[y, x] = l

    for y in prange(h):
        for x in range(w):
            # Arrays for sectors (using max possible size to be safe or dynamic list if numba supports, 
            # but fixed size array is better for numba. Max sectors is 15)
            sum_val = np.zeros(15, dtype=np.float32)
            squared_sum = np.zeros(15, dtype=np.float32)
            sample_count = np.zeros(15, dtype=np.float32)
            
            maximum = 0.0
            minimum = sharpness_multiplier
            center_luma = luma_map[y, x]
            
            # Center pixel processing
            center_val = center_luma
            scaled_center = center_luma * sharpness_multiplier
            maximum = max(maximum, scaled_center)
            minimum = min(minimum, scaled_center)
            
            # Add center to all sectors (logic from HLSL: if all(int2(i, j) == 0))
            # Actually HLSL loop includes center (0,0).
            # If 0,0: adds to ALL sectors.
            
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny = clamp(y + dy, 0, h - 1)
                    nx = clamp(x + dx, 0, w - 1)
                    
                    l = luma_map[ny, nx]
                    scaled_l = l * sharpness_multiplier
                    
                    if dy == 0 and dx == 0:
                        # Center pixel: add to all sectors
                        for k in range(sector_count):
                            sum_val[k] += scaled_l
                            squared_sum[k] += scaled_l * scaled_l
                            sample_count[k] += 1.0
                    else:
                        # Calculate sector
                        angle = np.arctan2(float(dy), float(dx)) + PI
                        sector_idx = int((angle * sector_count) / (PI * 2)) % sector_count
                        
                        dist = np.sqrt(float(dx*dx + dy*dy))
                        maximum = max(maximum, scaled_l / dist)
                        minimum = min(minimum, scaled_l * dist)
                        
                        sum_val[sector_idx] += scaled_l
                        squared_sum[sector_idx] += scaled_l * scaled_l
                        sample_count[sector_idx] += 1.0

            # Edge Multiplier
            edge_mult = max((1.0 - (maximum - minimum) / maximum), 1e-5) if maximum > 0 else 1e-5 # rcp(maximum)
            edge_mult = edge_mult * (1.0 - edge_floor) + edge_floor
            
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for k in range(sector_count):
                if sample_count[k] > 0:
                    mean = sum_val[k] / sample_count[k]
                    variance = (squared_sum[k] - (sum_val[k] * sum_val[k] / sample_count[k])) / sample_count[k]
                    
                    # weight = rcp(1 + variance^2)
                    w_val = max(variance, 1e-5)
                    w_val = w_val * w_val
                    w_val = w_val * w_val # power of 4 effectively? HLSL: weight *= weight; weight *= weight;
                    weight = 1.0 / (1.0 + w_val)
                    
                    weighted_sum += mean * weight
                    weight_sum += weight
            
            kuwahara = 0.0
            if weight_sum > 0:
                kuwahara = (weighted_sum / weight_sum) / sharpness_multiplier
            
            # Final mix
            # kuwahara = center + (center - kuwahara) * SharpenStrength * edgeMultiplier * 1.5;
            final_val = center_val + (center_val - kuwahara) * strength * edge_mult * 1.5
            
            if gamma_correct:
                final_val = np.sqrt(max(final_val, 0.0))
            
            # Reconstruct color
            # Original color
            r = img[y, x, 0]
            g = img[y, x, 1]
            b = img[y, x, 2]
            
            # YCbCr conversion (using shader coefficients)
            # float cb = dot(color, float3(-0.168736, -0.331264, 0.5));
            # float cr = dot(color, float3(0.5, -0.418688, -0.081312));
            cb = -0.168736 * r - 0.331264 * g + 0.5 * b
            cr = 0.5 * r - 0.418688 * g - 0.081312 * b
            
            # New RGB from (final_val, cb, cr)
            # output.r = dot(float2(y, cr), float2(1, 1.402));
            # output.g = dot(float3(y, cb, cr), float3(1, -0.344135, -0.714136));
            # output.b = dot(float2(y, cb), float2(1, 1.772));
            
            new_r = final_val + 1.402 * cr
            new_g = final_val - 0.344135 * cb - 0.714136 * cr
            new_b = final_val + 1.772 * cb
            
            out[y, x, 0] = saturate(new_r)
            out[y, x, 1] = saturate(new_g)
            out[y, x, 2] = saturate(new_b)
            
    return out

class SharpContrastFilter(BaseFilter):
    def __init__(self):
        super().__init__("SharpContrast", "샤프 대비")
        self.strength = 0.667
        self.edge_bias = 1.0
        self.edge_floor = 0.0
        self.gamma_correct = True
        self.size = 7
        
    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        self.strength = params.get("Strength", self.strength)
        self.edge_bias = params.get("EdgeBias", self.edge_bias)
        self.edge_floor = params.get("EdgeFloor", self.edge_floor)
        self.gamma_correct = params.get("GammaCorrect", self.gamma_correct)
        self.size = int(params.get("Size", self.size))
        
        img_float = image.astype(np.float32) / 255.0
        h, w = img_float.shape[:2]
        
        result = _sharp_contrast_kernel(
            img_float, h, w, 
            self.strength, self.edge_bias, self.edge_floor, 
            self.gamma_correct, self.size
        )
        
        return (result * 255).astype(np.uint8)
