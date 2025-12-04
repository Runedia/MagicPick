import numpy as np
from numba import njit, prange

from filters.base_filter import BaseFilter
from filters.reshade.numba_helpers import saturate


@njit(parallel=True, fastmath=True, cache=True)
def _splitscreen_kernel(img, orig, h, w, mode):
    out = np.empty((h, w, 3), dtype=np.uint8)
    
    for y in prange(h):
        for x in range(w):
            u = x / w
            v = y / h
            
            use_before = False
            
            if mode == 0: # Vertical 50/50
                if u < 0.5:
                    use_before = True
            elif mode == 1: # Vertical 25/50/25
                dist = abs(u - 0.5)
                if saturate(dist - 0.25) > 0.0:
                    use_before = True
            elif mode == 2: # Angled 50/50
                dist = (u - 3.0/8.0) + (v * 0.25)
                if saturate(dist - 0.25) > 0.0:
                    # Shader: color = dist ? BackBuffer : Before
                    # Wait, shader says: color = dist ? BackBuffer : Before_sampler
                    # If dist > 0 (True), use BackBuffer (After).
                    # So if saturate(...) > 0, use After.
                    # My logic: if condition, use_before = True.
                    # So if saturate(...) > 0, use_before = False.
                    use_before = False
                else:
                    use_before = True
            elif mode == 3: # Angled 25/50/25
                dist = (u - 3.0/8.0) + (v * 0.25)
                dist = abs(dist - 0.25)
                if saturate(dist - 0.25) > 0.0:
                    use_before = True
            elif mode == 4: # Horizontal 50/50
                if v < 0.5:
                    use_before = True
            elif mode == 5: # Horizontal 25/50/25
                dist = abs(v - 0.5)
                if saturate(dist - 0.25) > 0.0:
                    use_before = True
            elif mode == 6: # Diagonal
                dist = u + v
                if dist < 1.0:
                    use_before = True
            
            if use_before:
                out[y, x, 0] = orig[y, x, 0]
                out[y, x, 1] = orig[y, x, 1]
                out[y, x, 2] = orig[y, x, 2]
            else:
                out[y, x, 0] = img[y, x, 0]
                out[y, x, 1] = img[y, x, 1]
                out[y, x, 2] = img[y, x, 2]
                
    return out

class SplitscreenFilter(BaseFilter):
    def __init__(self):
        super().__init__("Splitscreen", "스플릿 스크린")
        self.mode = 0 # 0-6
        
    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        self.mode = int(params.get("mode", self.mode))
        original_image = params.get("original_image")
        
        if original_image is None:
            # If no original image provided, just return current image (no split possible)
            return image
            
        # Ensure original image matches size
        if original_image.shape != image.shape:
            # Resize original to match current? Or just return image.
            # For robustness, let's return image if shapes mismatch significantly.
            if original_image.shape[:2] != image.shape[:2]:
                 return image
        
        h, w = image.shape[:2]
        
        result = _splitscreen_kernel(image, original_image, h, w, self.mode)
        
        return result
