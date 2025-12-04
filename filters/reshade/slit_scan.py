import numpy as np
from numba import njit, prange

from filters.base_filter import BaseFilter
from filters.reshade.numba_helpers import clamp


@njit(parallel=True, fastmath=True, cache=True)
def _slit_scan_kernel(img, h, w, position, direction):
    out = np.empty((h, w, 3), dtype=np.uint8)
    
    # direction: 0=Left, 1=Right, 2=Up, 3=Down
    # "Left": Scanned area is on the left (x < position)
    # "Right": Scanned area is on the right (x > position)
    # "Up": Scanned area is on the top (y < position)
    # "Down": Scanned area is on the bottom (y > position)
    
    limit_x = int(position * w)
    limit_y = int(position * h)
    
    limit_x = clamp(limit_x, 0, w - 1)
    limit_y = clamp(limit_y, 0, h - 1)
    
    for y in prange(h):
        for x in range(w):
            sample_x = x
            sample_y = y
            
            is_scanned = False
            
            if direction == 0: # Left
                if x < limit_x:
                    sample_x = limit_x
                    is_scanned = True
            elif direction == 1: # Right
                if x > limit_x:
                    sample_x = limit_x
                    is_scanned = True
            elif direction == 2: # Up
                if y < limit_y:
                    sample_y = limit_y
                    is_scanned = True
            elif direction == 3: # Down
                if y > limit_y:
                    sample_y = limit_y
                    is_scanned = True
            
            if is_scanned:
                out[y, x, 0] = img[sample_y, sample_x, 0]
                out[y, x, 1] = img[sample_y, sample_x, 1]
                out[y, x, 2] = img[sample_y, sample_x, 2]
            else:
                out[y, x, 0] = img[y, x, 0]
                out[y, x, 1] = img[y, x, 1]
                out[y, x, 2] = img[y, x, 2]
                
    return out

class SlitScanFilter(BaseFilter):
    def __init__(self):
        super().__init__("SlitScan", "슬릿 스캔")
        self.position = 0.5
        self.direction = 0 # 0:Left, 1:Right, 2:Up, 3:Down
        self.animate = False # Not used in static
        
    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        self.position = params.get("position", self.position)
        self.direction = int(params.get("direction", self.direction))
        
        h, w = image.shape[:2]
        
        result = _slit_scan_kernel(image, h, w, self.position, self.direction)
        
        return result
