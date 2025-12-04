import os

import numpy as np
from numba import njit, prange
from PIL import Image

from filters.base_filter import BaseFilter
from filters.reshade.numba_helpers import lerp, saturate


@njit(parallel=True, fastmath=True, cache=True)
def _ui_mask_kernel(img, backup, mask, h, w, intensity, display_mask, 
                    multichannel, toggle_r, toggle_g, toggle_b):
    out = np.empty((h, w, 3), dtype=np.uint8)
    
    # Mask dimensions
    mh, mw = mask.shape[:2]
    
    for y in prange(h):
        for x in range(w):
            # Sample mask (Nearest neighbor or resize? Assuming mask matches screen or we sample UV)
            # If mask is different size, we sample UV.
            u = x / w
            v = y / h
            
            mx = int(u * mw)
            my = int(v * mh)
            mx = min(mx, mw - 1)
            my = min(my, mh - 1)
            
            # Mask value
            m_val = 0.0
            
            if not multichannel:
                # Single channel (Red)
                m_val = mask[my, mx, 0] / 255.0
            else:
                # Multichannel
                mr = mask[my, mx, 0] / 255.0
                mg = mask[my, mx, 1] / 255.0
                mb = mask[my, mx, 2] / 255.0
                
                # float mask = saturate(1.0 - dot(1.0 - mask_rgb, float3(bToggleRed, bToggleGreen, bToggleBlue)));
                # dot(1-m, toggle) = (1-mr)*tr + (1-mg)*tg + (1-mb)*tb
                
                tr = 1.0 if toggle_r else 0.0
                tg = 1.0 if toggle_g else 0.0
                tb = 1.0 if toggle_b else 0.0
                
                dot_val = (1.0 - mr) * tr + (1.0 - mg) * tg + (1.0 - mb) * tb
                m_val = saturate(1.0 - dot_val)
                
            # Blend
            # color = lerp(color, backup, mask * fMask_Intensity);
            factor = m_val * intensity
            
            if display_mask:
                out[y, x, 0] = int(m_val * 255)
                out[y, x, 1] = int(m_val * 255)
                out[y, x, 2] = int(m_val * 255)
            else:
                r = img[y, x, 0]
                g = img[y, x, 1]
                b = img[y, x, 2]
                
                br = backup[y, x, 0]
                bg = backup[y, x, 1]
                bb = backup[y, x, 2]
                
                out[y, x, 0] = int(lerp(float(r), float(br), factor))
                out[y, x, 1] = int(lerp(float(g), float(bg), factor))
                out[y, x, 2] = int(lerp(float(b), float(bb), factor))
                
    return out

class UIMaskFilter(BaseFilter):
    def __init__(self):
        super().__init__("UIMask", "UI 마스크")
        self.mask_intensity = 1.0
        self.display_mask = False
        self.multichannel = False
        self.toggle_red = True
        self.toggle_green = True
        self.toggle_blue = True
        self.mask_path = "UIMask.png" # Default
        self.cached_mask = None
        self.cached_mask_path = None
        
    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        self.mask_intensity = float(params.get("Mask_Intensity", self.mask_intensity))
        self.display_mask = bool(params.get("DisplayMask", self.display_mask))
        self.multichannel = bool(params.get("Multichannel", self.multichannel))
        self.toggle_red = bool(params.get("ToggleRed", self.toggle_red))
        self.toggle_green = bool(params.get("ToggleGreen", self.toggle_green))
        self.toggle_blue = bool(params.get("ToggleBlue", self.toggle_blue))
        self.mask_path = str(params.get("MaskPath", self.mask_path))
        
        original_image = params.get("original_image")
        if original_image is None:
            original_image = image # Fallback to current image (no effect)
            
        # Load Mask
        if self.cached_mask is None or self.cached_mask_path != self.mask_path:
            if os.path.exists(self.mask_path):
                try:
                    pil_mask = Image.open(self.mask_path).convert("RGB")
                    self.cached_mask = np.array(pil_mask)
                    self.cached_mask_path = self.mask_path
                except:
                    # Failed to load, use white mask (full revert) or black (no revert)?
                    # If mask fails, probably better to do nothing (Black mask).
                    self.cached_mask = np.zeros_like(image)
            else:
                # Try looking in resources?
                # For now, just black mask.
                self.cached_mask = np.zeros_like(image)
                
        h, w = image.shape[:2]
        
        # Ensure mask is loaded
        if self.cached_mask is None:
             self.cached_mask = np.zeros((h, w, 3), dtype=np.uint8)
             
        result = _ui_mask_kernel(
            image, original_image, self.cached_mask, h, w,
            self.mask_intensity, self.display_mask,
            self.multichannel, self.toggle_red, self.toggle_green, self.toggle_blue
        )
        
        return result
