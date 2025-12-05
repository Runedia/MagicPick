import numpy as np
from numba import njit, prange

from filters.base_filter import BaseFilter


@njit(parallel=True, fastmath=True, cache=True)
def _pattern_shading_kernel(img, h, w, threshold, steps, test):
    out = np.empty((h, w, 3), dtype=np.uint8)

    for y in prange(h):
        for x in range(w):
            # Luma
            r = img[y, x, 0] / 255.0
            g = img[y, x, 1] / 255.0
            b = img[y, x, 2] / 255.0

            luma = (r + g + b) / 3.0  # Shader uses dot(rgb, 1.0/3.0)

            if test:
                luma = x / w

            # Pixel position logic
            # Shader: pos.x % 2 <= 1.
            # In Python, x % 2 is 0 or 1.
            # 0 <= 1 True. 1 <= 1 True.
            # But we determined this means "Even" vs "Odd" based on SV_Position (x.5).
            # Even pixels (0, 2, 4) -> 0.5 % 2 = 0.5 <= 1 (True).
            # Odd pixels (1, 3, 5) -> 1.5 % 2 = 1.5 > 1 (False).

            is_even_x = (x % 2) == 0
            is_even_y = (y % 2) == 0

            # pattern (Default)
            # if(pos.x % 2 <= 1 && pos.y % 2 <= 1) -> Even X, Even Y
            if is_even_x and is_even_y:
                pattern = 0
            else:
                pattern = 1

            # pattern1
            # if((pos.x + 1) % 2 <= 1 && (pos.y - 1) % 2 <= 1)
            # (x+1) even -> x odd. (y-1) even -> y odd.
            # So Odd X, Odd Y.
            if (not is_even_x) and (not is_even_y):
                pattern1 = 1
            else:
                pattern1 = 0

            # pattern2
            # if(pos.x % 2 <= 1 && (pos.y - 1) % 2 <= 1)
            # Even X, Odd Y.
            if is_even_x and (not is_even_y):
                pattern2 = 1
            else:
                pattern2 = 0

            # pattern3
            # if(pos.x % 2 <= 1 && pos.y % 2 <= 1 || (pos.x + 1) % 2 <= 1 && (pos.y + 1) % 2 <= 1)
            # (Even X, Even Y) OR (Odd X, Odd Y)
            if (is_even_x and is_even_y) or ((not is_even_x) and (not is_even_y)):
                pattern3 = 0
            else:
                pattern3 = 1

            final_val = 0.0

            if steps == 0:  # 2 shades
                # ceil(1.0 - step(luma, threshold))
                # step(edge, x) -> 1 if x >= edge else 0.
                # step(luma, threshold) -> 1 if threshold >= luma? No, step(edge, x) returns 1 if x >= edge.
                # HLSL step(y, x) : (x >= y) ? 1 : 0.
                # step(luma, threshold) -> (threshold >= luma) ? 1 : 0.
                # 1.0 - (1 if thresh >= luma else 0) -> 0 if thresh >= luma else 1.
                # So if luma <= threshold, 0. Else 1.
                if luma <= threshold:
                    final_val = 0.0
                else:
                    final_val = 1.0

            elif steps == 1:  # 3 shades
                if luma <= threshold:
                    final_val = 0.0
                elif luma <= threshold * 2.0:
                    final_val = float(pattern3)
                else:
                    final_val = 1.0

            elif steps == 2:  # 4 shades
                if luma <= threshold:
                    final_val = 0.0
                elif luma <= threshold * 2.0:
                    final_val = float(pattern1)
                elif luma > threshold * 3.0:
                    final_val = 1.0
                else:
                    # Fallback to 'pattern' (dots)
                    final_val = float(pattern)

            else:  # 5 shades (steps == 3 or default)
                if luma <= threshold:
                    final_val = 0.0
                elif luma <= threshold * 2.0:
                    final_val = float(pattern2)
                elif luma <= threshold * 3.0:
                    final_val = float(pattern3)
                elif luma > threshold * 4.0:
                    final_val = 1.0
                else:
                    # Fallback?
                    # 3*thresh < luma <= 4*thresh
                    # Code doesn't specify else.
                    # It falls through to 'pattern'.
                    final_val = float(pattern)

            val_byte = int(final_val * 255)
            out[y, x, 0] = val_byte
            out[y, x, 1] = val_byte
            out[y, x, 2] = val_byte

    return out


class PatternShadingFilter(BaseFilter):
    def __init__(self):
        super().__init__("PatternShading", "패턴 쉐이딩")
        self.threshold = 0.1
        self.steps = 3  # 0:2, 1:3, 2:4, 3:5
        self.test = False

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        self.threshold = float(params.get("threshold", self.threshold))
        self.steps = int(params.get("steps", self.steps))
        self.test = bool(params.get("test", self.test))

        h, w = image.shape[:2]

        result = _pattern_shading_kernel(
            image, h, w, self.threshold, self.steps, self.test
        )

        return result
