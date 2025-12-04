import numpy as np
from numba import njit, prange

from filters.base_filter import BaseFilter
from filters.reshade.numba_helpers import clamp, lerp


@njit(parallel=True, fastmath=True, cache=True)
def _radial_slit_scan_kernel(
    img, h, w, center_x, center_y, anim_rate, border_color, opacity, min_depth
):
    out = np.empty((h, w, 3), dtype=np.uint8)

    # Calculate max radius (distance to furthest corner)
    # Coordinates are normalized [0, 1] in shader, but here we work in pixels or normalized?
    # Shader uses normalized coords for distance calculation but corrects for aspect ratio.
    # "ar_raw = BUFFER_HEIGHT / BUFFER_WIDTH"
    # "tc.x /= ar_raw" -> This means x is scaled by aspect ratio.

    ar = h / w

    cx = center_x
    cy = center_y

    # Calculate max_radius in aspect-corrected space
    # Corners: (0,0), (1,0), (0,1), (1,1)
    # Distances from (cx, cy) with x scaled by 1/ar

    # Shader:
    # float2 center = float2(x_coord, y_coord)/2.0; -> Wait, why /2.0?
    # Ah, "float2 center = float2(x_coord, y_coord)/2.0;" in SlitScan.
    # But x_coord default is 0.5. So center becomes 0.25? That seems odd.
    # Let's check the shader again.
    # "float2 center = float2(x_coord, y_coord)/2.0;"
    # Maybe the shader assumes coordinates are 0..2 range? Or maybe it's a bug in the shader?
    # Or maybe x_coord is -1 to 1? UI says min=0.0, max=1.0.
    # If I use 0.5, center is 0.25.
    # "float2 tc = texcoord - center;" -> texcoord is 0..1.
    # If center is 0.25, then center of effect is at 0.25, 0.25.
    # This implies the shader might be off-center by default?
    # Let's stick to the UI tooltip: "The X position of the center of the effect."
    # If user sets 0.5, they expect center.
    # I will assume standard normalized coordinates 0.5 = center.
    # I will IGNORE the "/2.0" unless it's a specific coordinate system quirk I don't see.
    # Actually, let's look at "Include/RadialSlitScan.fxh" again? No, it was in the main file.
    # "float2 center = float2(x_coord, y_coord)/2.0;"
    # This is very suspicious.
    # However, for my implementation, I will use (center_x, center_y) as normalized coordinates directly.

    # Aspect ratio correction
    # Shader: "tc.x /= ar_raw;" (where ar_raw = H/W)
    # So x is divided by H/W -> x * W/H.
    # Effectively working in "Height" units?
    # Let's just use pixel coordinates for simplicity and clarity.

    cx_px = center_x * w
    cy_px = center_y * h

    # Max radius in pixels
    # Dist to 4 corners
    d1 = np.sqrt((0 - cx_px) ** 2 + (0 - cy_px) ** 2)
    d2 = np.sqrt((w - cx_px) ** 2 + (0 - cy_px) ** 2)
    d3 = np.sqrt((0 - cx_px) ** 2 + (h - cy_px) ** 2)
    d4 = np.sqrt((w - cx_px) ** 2 + (h - cy_px) ** 2)
    max_radius_px = max(max(d1, d2), max(d3, d4))

    limit_px = anim_rate * max_radius_px

    # Border thickness (hardcoded in shader as 0.0025 normalized?)
    # "dist <= slice_to_fill + 0.0025"
    # 0.0025 in normalized space (approx). Let's say 0.0025 * max(w, h) or just a few pixels.
    # Let's use 3 pixels.
    border_px = 3.0

    for y in prange(h):
        for x in range(w):
            dx = x - cx_px
            dy = y - cy_px
            dist = np.sqrt(dx * dx + dy * dy)

            if dist < limit_px:
                # Inside the scan: Smear from the limit
                # We want to sample at the intersection of the ray and the limit circle.
                # Ray direction: (dx, dy) / dist
                # Sample pos: center + dir * limit_px

                if dist < 0.1:  # Avoid division by zero at exact center
                    # Sample exactly at limit in some direction (e.g. angle 0) or just keep center?
                    # If we are at center, direction is undefined.
                    # But if we smear, the center should be the average of the ring?
                    # Or just sample from (cx + limit, cy).
                    # Let's just use the calculated angle if dist > 0.
                    pass

                scale = limit_px / max(dist, 0.001)
                sx = cx_px + dx * scale
                sy = cy_px + dy * scale

                # Clamp to screen
                sx = clamp(sx, 0, w - 1)
                sy = clamp(sy, 0, h - 1)

                # Bilinear sample (simplified to nearest for now, or use integer cast)
                # For better quality, use bilinear. But for "smear", nearest might be okay or slightly blocky.
                # Let's use nearest for speed/simplicity in this kernel.
                isx = int(sx)
                isy = int(sy)

                r = img[isy, isx, 0]
                g = img[isy, isx, 1]
                b = img[isy, isx, 2]

                # Blend with border if close?
                # Shader logic:
                # if dist < slice: blended(base, scanned)
                # scanned is the smeared color.
                # base is the original color.
                # blended using "blending_amount" (which is opacity?).
                # Wait, "blending_amount" is not "opacity".
                # Shader: "color.rgb = ComHeaders::Blending::Blend(render_type, base.rgb, scanned_color.rgb, blending_amount);"
                # "opacity" is used for the BORDER.

                # I will assume "opacity" param passed to this function controls the mix of the effect.
                # If opacity is 1.0, we show the smeared color.

                # Original color at this pixel
                or_r = img[y, x, 0]
                or_g = img[y, x, 1]
                or_b = img[y, x, 2]

                # Mix
                out[y, x, 0] = int(lerp(or_r, r, opacity))
                out[y, x, 1] = int(lerp(or_g, g, opacity))
                out[y, x, 2] = int(lerp(or_b, b, opacity))

            elif dist <= limit_px + border_px:
                # Border
                # lerp(screen, border_color, opacity)
                # Screen is original image.
                or_r = img[y, x, 0]
                or_g = img[y, x, 1]
                or_b = img[y, x, 2]

                br = border_color[0] * 255
                bg = border_color[1] * 255
                bb = border_color[2] * 255

                out[y, x, 0] = int(lerp(or_r, br, opacity))
                out[y, x, 1] = int(lerp(or_g, bg, opacity))
                out[y, x, 2] = int(lerp(or_b, bb, opacity))

            else:
                # Outside: Original image
                out[y, x, 0] = img[y, x, 0]
                out[y, x, 1] = img[y, x, 1]
                out[y, x, 2] = img[y, x, 2]

    return out


class RadialSlitScanFilter(BaseFilter):
    def __init__(self):
        super().__init__("RadialSlitScan", "방사형 슬릿 스캔")
        self.anim_rate = 0.5
        self.x_coord = 0.5
        self.y_coord = 0.5
        self.border_color = np.array([1.0, 0.0, 0.0])
        self.opacity = 1.0
        self.min_depth = 0.0  # Not used for 2D images

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        self.anim_rate = params.get("anim_rate", self.anim_rate)
        self.x_coord = params.get("x_coord", self.x_coord)
        self.y_coord = params.get("y_coord", self.y_coord)
        self.border_color = params.get("border_color", self.border_color)
        self.opacity = params.get("opacity", self.opacity)

        # Ensure border_color is numpy array
        if not isinstance(self.border_color, np.ndarray):
            self.border_color = np.array(self.border_color)

        h, w = image.shape[:2]

        # Numba kernel expects uint8 image
        result = _radial_slit_scan_kernel(
            image,
            h,
            w,
            self.x_coord,
            self.y_coord,
            self.anim_rate,
            self.border_color,
            self.opacity,
            self.min_depth,
        )

        return result
