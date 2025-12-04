import numpy as np
from numba import njit, prange

from filters.base_filter import BaseFilter
from filters.reshade.numba_helpers import clamp, lerp, saturate


@njit(fastmath=True, inline="always", cache=True)
def get_radius(theta, rcp_f, k):
    if k > 0.0:
        return np.tan(abs(k) * theta) / rcp_f / abs(k)
    elif k < 0.0:
        return np.sin(abs(k) * theta) / rcp_f / abs(k)
    else:
        return theta / rcp_f

@njit(fastmath=True, inline="always", cache=True)
def get_theta(radius, rcp_f, k):
    if k > 0.0:
        return np.arctan(abs(k) * radius * rcp_f) / abs(k)
    elif k < 0.0:
        # asin domain check
        val = abs(k) * radius * rcp_f
        if val > 1.0: val = 1.0
        if val < -1.0: val = -1.0
        return np.arcsin(val) / abs(k)
    else:
        return radius * rcp_f

@njit(fastmath=True, inline="always", cache=True)
def get_vignette(theta, r, rcp_f):
    if r * rcp_f == 0: return 1.0
    return np.sin(theta) / r / rcp_f

@njit(parallel=True, fastmath=True, cache=True)
def _perfect_perspective_kernel(img, h, w, fov_angle, fov_type, k, ky, vignette_intensity, cropping_factor, border_color):
    out = np.empty((h, w, 3), dtype=np.uint8)
    
    # Constants
    PI = 3.1415926536
    
    # View Proportions (Aspect Ratio)
    # Shader: normalize(BUFFER_SCREEN_SIZE)
    # If W=1920, H=1080 -> normalize(1920, 1080)
    # len = sqrt(1920^2 + 1080^2) = 2202.9
    # vp.x = 0.87, vp.y = 0.49
    screen_len = np.sqrt(float(w*w + h*h))
    vp_x = w / screen_len
    vp_y = h / screen_len
    
    # Half Omega
    half_omega = np.radians(fov_angle * 0.5)
    
    # Radius of Omega
    radius_of_omega = 0.0
    if fov_type == 1: # Diagonal
        radius_of_omega = 1.0
    elif fov_type == 2: # Vertical
        radius_of_omega = vp_y
    elif fov_type == 3: # 4:3
        radius_of_omega = vp_y * 4.0 / 3.0
    elif fov_type == 4: # 16:9
        radius_of_omega = vp_y * 16.0 / 9.0
    else: # Horizontal (default)
        radius_of_omega = vp_x
        
    # Reciprocal Focal Length
    rcp_focal = 1.0
    # get_rcp_focal(halfOmega, radiusOfOmega, k) -> get_radius(...)
    if k > 0.0:
        rcp_focal = np.tan(abs(k) * half_omega) / radius_of_omega / abs(k)
    elif k < 0.0:
        rcp_focal = np.sin(abs(k) * half_omega) / radius_of_omega / abs(k)
    else:
        rcp_focal = half_omega / radius_of_omega
        
    # Cropping Scalar (Simplified logic from shader)
    # We will skip the complex binary search for cropping for now and use a simpler approximation or just 1.0 if cropping is 0.
    # Shader calculates croppingHorizontal, croppingVertical, etc.
    # For now, let's assume cropping_scalar = 1.0 if cropping_factor is not used much, or implement simplified.
    # Let's try to implement at least the horizontal/vertical cropping.
    
    # Horizontal point radius (at edge)
    # atan(tan(halfOmega)/radiusOfOmega*viewProportions.x)
    angle_h = np.arctan(np.tan(half_omega) / radius_of_omega * vp_x)
    crop_h = get_radius(angle_h, rcp_focal, k) / vp_x
    
    # Vertical point radius
    angle_v = np.arctan(np.tan(half_omega) / radius_of_omega * vp_y)
    crop_v = get_radius(angle_v, rcp_focal, ky) / vp_y
    
    circular_fisheye = max(crop_h, crop_v)
    cropped_circle = min(crop_h, crop_v)
    full_frame = np.sqrt(crop_h*crop_h + crop_v*crop_v) # Approx diagonal
    
    cropping_scalar = 1.0
    if cropping_factor < 0.5:
        t = max(cropping_factor * 2.0, 0.0)
        cropping_scalar = lerp(circular_fisheye, cropped_circle, t)
    else:
        t = min(cropping_factor * 2.0 - 1.0, 1.0)
        cropping_scalar = lerp(cropped_circle, full_frame, t)
        
    # Pre-calculate conversion factor
    # toUvCoord = radiusOfOmega/(tan(halfOmega)*viewProportions);
    to_uv_scale_x = radius_of_omega / (np.tan(half_omega) * vp_x)
    to_uv_scale_y = radius_of_omega / (np.tan(half_omega) * vp_y)

    for y in prange(h):
        for x in range(w):
            # UV 0..1
            u = x / w
            v = y / h
            
            # View Coord (Centered, Scaled by Aspect Ratio)
            # texCoord.x = viewCoord.x =  position.x; (position is -1..1 or similar? Shader says position.x = vertexId==2? 3f :-1f...)
            # Shader: texCoord = texCoord*0.5+0.5; -> So viewCoord is -1..1 range before normalization?
            # "viewCoord *= viewProportions;"
            
            vx = (u - 0.5) * 2.0 * vp_x
            vy = (v - 0.5) * 2.0 * vp_y # Inverted Y? Shader: texCoord.y = -position.y.
            # Let's keep Y consistent with image coords (0 top, 1 bottom).
            # If we want standard Cartesian, Y up. But images are Y down.
            # Distortion is usually radial so sign of Y doesn't matter for radius, but matters for direction.
            # Let's use (v - 0.5) * 2.0.
            
            # Apply Cropping Scalar
            vx *= cropping_scalar
            vy *= cropping_scalar
            
            radius = np.sqrt(vx*vx + vy*vy)
            
            # Get Theta
            # Aximorphic: theta from K and Ky
            # theta2 = get_theta(radius, rcp_focal, K/Ky)
            # phi weights = viewCoord^2 / (vx^2 + vy^2)
            
            if radius < 1e-6:
                theta = 0.0
            else:
                theta_k = get_theta(radius, rcp_focal, k)
                theta_ky = get_theta(radius, rcp_focal, ky)
                
                # Phi weights
                vx2 = vx*vx
                vy2 = vy*vy
                sum_sq = vx2 + vy2
                wx = vx2 / sum_sq
                wy = vy2 / sum_sq
                
                theta = wx * theta_k + wy * theta_ky
            
            # Vignette
            vig = 1.0
            if vignette_intensity > 0.0:
                v_val = get_vignette(theta, radius, rcp_focal)
                # exp(log(v)*clamp) -> v^clamp
                vig = pow(v_val, clamp(vignette_intensity, 0.0, 4.0))
            
            # Rectilinear Transform
            # viewCoord = tan(theta) * normalize(viewCoord)
            if radius > 1e-6:
                tan_theta = np.tan(theta)
                vx_new = tan_theta * (vx / radius)
                vy_new = tan_theta * (vy / radius)
            else:
                vx_new = 0.0
                vy_new = 0.0
                
            # Back to UV
            # viewCoord *= toUvCoord
            ux_new = vx_new * to_uv_scale_x
            uy_new = vy_new * to_uv_scale_y
            
            # 0..1
            final_u = ux_new * 0.5 + 0.5
            final_v = uy_new * 0.5 + 0.5
            
            # Sample
            sx = final_u * w
            sy = final_v * h
            
            if sx < 0 or sx >= w - 1 or sy < 0 or sy >= h - 1:
                # Border
                out[y, x, 0] = int(border_color[0] * 255)
                out[y, x, 1] = int(border_color[1] * 255)
                out[y, x, 2] = int(border_color[2] * 255)
            else:
                # Nearest neighbor for speed
                ix = int(sx)
                iy = int(sy)
                
                r = img[iy, ix, 0]
                g = img[iy, ix, 1]
                b = img[iy, ix, 2]
                
                # Apply Vignette
                if vignette_intensity > 0.0:
                    r = float(r) * vig
                    g = float(g) * vig
                    b = float(b) * vig
                    
                out[y, x, 0] = saturate(r / 255.0) * 255
                out[y, x, 1] = saturate(g / 255.0) * 255
                out[y, x, 2] = saturate(b / 255.0) * 255
                
    return out

class PerfectPerspectiveFilter(BaseFilter):
    def __init__(self):
        super().__init__("PerfectPerspective", "퍼펙트 퍼스펙티브")
        self.fov_angle = 90
        self.fov_type = 0 # 0:Horizontal, 1:Diagonal, 2:Vertical, 3:4:3, 4:16:9
        self.k = 0.5
        self.ky = 0.5
        self.vignette_intensity = 0.0 # Default 1.0 in shader, but usually 0 is better for default
        self.cropping_factor = 0.5
        self.border_color = np.array([0.027, 0.027, 0.027])
        
    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        self.fov_angle = int(params.get("FovAngle", self.fov_angle))
        self.fov_type = int(params.get("FovType", self.fov_type))
        self.k = float(params.get("K", self.k))
        self.ky = float(params.get("Ky", self.ky))
        self.vignette_intensity = float(params.get("VignetteIntensity", self.vignette_intensity))
        self.cropping_factor = float(params.get("CroppingFactor", self.cropping_factor))
        self.border_color = params.get("BorderColor", self.border_color)
        
        if not isinstance(self.border_color, np.ndarray):
            self.border_color = np.array(self.border_color)
            
        h, w = image.shape[:2]
        
        result = _perfect_perspective_kernel(
            image, h, w,
            self.fov_angle, self.fov_type, self.k, self.ky,
            self.vignette_intensity, self.cropping_factor, self.border_color
        )
        
        return result
