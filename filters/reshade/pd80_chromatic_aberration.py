import cv2
import numpy as np

from filters.base_filter import BaseFilter


class PD80ChromaticAberrationFilter(BaseFilter):
    """
    PD80_06_Chromatic_Aberration.fx 구현

    다중 샘플링을 통한 스펙트럼 색수차 효과를 제공합니다.
    방사형(Radial) 및 종방향(Longitudinal) 모드를 지원합니다.
    """

    def __init__(self):
        super().__init__("PD80ChromaticAberration", "PD80 색수차")
        self.ca_type = 0  # 0: Center Radial, 1: Center Longitudinal, 2: Full Radial, 3: Full Longitudinal
        self.degrees = 135
        self.ca_global_width = -12.0
        self.sample_steps = 6  # Default reduced from 24 for performance
        self.ca_strength = 1.0

        # Center Weighted params
        self.ca_width = 1.0
        self.ca_curve = 1.0
        self.center_x = 0.0
        self.center_y = 0.0
        self.shape_x = 1.0
        self.shape_y = 1.0

    def _hue_to_rgb(self, h):
        r = np.abs(h * 6.0 - 3.0) - 1.0
        g = 2.0 - np.abs(h * 6.0 - 2.0)
        b = 2.0 - np.abs(h * 6.0 - 4.0)
        return np.clip(np.array([r, g, b]), 0.0, 1.0)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        if "ca_type" in params:
            self.ca_type = int(params["ca_type"])
        if "degrees" in params:
            self.degrees = int(params["degrees"])
        if "ca_global_width" in params:
            self.ca_global_width = float(params["ca_global_width"])
        if "sample_steps" in params:
            self.sample_steps = int(params["sample_steps"])
        if "ca_strength" in params:
            self.ca_strength = float(params["ca_strength"])

        # Center Weighted Params
        if "ca_width" in params:
            self.ca_width = float(params["ca_width"])
        if "ca_curve" in params:
            self.ca_curve = float(params["ca_curve"])
        if "center_x" in params:
            self.center_x = float(params["center_x"])
        if "center_y" in params:
            self.center_y = float(params["center_y"])
        if "shape_x" in params:
            self.shape_x = float(params["shape_x"])
        if "shape_y" in params:
            self.shape_y = float(params["shape_y"])

        img_float = image.astype(np.float32) / 255.0
        height, width = img_float.shape[:2]

        # Coordinate Grid (Normalized -1 to 1)
        # This is heavy, optimize if possible
        y_indices, x_indices = np.indices((height, width))

        # Normalize coordinates to -1 ~ 1, centered at (center_x, center_y)
        # coords = texcoord * 2.0 - (oX + 1.0, oY + 1.0)
        # texcoord is 0~1

        # Optimize: x_coords and y_coords as 1D arrays and broadcast?
        # But CA intensity calculation involves length(coords), which couples x and y.

        norm_x = (x_indices / width) * 2.0 - (self.center_x + 1.0)
        norm_y = (y_indices / height) * 2.0 - (self.center_y + 1.0)

        # uv for direction calculation
        uv_x = norm_x
        uv_y = norm_y

        # Apply Shape
        aspect = width / height
        sx = norm_x / (self.shape_x / aspect)
        sy = norm_y / self.shape_y

        # CA Intensity (Center Weighted)
        # caintensity = length(coords.xy) * CA_width_n
        caintensity_sq = sx * sx + sy * sy
        # Optimized length calculation to avoid sqrt if possible, but needed for linear relationship
        # But shader uses:
        # caintensity.y = caintensity.x * caintensity.x + 1.0f
        # caintensity.x = 1.0f - ( 1.0f / ( caintensity.y * caintensity.y ));
        # caintensity.x = pow( caintensity.x, CA_curve );

        len_coords = np.sqrt(caintensity_sq) * self.ca_width

        # Shader logic implementation
        ci_y = len_coords * len_coords + 1.0
        ci_x = 1.0 - (1.0 / (ci_y * ci_y))
        ci_x = np.power(np.maximum(ci_x, 0.0), self.ca_curve)

        # Type specific logic
        degrees_rad = np.radians(self.degrees)
        degrees_y_rad = np.radians(self.degrees + 90)

        c = 0.0
        s = 0.0
        final_intensity = ci_x  # This will be modified based on type

        if self.ca_type == 0:  # Radial
            c = np.cos(degrees_rad) * uv_x
            s = np.sin(degrees_y_rad) * uv_y
        elif self.ca_type == 1:  # Longitudinal
            c = np.cos(degrees_rad)
            s = np.sin(degrees_y_rad)
            # Shader uses caintensity as calculated
        elif self.ca_type == 2:  # Full screen Radial
            final_intensity = 1.0
            c = np.cos(degrees_rad) * uv_x
            s = np.sin(degrees_y_rad) * uv_y
        elif self.ca_type == 3:  # Full screen Longitudinal
            final_intensity = 1.0
            c = np.cos(degrees_rad)
            s = np.sin(degrees_y_rad)

        # Prepare Accumulators
        accum_color = np.zeros_like(img_float)
        accum_weight = np.zeros((3,), dtype=np.float32)

        # Scale CA
        # float caWidth = CA * ( max( BUFFER_WIDTH, BUFFER_HEIGHT ) / 1920.0f );
        ca_scale = self.ca_global_width * (max(width, height) / 1920.0)

        # Pre-calculate base offsets
        # offsetX = px * c * final_intensity
        # offsetY = py * s * final_intensity
        # px = 1/width, py = 1/height

        base_off_x = (1.0 / width) * c * final_intensity
        base_off_y = (1.0 / height) * s * final_intensity

        # Sampling Loop
        for i in range(self.sample_steps):
            progress = i / max(1, self.sample_steps - 1)

            # o2 = lerp( -caWidth, caWidth, i / o1 );
            o2 = -ca_scale + (ca_scale - (-ca_scale)) * progress

            # Current offset in pixels (normalized 0-1 space)
            cur_off_x = o2 * base_off_x
            cur_off_y = o2 * base_off_y

            # Apply remap
            # We need map_x and map_y in pixel coordinates for remap
            # map_x = (x_indices_norm + cur_off_x) * width
            # But x_indices is 0..W-1.
            # Normalized texcoord is x / width.
            # New texcoord is x/width + cur_off_x.
            # So New X is (x/width + cur_off_x) * width = x + cur_off_x * width

            map_x = (x_indices + cur_off_x * width).astype(np.float32)
            map_y = (y_indices + cur_off_y * height).astype(np.float32)

            sampled = cv2.remap(
                img_float,
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            # Hue Color
            hue_color = self._hue_to_rgb(progress)  # RGB weight

            accum_color += sampled * hue_color
            accum_weight += hue_color

        # Normalize
        # color.xyz /= dot( d.xyz, 0.333333f );
        weight_sum = np.dot(accum_weight, [0.333333, 0.333333, 0.333333])
        if weight_sum > 0:
            accum_color /= weight_sum

        # Blend with original based on strength
        # color.xyz = lerp( orig.xyz, color.xyz, CA_strength );
        result = img_float * (1.0 - self.ca_strength) + accum_color * self.ca_strength

        return (np.clip(result, 0.0, 1.0) * 255).astype(np.uint8)
