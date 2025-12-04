"""
Emphasize.fx 정확한 구현

DoF-like emphasis effect with depth
Original: Infuse Project / Otis
3D emphasis code by SirCobra
"""

import numpy as np

from filters.base_filter import BaseFilter


class EmphasizeFilter(BaseFilter):
    """
    Emphasize - 강조 효과

    Features:
    - DoF 스타일의 depth 기반 desaturation
    - Manual/Spherical focus point
    - Color blending
    
    Note: 정적 이미지이므로 depth 기능은 제한적.
    사용자가 제공한 depth map이 있어야 완전한 동작 가능.
    """

    def __init__(self):
        super().__init__("Emphasize", "강조 효과")

        # Parameters
        self.focus_depth = 0.026  # 0.000 ~ 1.000
        self.focus_range_depth = 0.001  # 0.000 ~ 1.000
        self.focus_edge_depth = 0.050  # 0.000 ~ 1.000
        self.spherical = False
        self.sphere_fov = 75  # 1 ~ 180
        self.sphere_focus_horizontal = 0.5  # 0.0 ~ 1.0
        self.sphere_focus_vertical = 0.5 # 0.0 ~ 1.0
        self.blend_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.blend_factor = 0.0  # 0.0 ~ 1.0
        self.effect_factor = 0.9  # 0.0 ~ 1.0
        
        # Depth map (외부에서 제공 가능)
        self.depth_map = None

    def _calculate_depth_diff_coc(self, texcoord, h, w):
        """Calculate depth difference circle of confusion"""
        if self.depth_map is None:
            # No depth map, use distance from center as pseudo-depth
            scene_depth = np.sqrt(
                (texcoord[:, :, 0] - 0.5) ** 2 +
                (texcoord[:, :, 1] - 0.5) ** 2
            )
        else:
            scene_depth = self.depth_map
        
        scene_focus = self.focus_depth
        desaturate_full_range = self.focus_range_depth + self.focus_edge_depth
        
        if self.spherical:
            # Spherical mode
            offset_x = (texcoord[:, :, 0] - self.sphere_focus_horizontal) * w
            offset_y = (texcoord[:, :, 1] - self.sphere_focus_vertical) * h
            
            degree_per_pixel = self.sphere_fov / w
            fov_difference = np.sqrt(offset_x ** 2 + offset_y ** 2) * degree_per_pixel
            
            # Law of cosines
            fov_rad = fov_difference * (2 * np.pi / 360)
            depth_diff = np.sqrt(
                scene_depth ** 2 + scene_focus ** 2
                - 2 * scene_depth * scene_focus * np.cos(fov_rad)
            )
        else:
            # Planar mode
            depth_diff = np.abs(scene_depth - scene_focus)
        
        # Smoothstep
        coc = np.where(
            depth_diff > desaturate_full_range,
            1.0,
            np.clip(
                (depth_diff - self.focus_range_depth) /
                max(self.focus_edge_depth, 1e-6),
                0, 1
            )
        )
        
        # Smooth the transition (smoothstep)
        coc = coc * coc * (3.0 - 2.0 * coc)
        
        return coc

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Apply Emphasize filter

        Args:
            image: RGB image (uint8, 0-255)
            **params: Filter parameters
                - focus_depth: Focus depth (0.0 ~ 1.0, default 0.026)
                - focus_range_depth: Focus range (0.0 ~ 1.0, default 0.001)
                - focus_edge_depth: Focus edge (0.0 ~ 1.0, default 0.050)
                - spherical: Enable spherical mode (True/False, default False)
                - sphere_fov: Field of view (1 ~ 180, default 75)
                - sphere_focus_horizontal: Focus point X (0.0 ~ 1.0, default 0.5)
                - sphere_focus_vertical: Focus point Y (0.0 ~ 1.0, default 0.5)
                - blend_color: Blend color ([R,G,B], default [0,0,0])
                - blend_factor: Blend factor (0.0 ~ 1.0, default 0.0)
                - effect_factor: Effect factor (0.0 ~ 1.0, default 0.9)
                - depth_map: Optional depth map (H, W) array

        Returns:
            Emphasized image (uint8, 0-255)
        """
        # Update parameters
        self.focus_depth = params.get("focus_depth", self.focus_depth)
        self.focus_range_depth = params.get("focus_range_depth", self.focus_range_depth)
        self.focus_edge_depth = params.get("focus_edge_depth", self.focus_edge_depth)
        self.spherical = params.get("spherical", self.spherical)
        self.sphere_fov = params.get("sphere_fov", self.sphere_fov)
        self.sphere_focus_horizontal = params.get(
            "sphere_focus_horizontal", self.sphere_focus_horizontal
        )
        self.sphere_focus_vertical = params.get(
            "sphere_focus_vertical", self.sphere_focus_vertical
        )
        if "blend_color" in params:
            self.blend_color = np.array(params["blend_color"], dtype=np.float32)
        self.blend_factor = params.get("blend_factor", self.blend_factor)
        self.effect_factor = params.get("effect_factor", self.effect_factor)
        if "depth_map" in params:
            self.depth_map = params["depth_map"]

        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        h, w = img_float.shape[:2]

        # Create texcoord grid
        y_coords, x_coords = np.meshgrid(
            np.linspace(0, 1, h), np.linspace(0, 1, w), indexing='ij'
        )
        texcoord = np.stack([x_coords, y_coords], axis=2)

        # Calculate depth CoC
        depth_coc = self._calculate_depth_diff_coc(texcoord, h, w)

        # Grayscale conversion
        greyscale = (img_float[:, :, 0] + img_float[:, :, 1] + img_float[:, :, 2]) / 3.0
        greyscale = np.stack([greyscale, greyscale, greyscale], axis=2)

        # Blend with color
        des_color = greyscale * (1 - self.blend_factor) + self.blend_color * self.blend_factor

        # Apply effect
        depth_coc_3d = np.expand_dims(depth_coc, axis=2)
        result = img_float * (1 - depth_coc_3d * self.effect_factor) + \
                 des_color * (depth_coc_3d * self.effect_factor)

        # Clamp and convert
        result = np.clip(result, 0, 1)
        return (result * 255).astype(np.uint8)
