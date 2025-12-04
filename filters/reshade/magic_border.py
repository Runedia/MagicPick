"""
MagicBorder.fx 구현

Magic Border with depth-based frame
Original: Frans Bouma (Otis / Infuse Project)

Note: Depth 기능은 사용자 제공 depth map 필요
"""

import numpy as np

from filters.base_filter import BaseFilter


class MagicBorderFilter(BaseFilter):
    """
    MagicBorder - 매직 테두리

    Features:
    - Customizable picture frame
    - Border and frame colors
    - Depth-based visibility (선택적)
    """

    def __init__(self):
        super().__init__("MagicBorder", "매직 테두리")

        # Corner depths (0.0 ~ 300.0, normalized to 0~1)
        self.left_top_depth = 1.0
        self.right_top_depth = 1.0
        self.right_bottom_depth = 1.0
        self.left_bottom_depth = 1.0

        # Picture frame coords (0.0 ~ 1.0)
        self.frame_left_top = np.array([0.1, 0.1], dtype=np.float32)
        self.frame_right_top = np.array([0.9, 0.1], dtype=np.float32)
        self.frame_right_bottom = np.array([0.9, 0.9], dtype=np.float32)
        self.frame_left_bottom = np.array([0.1, 0.9], dtype=np.float32)

        # Colors (RGBA)
        self.border_color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.frame_color = np.array([0.7, 0.7, 0.7, 1.0], dtype=np.float32)

        # Debug
        self.show_depths = False

        # Optional depth map
        self.depth_map = None

    def _is_point_in_polygon(self, point, vertices):
        """
        Point-in-polygon test
        http://www.jeffreythompson.org/collision-detection/poly-point.php
        """
        is_inside = False
        n = len(vertices)

        for i in range(n):
            j = (i + 1) % n
            vi = vertices[i]
            vj = vertices[j]

            if ((vi[1] >= point[1] and vj[1] < point[1]) or 
                (vi[1] < point[1] and vj[1] >= point[1])):
                if point[0] < (vj[0] - vi[0]) * (point[1] - vi[1]) / (vj[1] - vi[1]) + vi[0]:
                    is_inside = not is_inside

        return is_inside

    def _calculate_frame_depth(self, coord):
        """Calculate interpolated depth at coordinate"""
        # Distance-weighted average of corner depths
        corners = [
            (np.array([0, 0]), self.left_top_depth),
            (np.array([1, 0]), self.right_top_depth),
            (np.array([1, 1]), self.right_bottom_depth),
            (np.array([0, 1]), self.left_bottom_depth)
        ]

        total_weight = 0
        weighted_depth = 0

        for corner_pos, corner_depth in corners:
            # Distance to corner
            distance = np.linalg.norm(coord - corner_pos)
            # Weight is inverse of distance (closer = stronger)
            weight = 1.0 - distance
            weighted_depth += (corner_depth / 1000.0) * weight
            total_weight += weight

        return np.clip(weighted_depth / total_weight, 0, 1)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Apply MagicBorder filter

        Args:
            image: RGB image (uint8, 0-255)
            **params: Filter parameters
                - left_top_depth: Left top corner depth (0~300, default 1)
                - right_top_depth: Right top corner depth (0~300, default 1)
                - right_bottom_depth: Right bottom corner depth
                    (0~300, default 1)
                - left_bottom_depth: Left bottom corner depth (0~300, default 1)
                - frame_left_top: Frame left top coord ([x,y], default [0.1,0.1])
                - frame_right_top: Frame right top coord ([x,y], default [0.9,0.1])
                - frame_right_bottom: Frame right bottom coord
                    ([x,y], default [0.9,0.9])
                - frame_left_bottom: Frame left bottom coord
                    ([x,y], default [0.1,0.9])
                - border_color: Border color RGBA (default [1,1,1,1])
                - frame_color: Frame color RGBA (default [0.7,0.7,0.7,1])
                - show_depths: Show depth visualization (True/False, default False)
                - depth_map: Optional depth map (H, W) array

        Returns:
            Image with magic border (uint8, 0-255)
        """
        # Update parameters
        self.left_top_depth = params.get("left_top_depth", self.left_top_depth)
        self.right_top_depth = params.get("right_top_depth", self.right_top_depth)
        self.right_bottom_depth = params.get(
            "right_bottom_depth", self.right_bottom_depth
        )
        self.left_bottom_depth = params.get(
            "left_bottom_depth", self.left_bottom_depth
        )

        if "frame_left_top" in params:
            self.frame_left_top = np.array(params["frame_left_top"], dtype=np.float32)
        if "frame_right_top" in params:
            self.frame_right_top = np.array(params["frame_right_top"], dtype=np.float32)
        if "frame_right_bottom" in params:
            self.frame_right_bottom = np.array(
                params["frame_right_bottom"], dtype=np.float32
            )
        if "frame_left_bottom" in params:
            self.frame_left_bottom = np.array(
                params["frame_left_bottom"], dtype=np.float32
            )

        if "border_color" in params:
            self.border_color = np.array(params["border_color"], dtype=np.float32)
        if "frame_color" in params:
            self.frame_color = np.array(params["frame_color"], dtype=np.float32)

        self.show_depths = params.get("show_depths", self.show_depths)

        if "depth_map" in params:
            self.depth_map = params["depth_map"]

        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        h, w = img_float.shape[:2]

        # Define frame vertices
        vertices = [
            self.frame_left_top,
            self.frame_right_top,
            self.frame_right_bottom,
            self.frame_left_bottom
        ]

        # Process each pixel
        result = img_float.copy()

        for y in range(h):
            for x in range(w):
                # Normalized coordinates
                coord = np.array([x / w, y / h], dtype=np.float32)

                # Check if in picture area
                is_in_picture = self._is_point_in_polygon(coord, vertices)

                # Calculate frame depth
                frame_depth = self._calculate_frame_depth(coord)

                # Get pixel depth
                if self.depth_map is not None:
                    pixel_depth = self.depth_map[y, x]
                else:
                    # No depth map, always show border
                    pixel_depth = 0.0

                # Select color
                if is_in_picture:
                    color = self.frame_color
                else:
                    color = self.border_color

                # Apply border/frame if pixel is behind frame
                if pixel_depth > frame_depth:
                    result[y, x] = (
                        img_float[y, x] * (1.0 - color[3]) +
                        color[:3] * color[3]
                    )

                # Debug: show depths
                if self.show_depths:
                    result[y, x] = np.array([frame_depth, frame_depth, frame_depth])

        # Clamp and convert
        result = np.clip(result, 0, 1)
        return (result * 255).astype(np.uint8)
