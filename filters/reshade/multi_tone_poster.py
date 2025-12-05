"""
MultiTonePoster.fx 정확한 구현

Multi-tone posterization effect
Original: Daodan
"""

import numpy as np

from filters.base_filter import BaseFilter


class MultiTonePosterFilter(BaseFilter):
    """
    MultiTonePoster - 멀티톤 포스터

    Features:
    - 4-color posterization
    - Pattern-based transitions
    - Luma-based color mapping
    """

    def __init__(self):
        super().__init__("MultiTonePoster", "멀티톤 포스터")

        # Color parameters
        self.color1 = np.array([0.0, 0.05, 0.17, 1.0], dtype=np.float32)
        self.color2 = np.array([0.20, 0.16, 0.25, 1.0], dtype=np.float32)
        self.color3 = np.array([1.0, 0.16, 0.10, 1.0], dtype=np.float32)
        self.color4 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        # Pattern parameters (0=Linear, 1=Vertical, 2=Horizontal, 3=Squares)
        self.pattern12 = 3
        self.width12 = 1
        self.pattern23 = 3
        self.width23 = 1
        self.pattern34 = 2
        self.width34 = 1

        # Effect strength
        self.strength = 1.0  # 0.0 ~ 1.0

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Apply MultiTonePoster filter

        Args:
            image: RGB image (uint8, 0-255)
            **params: Filter parameters
                - color1: Color 1 RGBA (default [0, 0.05, 0.17, 1])
                - color2: Color 2 RGBA (default [0.2, 0.16, 0.25, 1])
                - color3: Color 3 RGBA (default [1, 0.16, 0.1, 1])
                - color4: Color 4 RGBA (default [1, 1, 1, 1])
                - pattern12: Pattern 1-2 (0~3, default 3)
                - width12: Width 1-2 (1~10, default 1)
                - pattern23: Pattern 2-3 (0~3, default 3)
                - width23: Width 2-3 (1~10, default 1)
                - pattern34: Pattern 3-4 (0~3, default 2)
                - width34: Width 3-4 (1~10, default 1)
                - strength: Effect strength (0.0~1.0, default 1.0)

        Returns:
            Posterized image (uint8, 0-255)
        """
        # Update parameters
        if "color1" in params:
            self.color1 = np.array(params["color1"], dtype=np.float32)
        if "color2" in params:
            self.color2 = np.array(params["color2"], dtype=np.float32)
        if "color3" in params:
            self.color3 = np.array(params["color3"], dtype=np.float32)
        if "color4" in params:
            self.color4 = np.array(params["color4"], dtype=np.float32)
        self.pattern12 = params.get("pattern12", self.pattern12)
        self.width12 = params.get("width12", self.width12)
        self.pattern23 = params.get("pattern23", self.pattern23)
        self.width23 = params.get("width23", self.width23)
        self.pattern34 = params.get("pattern34", self.pattern34)
        self.width34 = params.get("width34", self.width34)
        self.strength = params.get("strength", self.strength)

        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        h, w = img_float.shape[:2]

        # Calculate luma
        luma = np.sum(img_float * np.array([0.2126, 0.7151, 0.0721]), axis=2)

        # Create result
        result = np.zeros_like(img_float)

        # Process each pixel
        for y in range(h):
            for x in range(w):
                # Calculate stripe factors for each pattern
                stripe_factor = np.zeros(12, dtype=np.float32)

                # Pattern 1-2
                stripe_factor[0] = 0.5  # Linear
                stripe_factor[1] = (
                    1.0 if (x % (self.width12 * 2)) < self.width12 else 0.0
                )  # Vertical
                stripe_factor[2] = (
                    1.0 if (y % (self.width12 * 2)) < self.width12 else 0.0
                )  # Horizontal
                stripe_factor[3] = (
                    1.0 if (stripe_factor[1] + stripe_factor[2]) == 0.0 else 0.0
                )  # Squares

                # Pattern 2-3
                stripe_factor[4] = 0.5
                stripe_factor[5] = (
                    1.0 if (x % (self.width23 * 2)) < self.width23 else 0.0
                )
                stripe_factor[6] = (
                    1.0 if (y % (self.width23 * 2)) < self.width23 else 0.0
                )
                stripe_factor[7] = (
                    1.0 if (stripe_factor[5] + stripe_factor[6]) == 0.0 else 0.0
                )

                # Pattern 3-4
                stripe_factor[8] = 0.5
                stripe_factor[9] = (
                    1.0 if (x % (self.width34 * 2)) < self.width34 else 0.0
                )
                stripe_factor[10] = (
                    1.0 if (y % (self.width34 * 2)) < self.width34 else 0.0
                )
                stripe_factor[11] = (
                    1.0 if (stripe_factor[9] + stripe_factor[10]) == 0.0 else 0.0
                )

                # Create color array
                colors = np.zeros((7, 4), dtype=np.float32)
                colors[0] = self.color1
                colors[2] = self.color2
                colors[4] = self.color3
                colors[6] = self.color4

                # Interpolate transition colors using patterns
                factor1 = stripe_factor[self.pattern12]
                colors[1] = colors[0] * (1.0 - factor1) + colors[2] * factor1

                factor2 = stripe_factor[self.pattern23 + 4]
                colors[3] = colors[2] * (1.0 - factor2) + colors[4] * factor2

                factor3 = stripe_factor[self.pattern34 + 8]
                colors[5] = colors[4] * (1.0 - factor3) + colors[6] * factor3

                # Blend with original based on alpha
                pixel_color = img_float[y, x]
                blended_colors = np.zeros_like(colors)
                blended_colors[0] = (
                    pixel_color * (1.0 - colors[0, 3]) + colors[0, :3] * colors[0, 3]
                )
                blended_colors[1] = pixel_color * (
                    1.0 - (colors[0, 3] + colors[2, 3]) / 2.0
                ) + colors[1, :3] * ((colors[0, 3] + colors[2, 3]) / 2.0)
                blended_colors[2] = (
                    pixel_color * (1.0 - colors[2, 3]) + colors[2, :3] * colors[2, 3]
                )
                blended_colors[3] = pixel_color * (
                    1.0 - (colors[2, 3] + colors[4, 3]) / 2.0
                ) + colors[3, :3] * ((colors[2, 3] + colors[4, 3]) / 2.0)
                blended_colors[4] = (
                    pixel_color * (1.0 - colors[4, 3]) + colors[4, :3] * colors[4, 3]
                )
                blended_colors[5] = pixel_color * (
                    1.0 - (colors[4, 3] + colors[6, 3]) / 2.0
                ) + colors[5, :3] * ((colors[4, 3] + colors[6, 3]) / 2.0)
                blended_colors[6] = (
                    pixel_color * (1.0 - colors[6, 3]) + colors[6, :3] * colors[6, 3]
                )

                # Select color based on luma
                num_colors = 7
                color_index = int(np.floor(luma[y, x] * num_colors))
                color_index = min(color_index, num_colors - 1)

                # Apply strength
                result[y, x] = (
                    pixel_color * (1.0 - self.strength)
                    + blended_colors[color_index] * self.strength
                )

        # Clamp and convert
        result = np.clip(result, 0, 1)
        return (result * 255).astype(np.uint8)
