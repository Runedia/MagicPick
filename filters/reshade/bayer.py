"""
BAYER.fx 구현

Bayer Pattern Dithering
Original: Ann-ReShade (cshade)
"""

import numpy as np

from filters.base_filter import BaseFilter


class BayerFilter(BaseFilter):
    """
    Bayer Filter - 베이어 패턴 디더링

    Features:
    - 디지털 카메라의 Bayer 마스크 시뮬레이션
    - Pixelation 효과
    - RGB 채널별 색상 조정
    """

    def __init__(self):
        super().__init__("Bayer", "베이어 디더링")

        # Parameters
        self.pixelation_size_x = 2  # 1 ~ 50
        self.pixelation_size_y = 2  # 1 ~ 50
        self.rebayer_enabled = True
        # Bayer color multipliers
        self.rebay_red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.rebay_green = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.rebay_blue = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Apply Bayer filter

        Args:
            image: RGB image (uint8, 0-255)
            **params: Filter parameters
                - pixelation_size_x: Horizontal pixel size (1 ~ 50, default 2)
                - pixelation_size_y: Vertical pixel size (1 ~ 50, default 2)
                - rebayer_enabled: Enable rebayer effect (True/False, default True)
                - rebay_red: Red channel color ([R,G,B], default [1,0,0])
                - rebay_green: Green channel color ([R,G,B], default [0,1,0])
                - rebay_blue: Blue channel color ([R,G,B], default [0,0,1])

        Returns:
            Bayer dithered image (uint8, 0-255)
        """
        # Update parameters
        self.pixelation_size_x = params.get("pixelation_size_x", self.pixelation_size_x)
        self.pixelation_size_y = params.get("pixelation_size_y", self.pixelation_size_y)
        self.rebayer_enabled = params.get("rebayer_enabled", self.rebayer_enabled)
        if "rebay_red" in params:
            self.rebay_red = np.array(params["rebay_red"], dtype=np.float32)
        if "rebay_green" in params:
            self.rebay_green = np.array(params["rebay_green"], dtype=np.float32)
        if "rebay_blue" in params:
            self.rebay_blue = np.array(params["rebay_blue"], dtype=np.float32)

        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        h, w = img_float.shape[:2]

        # Calculate pixel blocks
        block_h = self.pixelation_size_y
        block_w = self.pixelation_size_x

        # Pixelate: downsample then upsample
        new_h = h // block_h
        new_w = w // block_w

        result = np.zeros_like(img_float)

        for by in range(new_h):
            for bx in range(new_w):
                # Sample center of block
                center_y = by * block_h + block_h // 2
                center_x = bx * block_w + block_w // 2

                if center_y >= h:
                    center_y = h - 1
                if center_x >= w:
                    center_x = w - 1

                color = img_float[center_y, center_x].copy()

                # Apply Bayer pattern if enabled
                if self.rebayer_enabled:
                    # Bayer pattern:
                    # R G
                    # G B
                    # cellp = (bx % 2, by % 2)
                    cellp_x = bx % 2
                    cellp_y = by % 2

                    if cellp_x == cellp_y:
                        if cellp_x == 0:
                            # Red pixel
                            color = color * self.rebay_red
                        else:
                            # Blue pixel
                            color = color * self.rebay_blue
                    else:
                        # Green pixel
                        color = color * self.rebay_green

                # Fill block
                y_start = by * block_h
                y_end = min((by + 1) * block_h, h)
                x_start = bx * block_w
                x_end = min((bx + 1) * block_w, w)

                result[y_start:y_end, x_start:x_end] = color

        # Clamp and convert
        result = np.clip(result, 0, 1)
        return (result * 255).astype(np.uint8)
