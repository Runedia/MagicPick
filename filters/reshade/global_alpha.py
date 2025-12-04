"""
GlobalAlpha.fx 정확한 구현

Global Alpha Control
Original: Marot Satil for GShade
"""

import numpy as np

from filters.base_filter import BaseFilter


class GlobalAlphaFilter(BaseFilter):
    """
    GlobalAlpha - 글로벌 알파 조정

    Features:
    - 전체 이미지의 alpha 채널 조정
    - 완전 투명 픽셀 무시 옵션
    """

    def __init__(self):
        super().__init__("GlobalAlpha", "글로벌 알파")

        # Parameters
        self.opacity = 1.0  # 0.0 ~ 1.0
        self.ignore_transparent = True  # 완전 투명 픽셀 무시

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Apply GlobalAlpha filter

        Args:
            image: RGB or RGBA image (uint8, 0-255)
            **params: Filter parameters
                - opacity: Alpha value (0.0 ~ 1.0, default 1.0)
                - ignore_transparent: Ignore fully transparent pixels
                    (True/False, default True)

        Returns:
            Image with adjusted alpha (uint8, 0-255)
        """
        # Update parameters
        self.opacity = params.get("opacity", self.opacity)
        self.ignore_transparent = params.get(
            "ignore_transparent", self.ignore_transparent
        )

        # Check if image has alpha channel
        if image.shape[2] == 3:
            # Add alpha channel
            img_with_alpha = np.dstack(
                [image, np.full((image.shape[0], image.shape[1]), 255, dtype=np.uint8)]
            )
        else:
            img_with_alpha = image.copy()

        # Convert to float
        alpha_float = img_with_alpha[:, :, 3].astype(np.float32) / 255.0

        # Apply opacity
        if self.ignore_transparent:
            # Only modify pixels with alpha > 0
            mask = alpha_float > 0.0
            alpha_float[mask] = self.opacity
        else:
            # Modify all pixels
            alpha_float[:] = self.opacity

        # Convert back
        img_with_alpha[:, :, 3] = (alpha_float * 255).astype(np.uint8)

        return img_with_alpha
