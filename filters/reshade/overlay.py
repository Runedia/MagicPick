"""
Overlay.fx 정확한 구현

Image Overlay with positioning
Original: FXShaders

Note: 외부 이미지 파일 로딩이 필요하므로,
사용자가 제공한 이미지를 overlay_image 파라미터로 전달받음
"""

import numpy as np

from filters.base_filter import BaseFilter


class OverlayFilter(BaseFilter):
    """
    Overlay - 이미지 오버레이

    Features:
    - 외부 이미지 오버레이
    - Opacity 조정
    - Stretch/Scale 조정
    - Center positioning
    - Aspect ratio preservation
    """

    def __init__(self):
        super().__init__("Overlay", "이미지 오버레이")

        # Parameters
        self.opacity = 1.0  # 0.0 ~ 1.0
        self.stretch = 1.0  # 0.0 ~ 1.0
        self.center = np.array([0.5, 0.5], dtype=np.float32)  # 0.0 ~ 1.0
        self.keep_aspect_ratio = False

        # Overlay image (RGBA)
        self.overlay_image = None

    def _scale_uv(self, uv, scale, center):
        """Scale UV coordinates around a center point"""
        return (uv - center) * scale + center

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Apply Overlay filter

        Args:
            image: RGB image (uint8, 0-255)
            **params: Filter parameters
                - opacity: Opacity (0.0 ~ 1.0, default 1.0)
                - stretch: Stretch factor (0.0 ~ 1.0, default 1.0)
                - center: Center position ([x, y], default [0.5, 0.5])
                - keep_aspect_ratio: Preserve aspect ratio
                    (True/False, default False)
                - overlay_image: RGBA overlay image (required)

        Returns:
            Image with overlay (uint8, 0-255)
        """
        # Update parameters
        self.opacity = params.get("opacity", self.opacity)
        self.stretch = params.get("stretch", self.stretch)
        if "center" in params:
            self.center = np.array(params["center"], dtype=np.float32)
        self.keep_aspect_ratio = params.get("keep_aspect_ratio", self.keep_aspect_ratio)

        if "overlay_image" in params:
            self.overlay_image = params["overlay_image"]

        if self.overlay_image is None:
            # No overlay image provided, return original
            return image

        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        h, w = img_float.shape[:2]

        # Convert overlay to float
        overlay_float = self.overlay_image.astype(np.float32) / 255.0
        overlay_h, overlay_w = overlay_float.shape[:2]

        # Calculate aspect ratio
        if self.keep_aspect_ratio:
            overlay_ar = (
                np.array([overlay_w / overlay_h, 1.0])
                if overlay_w > overlay_h
                else np.array([1.0, overlay_h / overlay_w])
            )
        else:
            overlay_ar = np.array([1.0, 1.0])

        # Calculate stretch
        screen_size = np.array([w, h], dtype=np.float32)
        overlay_size = np.array([overlay_w, overlay_h], dtype=np.float32)
        pixel_size = 1.0 / overlay_size

        if self.keep_aspect_ratio:
            corrected = screen_size * overlay_ar
        else:
            corrected = screen_size

        stretch_scale = (
            screen_size
            * pixel_size
            / (overlay_ar if not self.keep_aspect_ratio else 1.0)
        ) * (1.0 - self.stretch) + (corrected / screen_size) * self.stretch

        # Create result
        result = img_float.copy()

        # Process each pixel
        for y in range(h):
            for x in range(w):
                # Calculate UV coordinates
                uv = np.array([x / w, y / h], dtype=np.float32)

                # Scale UV around center
                uv_overlay = self._scale_uv(
                    uv, stretch_scale, np.array([self.center[0], 1.0 - self.center[1]])
                )

                # Sample overlay
                if 0.0 <= uv_overlay[0] <= 1.0 and 0.0 <= uv_overlay[1] <= 1.0:
                    overlay_y = int(uv_overlay[1] * overlay_h)
                    overlay_x = int(uv_overlay[0] * overlay_w)
                    overlay_y = min(overlay_y, overlay_h - 1)
                    overlay_x = min(overlay_x, overlay_w - 1)

                    overlay_color = overlay_float[overlay_y, overlay_x]

                    # Extract alpha
                    if overlay_float.shape[2] == 4:
                        alpha = overlay_color[3] * self.opacity
                        overlay_rgb = overlay_color[:3]
                    else:
                        alpha = self.opacity
                        overlay_rgb = overlay_color

                    # Blend
                    result[y, x] = img_float[y, x] * (1.0 - alpha) + overlay_rgb * alpha

        # Clamp and convert
        result = np.clip(result, 0, 1)
        return (result * 255).astype(np.uint8)
