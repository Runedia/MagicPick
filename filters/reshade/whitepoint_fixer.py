import numpy as np

from filters.base_filter import BaseFilter


class WhitepointFixerFilter(BaseFilter):
    """
    WhitepointFixer.fx 구현

    이미지의 화이트 포인트를 조정하여 노출을 보정합니다.
    Manual 모드와 Automatic 모드를 지원합니다.
    """

    def __init__(self):
        super().__init__("WhitepointFixer", "화이트 포인트 수정")
        self.mode = 0  # 0: Manual, 1: Colorpicker (Not Impl), 2: Automatic
        self.whitepoint = 1.0  # Manual mode default

        # Automatic mode settings
        self.grayscale_formula = 0  # 0: Average, 1: Max, 2: Luma
        self.minimum_whitepoint = 0.8
        self.remapped_whitepoint = 1.0
        # Transition speed is for temporal smoothing, not applicable for single image

    def _get_grayscale(self, img, formula):
        if formula == 0:  # Average
            # dot(color, 0.333)
            return np.mean(img, axis=2)
        elif formula == 1:  # Max
            return np.max(img, axis=2)
        elif formula == 2:  # Luma
            # ITU-R BT.709
            return np.dot(img, [0.2126, 0.7152, 0.0722])
        return np.mean(img, axis=2)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        if "mode" in params:
            self.mode = int(params["mode"])
        if "whitepoint" in params:
            self.whitepoint = float(params["whitepoint"])
        if "grayscale_formula" in params:
            self.grayscale_formula = int(params["grayscale_formula"])
        if "minimum_whitepoint" in params:
            self.minimum_whitepoint = float(params["minimum_whitepoint"])
        if "remapped_whitepoint" in params:
            self.remapped_whitepoint = float(params["remapped_whitepoint"])

        img_float = image.astype(np.float32) / 255.0

        final_whitepoint = 1.0

        if self.mode == 0:  # Manual
            final_whitepoint = self.whitepoint

        elif self.mode == 2:  # Automatic
            # Find max value in the image according to formula
            gray = self._get_grayscale(img_float, self.grayscale_formula)
            max_val = np.max(gray)

            if max_val < self.minimum_whitepoint:
                final_whitepoint = self.remapped_whitepoint
            else:
                final_whitepoint = max_val

        else:
            # Fallback for unsupported modes (like Colorpicker)
            final_whitepoint = self.whitepoint

        # Apply correction
        # color.rgb /= max(whitepoint, 1e-6);
        final_whitepoint = max(final_whitepoint, 1e-6)

        result = img_float / final_whitepoint

        return (np.clip(result, 0.0, 1.0) * 255).astype(np.uint8)
