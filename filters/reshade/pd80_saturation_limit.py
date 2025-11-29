import cv2
import numpy as np

from filters.base_filter import BaseFilter


class PD80SaturationLimitFilter(BaseFilter):
    """
    PD80_04_Saturation_Limit.fx 구현

    이미지의 채도(Saturation)가 특정 값을 넘지 않도록 제한합니다.
    """

    def __init__(self):
        super().__init__("PD80SaturationLimit", "PD80 채도 제한")
        self.saturation_limit = 1.0  # 0.0 ~ 1.0

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        if "saturation_limit" in params:
            self.saturation_limit = float(params["saturation_limit"])

        # BGR to HLS (OpenCV는 HLS 사용, ReShade 코드는 HSL 사용하지만 S 채널 처리는 동일)
        # H: 0-179, L: 0-255, S: 0-255
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

        # Saturation limit (normalize to 0-255 range)
        limit_val = self.saturation_limit * 255.0

        # S channel is index 2 in HLS
        # min(S, limit)
        hls[:, :, 2] = np.minimum(hls[:, :, 2], limit_val)

        # HLS to RGB
        result = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

        return result
