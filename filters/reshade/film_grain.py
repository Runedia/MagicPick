"""
Film Grain - 필름 그레인 노이즈 효과

루미넌스 기반 가중치 적용 그레인
"""

import numpy as np

from filters.base_filter import BaseFilter


class FilmGrainFilter(BaseFilter):
    """Film Grain 필터 (필름 그레인 효과)"""

    def __init__(self):
        super().__init__("FilmGrain", "필름 그레인")
        self.intensity = 0.5
        self.variance = 0.4
        self.mean = 0.5
        self.signal_to_noise_ratio = 6.0

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        self.intensity = params.get("Intensity", self.intensity)
        self.variance = params.get("Variance", self.variance)
        self.mean = params.get("Mean", self.mean)
        self.signal_to_noise_ratio = params.get(
            "SignalToNoiseRatio", self.signal_to_noise_ratio
        )

        img_float = image.astype(np.float32) / 255.0

        grain_amount = self.intensity * 0.05

        lum_coeff = np.array([0.114, 0.587, 0.299])
        luminance = np.dot(img_float, lum_coeff)

        snr_factor = 1.0 / max(1.0, self.signal_to_noise_ratio)

        noise = np.random.normal(
            self.mean - 0.5, self.variance * snr_factor, image.shape[:2]
        )

        luma_weight = 1.0 - np.power(luminance, 2.0)
        weighted_noise = noise * luma_weight

        noise_3d = np.stack([weighted_noise] * 3, axis=2)

        result = img_float + noise_3d * grain_amount

        return (np.clip(result, 0, 1) * 255).astype(np.uint8)
