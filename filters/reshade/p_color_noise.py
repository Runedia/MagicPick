"""
pColorNoise.fx 간소화 구현

Gaussian Chroma Noise (Color Noise)
Original: Gimle Larpes (potatoFX)
"""

import numpy as np

from filters.base_filter import BaseFilter


class PColorNoiseFilter(BaseFilter):
    """
    pColorNoise - 컬러 노이즈

    Features:
    - Gaussian chroma noise
    - 디지털 카메라 앰프 노이즈 시뮬레이션
    - 밝기 적응형 노이즈
    """

    def __init__(self):
        super().__init__("pColorNoise", "컬러 노이즈")

        # Parameters
        self.strength = 0.12  # 0.0 ~ 1.0

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Apply pColorNoise filter

        Args:
            image: RGB image (uint8, 0-255)
            **params: Filter parameters
                - strength: Noise strength (0.0 ~ 1.0, default 0.12)

        Returns:
            Color noised image (uint8, 0-255)
        """
        # Update parameters
        self.strength = params.get("strength", self.strength)

        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        h, w = img_float.shape[:2]

        # Calculate luminance
        luminance = np.sum(
            img_float * np.array([0.2126, 0.7152, 0.0722]), axis=2
        )

        # Generate Gaussian noise for each channel
        # Box-Muller transform to generate Gaussian noise from uniform
        np.random.seed(None)  # Ensure randomness
        
        # Generate uniform random numbers
        noise1 = np.random.rand(h, w)
        noise2 = np.random.rand(h, w)
        noise3 = np.random.rand(h, w)

        # Box-Muller transform
        epsilon = 1e-10
        r = np.sqrt(-2.0 * np.log(noise1 + epsilon))
        theta1 = 2.0 * np.pi * noise2
        theta2 = 2.0 * np.pi * noise3

        # Sensor sensitivity to color channels
        # (from camera spec sheet simulation)
        gauss_noise_r = r * np.cos(theta1) * 1.33
        gauss_noise_g = r * np.sin(theta1) * 1.25
        gauss_noise_b = r * np.cos(theta2) * 2.0

        # Combine noise
        gauss_noise = np.stack(
            [gauss_noise_r, gauss_noise_g, gauss_noise_b], axis=2
        )

        # Luma-adaptive weight
        # Higher noise in darker areas, simulating wider dynamic range
        invnorm_factor = 100.0  # Simplified from OKLAB INVNORM_FACTOR
        noise_curve = max(invnorm_factor * 0.025, 1.0)
        
        weight = (
            (self.strength * self.strength)
            * noise_curve
            / (luminance * (1.0 + 1.0 / invnorm_factor) + 2.0)
        )
        weight = np.expand_dims(weight, axis=2)

        # Apply noise
        result = img_float + gauss_noise * weight

        # Clamp and convert
        result = np.clip(result, 0, 1)
        return (result * 255).astype(np.uint8)
