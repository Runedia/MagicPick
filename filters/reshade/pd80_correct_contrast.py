import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import lerp, saturate


class PD80_CorrectContrast(BaseFilter):
    """
    PD80_01A_RT_Correct_Contrast.fx implementation
    Author: prod80
    Python implementation for Gemini Image Editor
    """

    def __init__(self):
        super().__init__("PD80CorrectContrast", "대비 보정")

        # Default parameters
        self.rt_enable_whitepoint_correction = False
        self.rt_wp_str = 1.0
        self.rt_enable_blackpoint_correction = True
        self.rt_bp_str = 1.0

        # State for temporal smoothing (not fully utilized in single-image mode but kept for structure)
        self.prev_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.prev_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        if params:
            for k, v in params.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def apply(self, image, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        img_float = image.astype(np.float32) / 255.0

        # Calculate Min/Max of the current image
        # The shader does this by downscaling, but we can do it directly.
        # Channel-wise min/max
        current_min = np.min(img_float, axis=(0, 1))
        current_max = np.max(img_float, axis=(0, 1))

        # In a video feed, we would interpolate current_min/max with prev_min/max here.
        # For single image, we just use current values.

        # Logic from PS_CorrectContrast

        # float adjBlack = min( min( minValue.x, minValue.y ), minValue.z );
        adjBlack = np.min(current_min)

        # float adjWhite = max( max( maxValue.x, maxValue.y ), maxValue.z );
        adjWhite = np.max(current_max)

        # Set min value
        # adjBlack = lerp( 0.0f, adjBlack, rt_bp_str );
        adjBlack = lerp(0.0, adjBlack, self.rt_bp_str)

        # adjBlack = rt_enable_blackpoint_correction ? adjBlack : 0.0f;
        if not self.rt_enable_blackpoint_correction:
            adjBlack = 0.0

        # Set max value
        # adjWhite = lerp( 1.0f, adjWhite, rt_wp_str );
        adjWhite = lerp(1.0, adjWhite, self.rt_wp_str)

        # adjWhite = rt_enable_whitepoint_correction ? adjWhite : 1.0f;
        if not self.rt_enable_whitepoint_correction:
            adjWhite = 1.0

        # Main color correction
        # color.xyz = saturate( color.xyz - adjBlack ) / saturate( adjWhite - adjBlack );

        denominator = saturate(adjWhite - adjBlack)
        # Avoid division by zero
        if denominator < 1e-6:
            denominator = 1e-6

        result = saturate(img_float - adjBlack) / denominator

        result = saturate(result)

        return (result * 255).astype(np.uint8)
