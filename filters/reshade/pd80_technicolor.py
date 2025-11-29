import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import lerp, saturate


class PD80_Technicolor(BaseFilter):
    """
    PD80_04_Technicolor.fx implementation
    Author: prod80
    Python implementation for Gemini Image Editor
    """

    def __init__(self):
        super().__init__("PD80Technicolor", "PD80 테크니컬러")

        # Default parameters
        self.Red2strip = np.array([1.0, 0.098, 0.0], dtype=np.float32)
        self.Cyan2strip = np.array([0.0, 0.988, 1.0], dtype=np.float32)
        self.colorKey = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.Saturation2 = 1.5
        self.enable3strip = False
        self.ColorStrength = np.array([0.2, 0.2, 0.2], dtype=np.float32)
        self.Brightness = 1.0
        self.Saturation = 1.0
        self.Strength = 1.0

        # FIXME: 여기 필요한지?
        if params:
            for k, v in params.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def _quaternion_to_matrix(self, quat):
        x, y, z, w = quat

        c_x = y * z
        c_y = z * x
        c_z = x * y

        s_x = x * x
        s_y = y * y
        s_z = z * z

        wi_x = w * x
        wi_y = w * y
        wi_z = w * z

        sq_x = s_x + s_y
        sq_y = s_y + s_z
        sq_z = s_z + s_x

        d_x = 0.5 - sq_x
        d_y = 0.5 - sq_y
        d_z = 0.5 - sq_z

        a_x = c_x + wi_x
        a_y = c_y + wi_y
        a_z = c_z + wi_z

        b_x = c_x - wi_x
        b_y = c_y - wi_y
        b_z = c_z - wi_z

        m = np.array(
            [
                [2.0 * d_x, 2.0 * b_z, 2.0 * a_y],
                [2.0 * a_z, 2.0 * d_y, 2.0 * b_x],
                [2.0 * b_y, 2.0 * a_x, 2.0 * d_z],
            ],
            dtype=np.float32,
        )

        return m

    def apply(self, image, **kwargs):
        # Update parameters if provided
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        img_float = image.astype(np.float32) / 255.0

        # OpenCV uses BGR, shader logic is in RGB. Convert to RGB for processing.
        rgb = img_float[:, :, ::-1]
        orig = rgb.copy()

        # Constants
        root3 = 0.57735
        HueAdj = 0.52

        # --- Pre-calculate Rotation Matrix 1 (180 deg) ---
        half_angle_1 = 0.5 * np.pi  # 90 deg
        s1 = np.sin(half_angle_1)
        c1 = np.cos(half_angle_1)
        rot_quat_1 = (root3 * s1, root3 * s1, root3 * s1, c1)
        rot_Mat_1 = self._quaternion_to_matrix(rot_quat_1)

        # --- Pre-calculate Rotation Matrix 2 (HueAdj) ---
        half_angle_2 = 0.5 * (HueAdj * 2.0 * np.pi)  # HueAdj * 360
        s2 = np.sin(half_angle_2)
        c2 = np.cos(half_angle_2)
        rot_quat_2 = (root3 * s2, root3 * s2, root3 * s2, c2)
        rot_Mat_2 = self._quaternion_to_matrix(rot_quat_2)

        # --- Technicolor 2 Strip Logic ---
        negR = 1.0 - rgb[:, :, 0]
        negG = 1.0 - rgb[:, :, 1]

        # newR = 1.0 - negR * Cyan2strip
        newR = 1.0 - (negR[..., np.newaxis] * self.Cyan2strip)

        # newC = 1.0 - negG * Red2strip
        newC = 1.0 - (negG[..., np.newaxis] * self.Red2strip)

        # key.xyz = mul( rot_Mat, key.xyz );
        key_rotated = np.dot(rot_Mat_1, self.colorKey)

        # key.xyz = max( color.yyy, key.xyz ); (color.y is Green)
        color_g = rgb[:, :, 1]
        key_final = np.maximum(color_g[..., np.newaxis], key_rotated)

        # color.xyz = newR * newC * key
        color = newR * newC * key_final

        # Fix hue: color.xyz = mul( rot_Mat, color.xyz );
        color = np.tensordot(color, rot_Mat_2.T, axes=1)

        # Add saturation
        luma = np.dot(color, np.array([0.212656, 0.715158, 0.072186], dtype=np.float32))
        color = lerp(luma[..., np.newaxis], color, self.Saturation2)

        # --- Technicolor 3 Strip Logic (Optional) ---
        if self.enable3strip:
            temp = 1.0 - orig  # orig is RGB

            # target = temp.grg
            target = np.stack([temp[:, :, 1], temp[:, :, 0], temp[:, :, 1]], axis=2)

            # target2 = temp.bbr
            target2 = np.stack([temp[:, :, 2], temp[:, :, 2], temp[:, :, 0]], axis=2)

            # temp2 = orig * target * target2
            temp2 = orig * target * target2

            # temp = temp2 * ColorStrength
            temp_vec = temp2 * self.ColorStrength

            # temp2 *= Brightness
            temp2 *= self.Brightness

            # target = temp.yxy (using temp_vec)
            target_new = np.stack(
                [temp_vec[:, :, 1], temp_vec[:, :, 0], temp_vec[:, :, 1]], axis=2
            )

            # target2 = temp.zzx (using temp_vec)
            target2_new = np.stack(
                [temp_vec[:, :, 2], temp_vec[:, :, 2], temp_vec[:, :, 0]], axis=2
            )

            # temp = orig - target
            temp_new = orig - target_new

            # temp += temp2
            temp_new += temp2

            # temp2 = temp - target2
            temp2_final = temp_new - target2_new

            # color = lerp( orig, temp2, Strength )
            color_3strip = lerp(orig, temp2_final, self.Strength)

            # color = lerp( getLuminance(color), color, Saturation )
            luma_3 = np.dot(
                color_3strip, np.array([0.212656, 0.715158, 0.072186], dtype=np.float32)
            )
            color = lerp(luma_3[..., np.newaxis], color_3strip, self.Saturation)

        # Result is RGB, convert back to BGR
        result = color[:, :, ::-1]
        result = saturate(result)

        return (result * 255).astype(np.uint8)
