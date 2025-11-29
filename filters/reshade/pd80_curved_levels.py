import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import saturate


class PD80CurvedLevelsFilter(BaseFilter):
    """
    PD80_03_Curved_Levels.fx 구현

    곡선 기반의 레벨 및 대비 조정 필터입니다.
    Toe(어두운 영역)와 Shoulder(밝은 영역)의 위치를 조정하여 대비 곡선을 만듭니다.
    """

    def __init__(self):
        super().__init__("PD80CurvedLevels", "PD80 곡선 레벨 (대비)")
        # Global / Grey (Master) settings
        self.black_in = 0.0  # 0 ~ 255
        self.white_in = 255.0  # 0 ~ 255
        self.black_out = 0.0  # 0 ~ 255
        self.white_out = 255.0  # 0 ~ 255

        # Curve points (x, y) normalized 0.0 ~ 1.0
        self.toe_x = 0.25
        self.toe_y = 0.25
        self.shoulder_x = 0.75
        self.shoulder_y = 0.75

        # RGB specific settings (Optional, defaulting to Disabled behavior)
        self.enable_rgb = False

        # Red
        self.red_black_in = 0.0
        self.red_white_in = 255.0
        self.red_black_out = 0.0
        self.red_white_out = 255.0
        self.red_toe_x = 0.25
        self.red_toe_y = 0.25
        self.red_shoulder_x = 0.75
        self.red_shoulder_y = 0.75

        # Green
        self.green_black_in = 0.0
        self.green_white_in = 255.0
        self.green_black_out = 0.0
        self.green_white_out = 255.0
        self.green_toe_x = 0.25
        self.green_toe_y = 0.25
        self.green_shoulder_x = 0.75
        self.green_shoulder_y = 0.75

        # Blue
        self.blue_black_in = 0.0
        self.blue_white_in = 255.0
        self.blue_black_out = 0.0
        self.blue_white_out = 255.0
        self.blue_toe_x = 0.25
        self.blue_toe_y = 0.25
        self.blue_shoulder_x = 0.75
        self.blue_shoulder_y = 0.75

    def _prepare_tonemap_params(self, p1, p2, p3):
        """
        곡선 파라미터를 계산합니다.
        p1, p2, p3는 각각 (x, y) 튜플 또는 배열입니다.
        """
        p1_x, p1_y = p1
        p2_x, p2_y = p2
        p3_x, p3_y = p3

        # Calculate Slope
        denom = p2_x - p1_x
        if abs(denom) < 1e-5:
            denom = 1e-5 if denom >= 0 else -1e-5

        slope = (p2_y - p1_y) / denom

        # Mid parameters (Linear section)
        mid_x = slope
        mid_y = p1_y - slope * p1_x

        # Toe parameters
        denom_toe = p1_y - slope * p1_x
        if abs(denom_toe) < 1e-5:
            denom_toe = 1e-5 if denom_toe >= 0 else -1e-5

        toe_x = slope * p1_x * p1_x * p1_y * p1_y / (denom_toe * denom_toe)
        toe_y = slope * p1_x * p1_x / denom_toe
        toe_z = p1_y * p1_y / denom_toe

        # Shoulder parameters
        denom_sho = slope * (p2_x - p3_x) - p2_y + p3_y
        if abs(denom_sho) < 1e-5:
            denom_sho = 1e-5 if denom_sho >= 0 else -1e-5

        shoulder_x = (
            slope * (p2_x - p3_x) ** 2 * (p2_y - p3_y) ** 2 / (denom_sho * denom_sho)
        )
        shoulder_y = (slope * p2_x * (p3_x - p2_x) + p3_x * (p2_y - p3_y)) / denom_sho
        shoulder_z = (-p2_y * p2_y + p3_y * (slope * (p2_x - p3_x) + p2_y)) / denom_sho

        return {
            "mToe": (toe_x, toe_y, toe_z),
            "mMid": (mid_x, mid_y),
            "mShoulder": (shoulder_x, shoulder_y, shoulder_z),
            "mBx": (p1_x, p2_x),
        }

    def _tonemap(self, tc, x):
        """
        곡선을 적용합니다. x는 numpy 배열일 수 있습니다.
        """
        # Toe Section
        # toe = - tc.mToe.x / (x + tc.mToe.y) + tc.mToe.z;
        toe = -tc["mToe"][0] / (x + tc["mToe"][1] + 1e-6) + tc["mToe"][2]

        # Mid Section
        # mid = tc.mMid.x * x + tc.mMid.y;
        mid = tc["mMid"][0] * x + tc["mMid"][1]

        # Shoulder Section
        # shoulder = - tc.mShoulder.x / (x + tc.mShoulder.y) + tc.mShoulder.z;
        shoulder = (
            -tc["mShoulder"][0] / (x + tc["mShoulder"][1] + 1e-6) + tc["mShoulder"][2]
        )

        # Combine
        # result = ( x >= tc.mBx.x ) ? mid : toe;
        # result = ( x >= tc.mBx.y ) ? shoulder : result;

        result = np.where(x >= tc["mBx"][0], mid, toe)
        result = np.where(x >= tc["mBx"][1], shoulder, result)

        return result

    def _black_white_in(self, c, b, w):
        # return saturate( c - b )/max( w - b, 0.000001f );
        return saturate(c - b) / max(w - b, 0.000001)

    def _black_white_out(self, c, b, w):
        # return c * saturate( w - b ) + b;
        return c * saturate(w - b) + b

    def _set_boundaries(self, tx, ty, sx, sy):
        if tx > sx:
            tx = sx
        if ty > sy:
            ty = sy
        return tx, ty, sx, sy

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        for key in params:
            if hasattr(self, key):
                setattr(self, key, params[key])

        # Normalize image 0.0 - 1.0
        img_float = image.astype(np.float32) / 255.0

        # --- Master Curve (Applied to all channels) ---

        # Set boundaries for curve points
        tx, ty, sx, sy = self._set_boundaries(
            self.toe_x, self.toe_y, self.shoulder_x, self.shoulder_y
        )

        # Calculate curve parameters
        tc_grey = self._prepare_tonemap_params((tx, ty), (sx, sy), (1.0, 1.0))

        # Apply Black/White In
        b_in = self.black_in / 255.0
        w_in = self.white_in / 255.0
        img_float = self._black_white_in(img_float, b_in, w_in)

        # Apply Tonemap Curve
        img_float = self._tonemap(tc_grey, img_float)

        # Apply Black/White Out
        b_out = self.black_out / 255.0
        w_out = self.white_out / 255.0
        img_float = self._black_white_out(img_float, b_out, w_out)

        # --- RGB Curves (Optional) ---
        if self.enable_rgb:
            # Red
            r_tx, r_ty, r_sx, r_sy = self._set_boundaries(
                self.red_toe_x, self.red_toe_y, self.red_shoulder_x, self.red_shoulder_y
            )
            tc_red = self._prepare_tonemap_params(
                (r_tx, r_ty), (r_sx, r_sy), (1.0, 1.0)
            )

            r_in = self._black_white_in(
                img_float[:, :, 0], self.red_black_in / 255.0, self.red_white_in / 255.0
            )
            r_tm = self._tonemap(tc_red, r_in)
            img_float[:, :, 0] = self._black_white_out(
                r_tm, self.red_black_out / 255.0, self.red_white_out / 255.0
            )

            # Green
            g_tx, g_ty, g_sx, g_sy = self._set_boundaries(
                self.green_toe_x,
                self.green_toe_y,
                self.green_shoulder_x,
                self.green_shoulder_y,
            )
            tc_green = self._prepare_tonemap_params(
                (g_tx, g_ty), (g_sx, g_sy), (1.0, 1.0)
            )

            g_in = self._black_white_in(
                img_float[:, :, 1],
                self.green_black_in / 255.0,
                self.green_white_in / 255.0,
            )
            g_tm = self._tonemap(tc_green, g_in)
            img_float[:, :, 1] = self._black_white_out(
                g_tm, self.green_black_out / 255.0, self.green_white_out / 255.0
            )

            # Blue
            b_tx, b_ty, b_sx, b_sy = self._set_boundaries(
                self.blue_toe_x,
                self.blue_toe_y,
                self.blue_shoulder_x,
                self.blue_shoulder_y,
            )
            tc_blue = self._prepare_tonemap_params(
                (b_tx, b_ty), (b_sx, b_sy), (1.0, 1.0)
            )

            b_in = self._black_white_in(
                img_float[:, :, 2],
                self.blue_black_in / 255.0,
                self.blue_white_in / 255.0,
            )
            b_tm = self._tonemap(tc_blue, b_in)
            img_float[:, :, 2] = self._black_white_out(
                b_tm, self.blue_black_out / 255.0, self.blue_white_out / 255.0
            )

        return (np.clip(img_float, 0.0, 1.0) * 255).astype(np.uint8)
