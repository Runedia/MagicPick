"""
HSL Shift 필터

8가지 주요 색조(Red, Orange, Yellow, Green, Cyan, Blue, Purple, Magenta)를
개별적으로 조정할 수 있는 HSL 색공간 기반 색상 변환 필터입니다.
"""

import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import pow_safe, saturate


class HSLShiftFilter(BaseFilter):
    """HSL 색공간 시프트 필터"""

    def __init__(self):
        super().__init__("HSLShift", "HSL 색공간 시프트")
        # 8가지 색조의 기본값 (RGB 0.0~1.0)
        self.hue_red = [0.75, 0.25, 0.25]
        self.hue_orange = [0.75, 0.50, 0.25]
        self.hue_yellow = [0.75, 0.75, 0.25]
        self.hue_green = [0.25, 0.75, 0.25]
        self.hue_cyan = [0.25, 0.75, 0.75]
        self.hue_blue = [0.25, 0.25, 0.75]
        self.hue_purple = [0.50, 0.25, 0.75]
        self.hue_magenta = [0.75, 0.25, 0.75]

        self.threshold_base = 0.05
        self.threshold_curve = 1.0

    def rgb_to_hsl(self, rgb):
        """RGB를 HSL로 변환"""
        r, g, b = rgb[:, :, 2], rgb[:, :, 1], rgb[:, :, 0]  # BGR to RGB

        max_c = np.maximum(r, np.maximum(g, b))
        min_c = np.minimum(r, np.minimum(g, b))
        chroma = max_c - min_c

        # Lightness
        l = max_c - 0.5 * chroma

        # Hue 계산
        h = np.zeros_like(r)

        # chroma != 0인 경우만 처리
        mask = chroma != 0

        # max = R
        mask_r = mask & (max_c == r)
        h[mask_r] = ((g[mask_r] - b[mask_r]) / chroma[mask_r]) % 6.0

        # max = G
        mask_g = mask & (max_c == g)
        h[mask_g] = ((b[mask_g] - r[mask_g]) / chroma[mask_g]) + 2.0

        # max = B
        mask_b = mask & (max_c == b)
        h[mask_b] = ((r[mask_b] - g[mask_b]) / chroma[mask_b]) + 4.0

        h = h / 6.0
        h = h % 1.0  # frac

        # Saturation
        s = np.zeros_like(r)
        s[l != 1] = chroma[l != 1] / (1 - np.abs(2 * l[l != 1] - 1))

        return np.stack([h, s, l], axis=2)

    def hue_to_rgb(self, h):
        """Hue 값을 RGB로 변환"""
        r = saturate(np.abs(h * 6.0 - 3.0) - 1.0)
        g = saturate(2.0 - np.abs(h * 6.0 - 2.0))
        b = saturate(2.0 - np.abs(h * 6.0 - 4.0))
        return np.stack([r, g, b], axis=2)

    def hsl_to_rgb(self, hsl):
        """HSL을 RGB로 변환"""
        h = hsl[:, :, 0]
        s = hsl[:, :, 1]
        l = hsl[:, :, 2]

        hue_rgb = self.hue_to_rgb(h)
        rgb = (hue_rgb - 0.5) * (1.0 - np.abs(2.0 * l[:, :, np.newaxis] - 1)) * s[
            :, :, np.newaxis
        ] + l[:, :, np.newaxis]

        return rgb

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """HSL Shift 필터 적용"""
        # 파라미터 업데이트
        for key, attr in [
            ("hue_red", "hue_red"),
            ("hue_orange", "hue_orange"),
            ("hue_yellow", "hue_yellow"),
            ("hue_green", "hue_green"),
            ("hue_cyan", "hue_cyan"),
            ("hue_blue", "hue_blue"),
            ("hue_purple", "hue_purple"),
            ("hue_magenta", "hue_magenta"),
        ]:
            if key in params:
                setattr(self, attr, params[key])

        img_float = image.astype(np.float32) / 255.0

        # RGB to HSL
        hsl = self.rgb_to_hsl(img_float)

        # 8개 노드 정의 (RGB, angle)
        nodes = [
            (np.array(self.hue_red, dtype=np.float32), 0.0),
            (np.array(self.hue_orange, dtype=np.float32), 30.0),
            (np.array(self.hue_yellow, dtype=np.float32), 60.0),
            (np.array(self.hue_green, dtype=np.float32), 120.0),
            (np.array(self.hue_cyan, dtype=np.float32), 180.0),
            (np.array(self.hue_blue, dtype=np.float32), 240.0),
            (np.array(self.hue_purple, dtype=np.float32), 270.0),
            (np.array(self.hue_magenta, dtype=np.float32), 300.0),
            (np.array(self.hue_red, dtype=np.float32), 360.0),  # Red 반복
        ]

        h, w = img_float.shape[:2]
        hue_deg = hsl[:, :, 0] * 360.0

        # 각 픽셀의 base 노드 찾기
        result_rgb = np.zeros_like(img_float)

        for i in range(8):
            node0_rgb, angle0 = nodes[i]
            node1_rgb, angle1 = nodes[i + 1]

            # 현재 구간에 속하는 픽셀 마스크
            mask = (hue_deg >= angle0) & (hue_deg < angle1)

            if not np.any(mask):
                continue

            # 보간 가중치
            weight = (hue_deg[mask] - angle0) / (angle1 - angle0)
            weight = saturate(weight)

            # 노드의 HSL 변환
            H0 = self.rgb_to_hsl(node0_rgb.reshape(1, 1, 3))
            H1 = self.rgb_to_hsl(node1_rgb.reshape(1, 1, 3))

            H0_h, H0_s, H0_l = H0[0, 0, 0], H0[0, 0, 1], H0[0, 0, 2]
            H1_h, H1_s, H1_l = H1[0, 0, 0], H1[0, 0, 1], H1[0, 0, 2]

            # Hue wrap around
            if H1_h < H0_h:
                H1_h += 1.0

            # 보간
            shift_h = (H0_h + weight * (H1_h - H0_h)) % 1.0
            shift_s = H0_s + weight * (H1_s - H0_s)
            shift_l = H0_l + weight * (H1_l - H0_l)

            # Saturation/Lightness 기반 가중치
            s_orig = hsl[:, :, 1][mask]
            l_orig = hsl[:, :, 2][mask]

            w_factor = np.maximum(s_orig, 0.0) * np.maximum(1.0 - l_orig, 0.0)
            shift_l_offset = (
                (shift_l - 0.5)
                * (
                    pow_safe(w_factor, self.threshold_curve)
                    * (1.0 - self.threshold_base)
                    + self.threshold_base
                )
                * 2.0
            )

            # 최종 HSL
            final_h = shift_h
            final_s = s_orig * (shift_s * 2.0)
            final_l = l_orig * (1.0 + shift_l_offset)

            final_hsl = np.stack([final_h, final_s, final_l], axis=1)
            final_hsl = saturate(final_hsl)
            final_hsl = final_hsl.reshape(-1, 1, 3)

            # HSL to RGB
            final_rgb = self.hsl_to_rgb(final_hsl).reshape(-1, 3)
            result_rgb[mask] = final_rgb

        # 마스크되지 않은 픽셀은 원본 유지
        all_mask = np.zeros((h, w), dtype=bool)
        for i in range(8):
            _, angle0 = nodes[i]
            _, angle1 = nodes[i + 1]
            all_mask |= (hue_deg >= angle0) & (hue_deg < angle1)

        result_rgb[~all_mask] = img_float[~all_mask]

        return (saturate(result_rgb) * 255).astype(np.uint8)
