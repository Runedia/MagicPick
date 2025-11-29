import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import saturate


class PD80_ColorGamut(BaseFilter):
    """
    PD80_01_Color_Gamut.fx implementation
    Author: prod80
    Python implementation for Gemini Image Editor
    """

    def __init__(self):
        super().__init__("PD80ColorGamut", "색역 조정")

        # Default parameters
        self.colorgamut = 0  # 0: No Change

        # --- Constants (Transposed for NumPy's dot convention: v @ M) ---
        # Or stick to HLSL mul(M, v) -> M @ v.
        # NumPy: if v is (N, 3), v @ M.T is equivalent to (M @ v.T).T

        self.sRGB_To_XYZ = np.array(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ],
            dtype=np.float32,
        )

        # Gamut Matrices
        self.gamut_matrices = [
            # 0: XYZ_To_sRGB
            np.array(
                [
                    [3.2404542, -1.5371385, -0.4985314],
                    [-0.9692660, 1.8760108, 0.0415560],
                    [0.0556434, -0.2040259, 1.0572252],
                ],
                dtype=np.float32,
            ),
            # 1: XYZ_To_AdobeRGB
            np.array(
                [
                    [2.0413690, -0.5649464, -0.3446944],
                    [-0.9692660, 1.8760108, 0.0415560],
                    [0.0134474, -0.1183897, 1.0154096],
                ],
                dtype=np.float32,
            ),
            # 2: XYZ_To_AppleRGB
            np.array(
                [
                    [2.9515373, -1.2894116, -0.4738445],
                    [-1.0851093, 1.9908566, 0.0372026],
                    [0.0854934, -0.2694964, 1.0912975],
                ],
                dtype=np.float32,
            ),
            # 3: XYZ_To_BestRGB
            np.array(
                [
                    [1.7552599, -0.4836786, -0.2530000],
                    [-0.5441336, 1.5068789, 0.0215528],
                    [0.0063467, -0.0175761, 1.2256959],
                ],
                dtype=np.float32,
            ),
            # 4: XYZ_To_BetaRGB
            np.array(
                [
                    [1.6832270, -0.4282363, -0.2360185],
                    [-0.7710229, 1.7065571, 0.0446900],
                    [0.0400013, -0.0885376, 1.2723640],
                ],
                dtype=np.float32,
            ),
            # 5: XYZ_To_BruceRGB
            np.array(
                [
                    [2.7454669, -1.1358136, -0.4350269],
                    [-0.9692660, 1.8760108, 0.0415560],
                    [0.0112723, -0.1139754, 1.0132541],
                ],
                dtype=np.float32,
            ),
            # 6: XYZ_To_CIERGB
            np.array(
                [
                    [2.3706743, -0.9000405, -0.4706338],
                    [-0.5138850, 1.4253036, 0.0885814],
                    [0.0052982, -0.0146949, 1.0093968],
                ],
                dtype=np.float32,
            ),
            # 7: XYZ_To_ColorMatch
            np.array(
                [
                    [2.6422874, -1.2234270, -0.3930143],
                    [-1.1119763, 2.0590183, 0.0159614],
                    [0.0821699, -0.2807254, 1.4559877],
                ],
                dtype=np.float32,
            ),
            # 8: XYZ_To_Don
            np.array(
                [
                    [1.7603902, -0.4881198, -0.2536126],
                    [-0.7126288, 1.6527432, 0.0416715],
                    [0.0078207, -0.0347411, 1.2447743],
                ],
                dtype=np.float32,
            ),
            # 9: XYZ_To_ECI
            np.array(
                [
                    [1.7827618, -0.4969847, -0.2690101],
                    [-0.9593623, 1.9477962, -0.0275807],
                    [0.0859317, -0.1744674, 1.3228273],
                ],
                dtype=np.float32,
            ),
            # 10: XYZ_To_EktaSpacePS5
            np.array(
                [
                    [2.0043819, -0.7304844, -0.2450052],
                    [-0.7110285, 1.6202126, 0.0792227],
                    [0.0381263, -0.0868780, 1.2725438],
                ],
                dtype=np.float32,
            ),
            # 11: XYZ_To_NTSC
            np.array(
                [
                    [1.9099961, -0.5324542, -0.2882091],
                    [-0.9846663, 1.9991710, -0.0283082],
                    [0.0583056, -0.1183781, 0.8975535],
                ],
                dtype=np.float32,
            ),
            # 12: XYZ_To_PALSECAM
            np.array(
                [
                    [3.0628971, -1.3931791, -0.4757517],
                    [-0.9692660, 1.8760108, 0.0415560],
                    [0.0678775, -0.2288548, 1.0693490],
                ],
                dtype=np.float32,
            ),
            # 13: XYZ_To_ProPhoto
            np.array(
                [
                    [1.3459433, -0.2556075, -0.0511118],
                    [-0.5445989, 1.5081673, 0.0205351],
                    [0.0000000, 0.0000000, 1.2118128],
                ],
                dtype=np.float32,
            ),
            # 14: XYZ_To_SMPTEC
            np.array(
                [
                    [3.5053960, -1.7394894, -0.5439640],
                    [-1.0690722, 1.9778245, 0.0351722],
                    [0.0563200, -0.1970226, 1.0502026],
                ],
                dtype=np.float32,
            ),
            # 15: XYZ_To_WideGamutRGB
            np.array(
                [
                    [1.4628067, -0.1840623, -0.2743606],
                    [-0.5217933, 1.4472381, 0.0677227],
                    [0.0349342, -0.0968930, 1.2884099],
                ],
                dtype=np.float32,
            ),
        ]

        # White Point Matrices
        self.D65_To_D50 = np.array(
            [
                [1.0478112, 0.0228866, -0.0501270],
                [0.0295424, 0.9904844, -0.0170491],
                [-0.0092345, 0.0150436, 0.7521316],
            ],
            dtype=np.float32,
        )

        self.D65_To_E = np.array(
            [
                [1.0502616, 0.0270757, -0.0232523],
                [0.0390650, 0.9729502, -0.0092579],
                [-0.0024047, 0.0026446, 0.918087],
            ],
            dtype=np.float32,
        )

        self.D65_To_C = np.array(
            [
                [1.0097785, 0.0070419, 0.0127971],
                [0.0123113, 0.9847094, 0.0032962],
                [0.0038284, -0.0072331, 1.0891639],
            ],
            dtype=np.float32,
        )

        if params:
            for k, v in params.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def _linear_to_srgb(self, color):
        # color: (H, W, 3)

        # Branch 1: x = color * 12.92
        x = color * 12.92

        # Branch 2: y = 1.055 * pow(saturate(color), 1.0/2.4) - 0.055
        y = 1.055 * np.power(saturate(color), 1.0 / 2.4) - 0.055

        # Condition: color < 0.0031308
        mask = color < 0.0031308

        return np.where(mask, x, y)

    def _srgb_to_linear(self, color):
        # color: (H, W, 3)

        # Branch 1: x = color / 12.92
        x = color / 12.92

        # Branch 2: y = pow(max((color + 0.055) / 1.055, 0.0), 2.4)
        y = np.power(np.maximum((color + 0.055) / 1.055, 0.0), 2.4)

        # Condition: color <= 0.04045
        mask = color <= 0.04045

        return np.where(mask, x, y)

    def apply(self, image, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        if self.colorgamut < 0 or self.colorgamut >= len(self.gamut_matrices):
            return image

        img_float = image.astype(np.float32) / 255.0

        # OpenCV BGR to RGB for processing
        rgb = img_float[:, :, ::-1]

        # SRGB To Linear
        linear = self._srgb_to_linear(rgb)

        # Linear To XYZ
        # mul(sRGB_To_XYZ, color) -> sRGB_To_XYZ @ color
        # NumPy: color @ sRGB_To_XYZ.T
        xyz = np.tensordot(linear, self.sRGB_To_XYZ.T, axes=1)

        # Select Gamut Matrix
        gamut_matrix = self.gamut_matrices[self.colorgamut]

        # Select White Point Matrix
        refwhitemat = np.eye(3, dtype=np.float32)

        if self.colorgamut in [3, 4, 7, 8, 9, 10, 13, 15]:
            refwhitemat = self.D65_To_D50
        elif self.colorgamut == 6:
            refwhitemat = self.D65_To_E
        elif self.colorgamut == 11:
            refwhitemat = self.D65_To_C

        # Transform
        # color = mul( gamut, mul( refwhitemat, color ));
        # 1. mul(refwhitemat, color) -> refwhitemat @ color
        # 2. mul(gamut, result) -> gamut @ result

        # In NumPy with (H,W,3) vectors:
        # v1 = v @ refwhitemat.T
        # v2 = v1 @ gamut.T

        temp = np.tensordot(xyz, refwhitemat.T, axes=1)
        transformed = np.tensordot(temp, gamut_matrix.T, axes=1)

        # Linear To sRGB
        result_rgb = self._linear_to_srgb(transformed)

        # RGB to BGR
        result_bgr = result_rgb[:, :, ::-1]

        # Clip and Convert
        result_bgr = saturate(result_bgr)
        return (result_bgr * 255).astype(np.uint8)
