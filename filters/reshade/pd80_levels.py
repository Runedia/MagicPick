import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import saturate


class PD80LevelsFilter(BaseFilter):
    """
    PD80_03_Levels.fx 구현

    기본적인 레벨 조정 (Black Point, White Point, Gamma)을 제공합니다.
    RGB 채널별로 입력 레벨과 출력 레벨을 조정할 수 있습니다.
    """

    def __init__(self):
        super().__init__("PD80Levels", "PD80 레벨 조정")
        self.black_in = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.white_in = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.gamma = 1.0
        self.black_out = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.white_out = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        레벨 조정 적용
        """
        # 파라미터 업데이트
        if "black_in" in params:
            self.black_in = np.array(params["black_in"], dtype=np.float32)
        if "white_in" in params:
            self.white_in = np.array(params["white_in"], dtype=np.float32)
        if "gamma" in params:
            self.gamma = float(params["gamma"])
        if "black_out" in params:
            self.black_out = np.array(params["black_out"], dtype=np.float32)
        if "white_out" in params:
            self.white_out = np.array(params["white_out"], dtype=np.float32)

        # 이미지 정규화 (0.0 ~ 1.0)
        img_float = image.astype(np.float32) / 255.0

        # 레벨 계산
        # ret = saturate( color.xyz - blackin.xyz ) / max( whitein.xyz - blackin.xyz, 0.000001f );
        # ret.xyz = pow( ret.xyz, gamma );
        # ret.xyz = ret.xyz * saturate( outwhite.xyz - outblack.xyz ) + outblack.xyz;

        # 입력 레벨 조정
        denom = np.maximum(self.white_in - self.black_in, 0.000001)
        ret = saturate(img_float - self.black_in) / denom

        # 감마 보정
        # 0의 거듭제곱 방지 등을 위해 안전하게 처리하면 좋음, 하지만 여기서는 HLSL 로직을 따름
        ret = np.power(np.maximum(ret, 0.0), self.gamma)

        # 출력 레벨 조정
        out_range = saturate(self.white_out - self.black_out)
        ret = ret * out_range + self.black_out

        return (np.clip(ret, 0.0, 1.0) * 255).astype(np.uint8)
