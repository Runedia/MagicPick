"""
Film Grain 2 필터

Perlin 노이즈 기반의 필름 그레인 효과입니다.
루미넌스에 따라 그레인 강도가 조정됩니다.
"""

import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import lerp, saturate, smoothstep


class FilmGrain2Filter(BaseFilter):
    """필름 그레인 v2 필터 (Perlin 노이즈 기반)"""

    def __init__(self):
        super().__init__("FilmGrain2", "필름 그레인 v2")
        self.grain_amount = 0.05  # 0.0 ~ 0.2
        self.color_amount = 0.6  # 0.0 ~ 1.0
        self.luma_amount = 1.0  # 0.0 ~ 1.0
        self.grain_size = 1.6  # 1.5 ~ 2.5

        # 타이머 시뮬레이션 (랜덤 시드)
        self.timer = np.random.rand() * 1000.0

    def _rnm(self, tc):
        """랜덤 텍스처 생성 (간단한 해시 함수)"""
        noise = np.sin(tc[:, :, 0] * 12.9898 + tc[:, :, 1] * 78.233) * 43758.5453

        noiseR = (noise % 1.0) * 2.0 - 1.0
        noiseG = ((noise * 1.2154) % 1.0) * 2.0 - 1.0
        noiseB = ((noise * 1.3453) % 1.0) * 2.0 - 1.0
        noiseA = ((noise * 1.3647) % 1.0) * 2.0 - 1.0

        return np.stack([noiseR, noiseG, noiseB, noiseA], axis=2)

    def _pnoise3D(self, p):
        """간단한 3D Perlin 노이즈 (근사)"""
        # 실제 Perlin 노이즈는 복잡하므로 간단한 버전 사용
        perm_tex_unit = 1.0 / 256.0
        perm_tex_unit_half = 0.5 / 256.0

        pi = perm_tex_unit * np.floor(p) + perm_tex_unit_half
        pf = p - np.floor(p)

        # 2D 좌표로 단순화 (z=0 평면)
        pi_2d = pi[:, :, :2]
        perm00 = self._rnm(pi_2d)[:, :, 3]

        # 그래디언트 근사
        perm_coords = np.stack([perm00, pi[:, :, 2]], axis=2)
        grad000 = self._rnm(perm_coords)[:, :, :3] * 4.0 - 1.0

        # 내적
        n000 = np.sum(grad000 * pf, axis=2)

        # 단순화된 노이즈 (fade 없이)
        return n000

    def _coord_rot(self, texcoord, angle, aspect_ratio):
        """좌표 회전"""
        tc_x = texcoord[:, :, 0] * 2.0 - 1.0
        tc_y = texcoord[:, :, 1] * 2.0 - 1.0

        rot_x = (tc_x * aspect_ratio * np.cos(angle)) - (tc_y * np.sin(angle))
        rot_y = (tc_y * np.cos(angle)) + (tc_x * aspect_ratio * np.sin(angle))

        rot_x = (rot_x / aspect_ratio) * 0.5 + 0.5
        rot_y = rot_y * 0.5 + 0.5

        return np.stack([rot_x, rot_y], axis=2)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """Film Grain 2 필터 적용"""
        # 파라미터 업데이트
        self.grain_amount = params.get("grain_amount", self.grain_amount)
        self.color_amount = params.get("color_amount", self.color_amount)
        self.luma_amount = params.get("luma_amount", self.luma_amount)
        self.grain_size = params.get("grain_size", self.grain_size)

        # 새 랜덤 시드 (애니메이션 효과)
        if params.get("animate", True):
            self.timer = np.random.rand() * 1000.0

        img_float = image.astype(np.float32) / 255.0
        h, w = img_float.shape[:2]
        aspect_ratio = w / h

        # 텍스처 좌표 생성
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        texcoord = np.stack([x_coords / w, y_coords / h], axis=2)

        # 회전 오프셋
        rot_offset = np.array([1.425, 3.892, 5.835])

        # R 채널 노이즈
        rot_coords_r = self._coord_rot(
            texcoord, self.timer + rot_offset[0], aspect_ratio
        )
        noise_coords = np.stack(
            [
                rot_coords_r[:, :, 0] * w / self.grain_size,
                rot_coords_r[:, :, 1] * h / self.grain_size,
                np.zeros((h, w)),
            ],
            axis=2,
        )
        noise_r = self._pnoise3D(noise_coords)

        noise = np.stack([noise_r, noise_r, noise_r], axis=2)

        # 컬러 노이즈 추가
        if self.color_amount > 0:
            rot_coords_g = self._coord_rot(
                texcoord, self.timer + rot_offset[1], aspect_ratio
            )
            noise_coords_g = np.stack(
                [
                    rot_coords_g[:, :, 0] * w / self.grain_size,
                    rot_coords_g[:, :, 1] * h / self.grain_size,
                    np.ones((h, w)),
                ],
                axis=2,
            )
            noise_g = self._pnoise3D(noise_coords_g)
            noise[:, :, 1] = lerp(noise_r, noise_g, self.color_amount)

            rot_coords_b = self._coord_rot(
                texcoord, self.timer + rot_offset[2], aspect_ratio
            )
            noise_coords_b = np.stack(
                [
                    rot_coords_b[:, :, 0] * w / self.grain_size,
                    rot_coords_b[:, :, 1] * h / self.grain_size,
                    np.ones((h, w)) * 2.0,
                ],
                axis=2,
            )
            noise_b = self._pnoise3D(noise_coords_b)
            noise[:, :, 0] = lerp(noise_r, noise_b, self.color_amount)  # BGR 순서

        # 루미넌스 계산
        lum_coeff = np.array([0.114, 0.587, 0.299])  # BGR
        luminance = lerp(0.0, np.dot(img_float, lum_coeff), self.luma_amount)
        lum = smoothstep(0.2, 0.0, luminance)
        lum = lum + luminance

        # 루미넌스에 따라 노이즈 조정
        lum_factor = np.power(lum, 4.0)
        noise = lerp(noise, 0.0, lum_factor[:, :, np.newaxis])

        # 노이즈 추가
        result = img_float + noise * self.grain_amount

        return (saturate(result) * 255).astype(np.uint8)
