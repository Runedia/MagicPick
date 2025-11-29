"""
ASCII 필터

이미지를 ASCII 문자로 변환하여 렌더링하는 필터입니다.
비트맵 폰트와 그레이스케일 매핑을 사용하여 이미지를 재구성합니다.
"""

import cv2
import numpy as np

from filters.base_filter import BaseFilter

from .hlsl_helpers import rgb_to_luma_fast


class ASCIIFilter(BaseFilter):
    """ASCII Art 필터 구현"""

    def __init__(self):
        super().__init__()
        self.spacing = 1  # 문자 간격 (0~5)
        self.font_size_mode = 1  # 1: 5x5, 2: 3x5
        self.font_color_mode = 1  # 0: 단색, 1: 그레이스케일 컬러, 2: 풀 컬러
        self.font_color = [1.0, 1.0, 1.0]  # 폰트 색상
        self.background_color = [0.0, 0.0, 0.0]  # 배경 색상
        self.swap_colors = False  # 색상 반전
        self.invert_brightness = False  # 밝기 반전
        self.dithering = True  # 디더링 사용 여부
        self.dithering_intensity = 2.0  # 디더링 강도

        # 폰트 패턴 캐시
        self._font_patterns_cache = {}

    def _get_bitfield_5x5(self, gray_level, quant):
        """5x5 폰트의 비트필드 값을 반환"""
        # 17 characters
        # .:^"~cvo*wSO8Q0#

        n12 = 4194304.0 if gray_level < (2.0 * quant) else 131200.0
        n34 = 324.0 if gray_level < (4.0 * quant) else 330.0
        n56 = 283712.0 if gray_level < (6.0 * quant) else 12650880.0
        n78 = 4532768.0 if gray_level < (8.0 * quant) else 13191552.0
        n910 = 10648704.0 if gray_level < (10.0 * quant) else 11195936.0
        n1112 = 15218734.0 if gray_level < (12.0 * quant) else 15255086.0
        n1314 = 15252014.0 if gray_level < (14.0 * quant) else 32294446.0
        n1516 = 15324974.0 if gray_level < (16.0 * quant) else 11512810.0

        n1234 = n12 if gray_level < (3.0 * quant) else n34
        n5678 = n56 if gray_level < (7.0 * quant) else n78
        n9101112 = n910 if gray_level < (11.0 * quant) else n1112
        n13141516 = n1314 if gray_level < (15.0 * quant) else n1516

        n12345678 = n1234 if gray_level < (5.0 * quant) else n5678
        n910111213141516 = n9101112 if gray_level < (13.0 * quant) else n13141516

        n = n12345678 if gray_level < (9.0 * quant) else n910111213141516

        # 가장 어두운 단계(0)는 공백 처리 (n=0)
        if gray_level <= (1.0 * quant):
            n = 0.0

        return n

    def _get_bitfield_3x5(self, gray_level, quant):
        """3x5 폰트의 비트필드 값을 반환"""
        # 14 characters
        # .:;s*oSOXH0

        n12 = 4096.0 if gray_level < (2.0 * quant) else 1040.0
        n34 = 5136.0 if gray_level < (4.0 * quant) else 5200.0
        n56 = 2728.0 if gray_level < (6.0 * quant) else 11088.0
        n78 = 14478.0 if gray_level < (8.0 * quant) else 11114.0
        n910 = 23213.0 if gray_level < (10.0 * quant) else 15211.0
        n1112 = 23533.0 if gray_level < (12.0 * quant) else 31599.0
        n13 = 31727.0

        n1234 = n12 if gray_level < (3.0 * quant) else n34
        n5678 = n56 if gray_level < (7.0 * quant) else n78
        n9101112 = n910 if gray_level < (11.0 * quant) else n1112

        n12345678 = n1234 if gray_level < (5.0 * quant) else n5678
        n910111213 = n9101112 if gray_level < (13.0 * quant) else n13

        n = n12345678 if gray_level < (9.0 * quant) else n910111213

        if gray_level <= (1.0 * quant):
            n = 0.0

        return n

    def _generate_font_patterns(self, font_w, font_h, num_chars):
        """
        비트필드로부터 문자 패턴 이미지를 미리 생성
        Returns:
            patterns: (num_chars, block_h, block_w) 형태의 NumPy 배열
        """
        cache_key = (font_w, font_h, self.spacing)
        if cache_key in self._font_patterns_cache:
            return self._font_patterns_cache[cache_key]

        block_w = font_w + self.spacing
        block_h = font_h + self.spacing
        quant = 1.0 / (num_chars - 1.0)

        patterns = np.zeros((num_chars, block_h, block_w), dtype=np.float32)

        for i in range(num_chars):
            # i번째 레벨에 해당하는 비트필드 값 가져오기
            # gray level을 i * quant + epsilon 으로 설정하여 해당 문자의 n값 획득
            gray_val = i * quant + 0.0001

            if font_w == 5:
                n = self._get_bitfield_5x5(gray_val, quant)
            else:
                n = self._get_bitfield_3x5(gray_val, quant)

            # 비트필드 디코딩하여 패턴 생성
            for py in range(block_h):
                for px in range(block_w):
                    # 여백 처리
                    if px >= font_w or py >= font_h:
                        patterns[i, py, px] = 0.0
                        continue

                    # 픽셀 위치를 비트 인덱스로 변환
                    x = font_w * py + px

                    # 비트 확인
                    # character = ( frac( abs( n*exp2(-x-1.0))) >= 0.5) ? lit : signbit;
                    val = abs(n * (2.0 ** (-x - 1.0)))
                    frac_val = val - int(val)

                    is_set = frac_val >= 0.5

                    # signbit 처리 (n < 0 이고 첫 픽셀(x>23.5)인 경우)
                    # 5x5에서 x는 0~24. x>23.5는 24(마지막 픽셀, (4,4))를 의미하는 듯하나
                    # HLSL 코드 로직상 x는 0부터 시작.
                    # signbit = (x > 23.5) ? signbit : 0.0
                    # 여기서는 간단히 n < 0 처리는 무시하거나(대부분 양수),
                    # Python의 비트 연산으로 정확히 처리.
                    # 셰이더의 float 비트 연산은 복잡하므로, 여기서는
                    # n을 정수로 변환하여 비트 연산하는 것이 더 안전할 수 있음.
                    # 하지만 n이 매우 큰 수(float)라 정밀도 문제 주의.
                    # 셰이더 로직을 그대로 따름.

                    pixel_val = 1.0 if is_set else 0.0

                    # n이 음수인 경우 처리는 생략 (주어진 문자셋에서는 음수 사용 안함)
                    # 단, -17895696.0 같은 음수 값들이 주석에 있음.
                    # 현재 구현된 _get_bitfield에서는 모두 양수만 반환하도록 되어있음(초기 버전).
                    # 만약 음수 문자(특수 문자)를 지원하려면 추가 구현 필요.

                    patterns[i, py, px] = pixel_val

        self._font_patterns_cache[cache_key] = patterns
        return patterns

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        ASCII 필터 적용
        """
        # 파라미터 업데이트
        self.spacing = int(params.get("Ascii_spacing", self.spacing))
        self.font_size_mode = int(params.get("Ascii_font", self.font_size_mode))
        self.font_color_mode = int(
            params.get("Ascii_font_color_mode", self.font_color_mode)
        )
        self.swap_colors = bool(params.get("Ascii_swap_colors", self.swap_colors))
        self.invert_brightness = bool(
            params.get("Ascii_invert_brightness", self.invert_brightness)
        )
        self.dithering = bool(params.get("Ascii_dithering", self.dithering))

        if "Ascii_font_color" in params:
            self.font_color = params["Ascii_font_color"]
        if "Ascii_background_color" in params:
            self.background_color = params["Ascii_background_color"]

        # 설정에 따른 폰트 정보
        if self.font_size_mode == 1:  # 5x5
            font_w, font_h = 5, 5
            num_chars = 17
        else:  # 3x5
            font_w, font_h = 3, 5
            num_chars = 14

        block_w = font_w + self.spacing
        block_h = font_h + self.spacing

        h, w = image.shape[:2]

        # 그리드 크기 계산
        grid_w = w // block_w
        grid_h = h // block_h

        if grid_w == 0 or grid_h == 0:
            return image

        # 1. 이미지 다운샘플링 (그리드 단위로 평균 색상 구하기)
        # 리사이즈를 통해 간단히 평균 색상을 얻음 (INTER_AREA가 평균과 유사)
        small_img = cv2.resize(image, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
        small_img_float = small_img.astype(np.float32) / 255.0

        # 2. 밝기(Luma) 계산
        luma = rgb_to_luma_fast(small_img_float)  # (grid_h, grid_w, 1)
        luma = luma[:, :, 0]  # (grid_h, grid_w)

        if self.invert_brightness:
            luma = 1.0 - luma

        # 3. 디더링 적용
        if self.dithering:
            quant = 1.0 / (num_chars - 1.0)

            # 노이즈 생성 (좌표 기반 의사 난수)
            y_coords, x_coords = np.mgrid[0:grid_h, 0:grid_w]
            # float seed = dot(cursor_position, float2(12.9898,78.233));
            # cursor_position은 텍스처 좌표가 아니라 픽셀 좌표(그리드 인덱스)에 해당
            seed = x_coords * 12.9898 + y_coords * 78.233
            sine = np.sin(seed)
            noise = np.mod(sine * 43758.5453 + y_coords, 1.0)

            dither_shift = quant * self.dithering_intensity
            dither_shift_half = dither_shift * 0.5
            dither_val = dither_shift * noise - dither_shift_half

            luma += dither_val
            luma = np.clip(luma, 0.0, 1.0)

        # 4. 문자 인덱스로 변환
        # 0 ~ num_chars-1 범위의 정수 인덱스
        char_indices = (luma * (num_chars - 1)).astype(np.int32)
        char_indices = np.clip(char_indices, 0, num_chars - 1)

        # 5. 패턴 룩업 테이블을 사용하여 마스크 생성
        patterns = self._generate_font_patterns(
            font_w, font_h, num_chars
        )  # (num_chars, block_h, block_w)

        # Fancy indexing으로 (grid_h, grid_w)의 각 위치에 (block_h, block_w) 패턴 배치
        # mask shape: (grid_h, grid_w, block_h, block_w)
        mask = patterns[char_indices]

        # 차원 재배치하여 전체 이미지 마스크로 변환
        # (grid_h, grid_w, block_h, block_w) -> (grid_h, block_h, grid_w, block_w)
        mask = mask.transpose(0, 2, 1, 3)

        # (grid_h * block_h, grid_w * block_w)
        full_mask = mask.reshape(grid_h * block_h, grid_w * block_w)

        # 원본 이미지 크기에 맞게 크롭 (혹은 패딩)
        # 리사이즈 과정에서 버려진 자투리 부분은 검은색 등으로 처리되거나 잘림
        target_h = grid_h * block_h
        target_w = grid_w * block_w

        # 색상 합성
        font_col = np.array(self.font_color, dtype=np.float32)
        bg_col = np.array(self.background_color, dtype=np.float32)

        # Swap colors 처리
        if self.swap_colors:
            font_col, bg_col = bg_col, font_col

        # 색상 모드 처리
        # full_mask 확장 (H, W, 1)
        mask_3d = full_mask[:, :, np.newaxis]

        # 배경 이미지 준비
        # small_img를 다시 확대 (Nearest Neighbor)하여 각 블록 영역이 동일한 색상을 갖게 함
        # upscale = cv2.resize(small_img_float, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # 더 빠른 방법: repeat (Kronecker product)
        # small_img_float (grid_h, grid_w, 3)
        upscaled_color = np.repeat(
            np.repeat(small_img_float, block_h, axis=0), block_w, axis=1
        )

        output = np.zeros((target_h, target_w, 3), dtype=np.float32)

        if self.font_color_mode == 2:  # Full color
            # 문자 부분: 마스크 * 원본색 * 문자값? (원본 Shader 참조)
            # Shader: color = (character) ? character * color : background;
            # 여기서는 mask가 0 or 1 (character val)
            # mask가 1인 부분은 upscaled_color 사용, 0인 부분은 bg_col 사용
            output = mask_3d * upscaled_color + (1.0 - mask_3d) * bg_col

        elif self.font_color_mode == 1:  # Colorized grayscale
            # Shader: color = (character) ? font_color * gray : background;
            # gray 값도 upscale 필요
            luma_upscaled = np.repeat(np.repeat(luma, block_h, axis=0), block_w, axis=1)
            luma_upscaled = luma_upscaled[:, :, np.newaxis]

            output = mask_3d * (font_col * luma_upscaled) + (1.0 - mask_3d) * bg_col

        else:  # Font color (Solid)
            # Shader: color = (character) ? font_color : background;
            output = mask_3d * font_col + (1.0 - mask_3d) * bg_col

        # 결과 이미지가 원본보다 작을 수 있으므로 원본 크기의 캔버스에 배치
        result_img = np.zeros_like(image, dtype=np.uint8)

        # float -> uint8
        output_uint8 = (np.clip(output, 0, 1) * 255).astype(np.uint8)

        # 중앙 정렬 또는 좌상단 정렬 (여기선 좌상단)
        result_img[:target_h, :target_w] = output_uint8

        return result_img
