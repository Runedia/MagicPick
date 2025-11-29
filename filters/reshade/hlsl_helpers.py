"""
HLSL 내장 함수 Python 구현 헬퍼 모듈

HLSL 셰이더를 Python/NumPy로 변환할 때 사용하는 유틸리티 함수들
"""

import numpy as np


def saturate(x):
    """HLSL saturate 함수: 값을 [0, 1] 범위로 클램핑"""
    return np.clip(x, 0.0, 1.0)


def clamp(x, min_val, max_val):
    """HLSL clamp 함수: 값을 [min, max] 범위로 클램핑"""
    return np.clip(x, min_val, max_val)


def lerp(a, b, t):
    """HLSL lerp 함수: 선형 보간"""
    return a + t * (b - a)


def smoothstep(edge0, edge1, x):
    """HLSL smoothstep 함수: 부드러운 Hermite 보간"""
    t = saturate((x - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)


def dot(a, b):
    """HLSL dot 함수: 내적"""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a * b
    if len(a.shape) == 1:
        return np.dot(a, b)
    return np.sum(a * b, axis=-1, keepdims=True)


def dot3(rgb, weights):
    """
    3채널 RGB 이미지와 3요소 가중치 벡터 간의 내적 계산

    Args:
        rgb: RGB 이미지 (H, W, 3)
        weights: 가중치 벡터 (3,) 또는 (1, 1, 3)

    Returns:
        내적 결과 (H, W) 또는 (H, W, 1)
    """
    if len(weights.shape) == 1:
        # weights가 (3,) 형태일 때
        result = np.sum(rgb * weights, axis=2)
    else:
        # weights가 (1, 1, 3) 형태일 때
        result = np.sum(rgb * weights, axis=2)
    return result


def length(x):
    """HLSL length 함수: 벡터 길이"""
    if isinstance(x, (int, float)):
        return abs(x)
    if len(x.shape) == 2:
        return x
    return np.sqrt(np.sum(x * x, axis=-1, keepdims=True))


def normalize(x):
    """HLSL normalize 함수: 벡터 정규화"""
    return x / (length(x) + 1e-8)


def pow_safe(x, y):
    """HLSL pow 함수: 안전한 거듭제곱 (음수 처리)"""
    return np.power(np.abs(x), y)


def exp2(x):
    """HLSL exp2 함수: 2^x"""
    return np.power(2.0, x)


def rcp(x):
    """HLSL rcp 함수: 역수 (1/x)"""
    return 1.0 / (x + 1e-8)


def rsqrt(x):
    """HLSL rsqrt 함수: 제곱근의 역수 (1/sqrt(x))"""
    return 1.0 / (np.sqrt(x + 1e-8))


def sqr(x):
    """제곱 함수"""
    return x * x


def max4(a, b, c, d):
    """4개 값 중 최대값"""
    return np.maximum(np.maximum(a, b), np.maximum(c, d))


def any_greater(x, threshold):
    """HLSL any 함수: 어떤 채널이라도 threshold보다 크면 True"""
    if len(x.shape) == 3:
        return np.any(x > threshold, axis=2, keepdims=True)
    return x > threshold


def all_greater(x, threshold):
    """HLSL all 함수: 모든 채널이 threshold보다 크면 True"""
    if len(x.shape) == 3:
        return np.all(x > threshold, axis=2, keepdims=True)
    return x > threshold


def soft_lim_tanh_approx(v, s):
    """
    Soft limit, modified tanh approximation (fast_ops = 1)
    saturate(abs(v/s)*(27 + sqr(v/s))/(27 + 9*sqr(v/s)))*s
    """
    ratio = v / (s + 1e-8)
    ratio_sq = sqr(ratio)
    result = saturate(np.abs(ratio) * (27 + ratio_sq) / (27 + 9 * ratio_sq)) * s
    return result


def soft_lim_tanh_exact(v, s):
    """
    Soft limit, exact tanh (fast_ops = 0)
    (exp(2*min(abs(v), s*24)/s) - 1)/(exp(2*min(abs(v), s*24)/s) + 1)*s
    """
    clamped = np.minimum(np.abs(v), s * 24)
    exp_val = np.exp(2 * clamped / (s + 1e-8))
    result = ((exp_val - 1) / (exp_val + 1)) * s
    return result


def wpmean(a, b, w, pm_p):
    """
    Weighted power mean
    pow(abs(w)*pow(abs(a), pm_p) + abs(1-w)*pow(abs(b), pm_p), (1.0/pm_p))
    """
    term_a = np.abs(w) * pow_safe(a, pm_p)
    term_b = np.abs(1 - w) * pow_safe(b, pm_p)
    result = pow_safe(term_a + term_b, 1.0 / pm_p)
    return result


def rgb_to_luma_fast(rgb):
    """
    Fast gamma-aware luma conversion
    sqrt(dot(float3(0.2558, 0.6511, 0.0931), sqr(rgb)))
    """
    coeffs = np.array([0.2558, 0.6511, 0.0931])
    luma = np.sqrt(np.sum(sqr(rgb) * coeffs, axis=2, keepdims=True))
    return luma


def rgb_to_luma_bt709(rgb):
    """BT.709 luma conversion"""
    coeffs = np.array([0.2126, 0.7152, 0.0722])
    luma = np.sum(rgb * coeffs, axis=2, keepdims=True)
    return luma


def rgb_to_yuv(rgb_image: np.ndarray) -> np.ndarray:
    """
    RGB 이미지를 YUV 색 공간으로 변환합니다 (BT.709 표준).
    Shader의 `mul(RGBtoYUV(0.0722, 0.2126), color)` 와 동일

    Args:
        rgb_image: RGB 이미지 (NumPy array, float32, 0-255)

    Returns:
        YUV 이미지 (NumPy array, float32, Y: 0-255, U/V: 0-255 with 127.5 offset)
    """
    # BT.709 계수
    kr = 0.2126
    kb = 0.0722
    kg = 1.0 - kr - kb

    # RGB to YUV 변환 행렬
    transform_matrix = np.array(
        [
            [kr, kg, kb],
            [-0.5 * kr / (1.0 - kb), -0.5 * kg / (1.0 - kb), 0.5],
            [0.5, -0.5 * kg / (1.0 - kr), -0.5 * kb / (1.0 - kr)],
        ]
    )

    img_float = rgb_image / 255.0
    yuv = np.einsum("ij, ...j -> ...i", transform_matrix, img_float)

    # ReShade는 U와 V에 0.5 오프셋을 더함
    yuv[:, :, 1:] += 0.5

    return yuv * 255.0


def yuv_to_rgb(yuv_image: np.ndarray) -> np.ndarray:
    """
    YUV 이미지를 RGB 색 공간으로 변환합니다 (BT.709 표준).
    Shader의 `mul(YUVtoRGB(0.0722, 0.2126), color)` 와 동일

    Args:
        yuv_image: YUV 이미지 (NumPy array, float32, Y: 0-255, U/V: 0-255 with 127.5 offset)

    Returns:
        RGB 이미지 (NumPy array, float32)
    """
    # BT.709 계수
    kr = 0.2126
    kb = 0.0722
    kg = 1.0 - kr - kb

    # YUV to RGB 변환 행렬
    inverse_transform_matrix = np.array(
        [
            [1.0, 0.0, 2.0 * (1.0 - kr)],
            [1.0, -2.0 * (1.0 - kb) * kb / kg, -2.0 * (1.0 - kr) * kr / kg],
            [1.0, 2.0 * (1.0 - kb), 0.0],
        ]
    )

    img_float = yuv_image / 255.0
    img_float[:, :, 1:] -= 0.5  # U, V 오프셋 제거

    rgb = np.einsum("ij, ...j -> ...i", inverse_transform_matrix, img_float)

    return rgb * 255.0


def shift_image_approx(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    이미지를 주어진 dx, dy (픽셀 단위)만큼 이동 (Nearest Neighbor with Edge Clamp).

    Args:
        image: 입력 이미지 (H, W, C)
        dx: X축 이동량 (float)
        dy: Y축 이동량 (float)

    Returns:
        이동된 이미지 (H, W, C)
    """
    dx_int = int(round(dx))
    dy_int = int(round(dy))

    if dx_int == 0 and dy_int == 0:
        return image.copy()

    h, w = image.shape[:2]

    # 필요한 패딩 계산
    pad_y = abs(dy_int)
    pad_x = abs(dx_int)

    # 가장자리 복제 패딩 (Clamp-to-edge)
    # 3차원 배열(H, W, C)이라고 가정
    if image.ndim == 3:
        padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="edge")
    else:
        padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")

    # 슬라이싱으로 이동된 이미지 추출
    # 원본의 (0,0)은 패딩된 이미지의 (pad_y, pad_x)
    # 이동된 이미지의 (0,0)값은 원본의 (-dy, -dx) 위치에서 가져와야 함
    # 즉, 패딩된 이미지에서 (pad_y - dy, pad_x - dx) 위치

    start_y = pad_y - dy_int
    start_x = pad_x - dx_int

    return padded[start_y : start_y + h, start_x : start_x + w]


def rgb_to_hsl(rgb: np.ndarray) -> np.ndarray:
    """
    RGB 이미지를 HSL 색 공간으로 변환합니다.

    Args:
        rgb: RGB 이미지 (NumPy array, float32, 0-1)

    Returns:
        HSL 이미지 (NumPy array, float32, 0-1)
    """
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    max_c = np.maximum(r, np.maximum(g, b))
    min_c = np.minimum(r, np.minimum(g, b))
    chroma = max_c - min_c

    # Lightness
    l = (max_c + min_c) / 2.0

    h = np.zeros_like(r)
    s = np.zeros_like(r)

    mask = chroma > 1e-7

    # Saturation
    # L=0 또는 L=1일 때 분모가 0이 되는 것 방지
    denom = 1.0 - np.abs(2.0 * l[mask] - 1.0)
    s[mask] = chroma[mask] / (denom + 1e-8)

    # Hue
    mask_r = mask & (max_c == r)
    mask_g = mask & (max_c == g) & (max_c != r)  # 우선순위 처리
    mask_b = mask & (max_c == b) & (max_c != r) & (max_c != g)

    # R max: (G-B)/C
    h[mask_r] = (g[mask_r] - b[mask_r]) / chroma[mask_r]
    # G max: (B-R)/C + 2
    h[mask_g] = (b[mask_g] - r[mask_g]) / chroma[mask_g] + 2.0
    # B max: (R-G)/C + 4
    h[mask_b] = (r[mask_b] - g[mask_b]) / chroma[mask_b] + 4.0

    h = (h / 6.0) % 1.0

    return np.stack([h, s, l], axis=2)


def hsl_to_rgb(hsl: np.ndarray) -> np.ndarray:
    """
    HSL 이미지를 RGB 색 공간으로 변환합니다.

    Args:
        hsl: HSL 이미지 (NumPy array, float32, 0-1)

    Returns:
        RGB 이미지 (NumPy array, float32, 0-1)
    """
    h = hsl[:, :, 0]
    s = hsl[:, :, 1]
    l = hsl[:, :, 2]

    def hue_to_rgb_channel(p, q, t):
        t = (t + 1.0) % 1.0

        mask1 = t < 1 / 6
        mask2 = (~mask1) & (t < 1 / 2)
        mask3 = (~mask1) & (~mask2) & (t < 2 / 3)
        # mask4 = rest

        val = np.zeros_like(t)
        val[mask1] = p[mask1] + (q[mask1] - p[mask1]) * 6.0 * t[mask1]
        val[mask2] = q[mask2]
        val[mask3] = p[mask3] + (q[mask3] - p[mask3]) * (2 / 3 - t[mask3]) * 6.0
        val[~mask1 & ~mask2 & ~mask3] = p[~mask1 & ~mask2 & ~mask3]

        return val

    q = np.where(l < 0.5, l * (1.0 + s), l + s - l * s)
    p = 2.0 * l - q

    r = hue_to_rgb_channel(p, q, h + 1 / 3)
    g = hue_to_rgb_channel(p, q, h)
    b = hue_to_rgb_channel(p, q, h - 1 / 3)

    return np.stack([r, g, b], axis=2)
