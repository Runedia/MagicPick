"""
Numba Accelerated HLSL Helpers
HLSL 셰이더 함수들의 Numba JIT 구현 모음
"""

import numpy as np
from numba import njit

# -----------------------------------------------------------------------------
# Scalar Math Helpers
# -----------------------------------------------------------------------------


@njit(fastmath=True, inline="always", cache=True)
def saturate(x):
    """값을 [0, 1] 범위로 클램핑"""
    return min(max(x, 0.0), 1.0)


@njit(fastmath=True, inline="always", cache=True)
def clamp(x, min_val, max_val):
    """값을 [min, max] 범위로 클램핑"""
    return min(max(x, min_val), max_val)


@njit(fastmath=True, inline="always", cache=True)
def lerp(a, b, t):
    """선형 보간 (Linear Interpolation)"""
    return a + t * (b - a)


@njit(fastmath=True, inline="always", cache=True)
def smoothstep(edge0, edge1, x):
    """Hermite 보간"""
    t = saturate((x - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)


@njit(fastmath=True, inline="always", cache=True)
def sqr(x):
    """제곱"""
    return x * x


@njit(fastmath=True, inline="always", cache=True)
def pow_safe(x, y):
    """안전한 거듭제곱 (절댓값 사용)"""
    return abs(x) ** y


# -----------------------------------------------------------------------------
# Vector/Color Helpers
# -----------------------------------------------------------------------------


@njit(fastmath=True, inline="always", cache=True)
def dot3(a_r, a_g, a_b, b_r, b_g, b_b):
    """3성분 벡터 내적 (개별 컴포넌트 입력)"""
    return a_r * b_r + a_g * b_g + a_b * b_b


@njit(fastmath=True, inline="always", cache=True)
def get_luma_fast(r, g, b):
    """
    Fast gamma-aware luma conversion
    sqrt(dot(float3(0.2558, 0.6511, 0.0931), sqr(rgb)))
    """
    return np.sqrt(r * r * 0.2558 + g * g * 0.6511 + b * b * 0.0931)


@njit(fastmath=True, inline="always", cache=True)
def get_luma_bt709(r, g, b):
    """BT.709 luma conversion"""
    return r * 0.2126 + g * 0.7152 + b * 0.0722


@njit(fastmath=True, inline="always", cache=True)
def get_luma_bt601(r, g, b):
    """BT.601 luma conversion (SDTV)"""
    return r * 0.299 + g * 0.587 + b * 0.114


# -----------------------------------------------------------------------------
# Advanced Math Helpers (from AdaptiveSharpen etc.)
# -----------------------------------------------------------------------------


@njit(fastmath=True, inline="always", cache=True)
def soft_lim_tanh_approx(v, s):
    """Soft limit, modified tanh approximation"""
    if s == 0:
        return 0.0
    ratio = v / s
    ratio_sq = ratio * ratio
    return saturate(abs(ratio) * (27.0 + ratio_sq) / (27.0 + 9.0 * ratio_sq)) * s


@njit(fastmath=True, inline="always", cache=True)
def wpmean(a, b, w, pm_p):
    """Weighted power mean"""
    term_a = abs(w) * pow_safe(a, pm_p)
    term_b = abs(1.0 - w) * pow_safe(b, pm_p)
    return pow_safe(term_a + term_b, 1.0 / pm_p)
