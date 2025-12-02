# compile_aot.py
"""
Numba Helpers를 AOT로 컴파일해서 _numba_hlsl_helpers.so 파일 생성
python compile_aot.py   ← 한 번만 실행하면 끝!
"""

import numpy as np
from numba.pycc import CC

cc = CC("numba_hlsl_helpers")  # 생성될 .so 파일 이름의 베이스
cc.verbose = True

# -----------------------------------------------------------------------------
# Scalar Math Helpers
# -----------------------------------------------------------------------------
cc.export("saturate", "f8(f8)")(lambda x: min(max(x, 0.0), 1.0))
cc.export("clamp", "f8(f8, f8, f8)")(
    lambda x, min_val, max_val: min(max(x, min_val), max_val)
)
cc.export("lerp", "f8(f8, f8, f8)")(lambda a, b, t: a + t * (b - a))


def smoothstep(edge0, edge1, x):
    t = min(max((x - edge0) / (edge1 - edge0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


cc.export("smoothstep", "f8(f8, f8, f8)")(smoothstep)

cc.export("sqr", "f8(f8)")(lambda x: x * x)
cc.export("pow_safe", "f8(f8, f8)")(lambda x, y: abs(x) ** y)

# -----------------------------------------------------------------------------
# Vector/Color Helpers
# -----------------------------------------------------------------------------
cc.export("dot3", "f8(f8,f8,f8,f8,f8,f8)")(
    lambda a_r, a_g, a_b, b_r, b_g, b_b: a_r * b_r + a_g * b_g + a_b * b_b
)

cc.export("get_luma_fast", "f8(f8,f8,f8)")(
    lambda r, g, b: np.sqrt(r * r * 0.2558 + g * g * 0.6511 + b * b * 0.0931)
)

cc.export("get_luma_bt709", "f8(f8,f8,f8)")(
    lambda r, g, b: r * 0.2126 + g * 0.7152 + b * 0.0722
)

cc.export("get_luma_bt601", "f8(f8,f8,f8)")(
    lambda r, g, b: r * 0.299 + g * 0.587 + b * 0.114
)


# -----------------------------------------------------------------------------
# Advanced Math Helpers
# -----------------------------------------------------------------------------
def soft_lim_tanh_approx(v, s):
    if s == 0:
        return 0.0
    ratio = v / s
    ratio_sq = ratio * ratio
    return (
        min(max(abs(ratio) * (27.0 + ratio_sq) / (27.0 + 9.0 * ratio_sq), 0.0), 1.0) * s
    )


cc.export("soft_lim_tanh_approx", "f8(f8, f8)")(soft_lim_tanh_approx)


def wpmean(a, b, w, pm_p):
    term_a = abs(w) * (abs(a) ** pm_p)
    term_b = abs(1.0 - w) * (abs(b) ** pm_p)
    return (term_a + term_b) ** (1.0 / pm_p)


cc.export("wpmean", "f8(f8, f8, f8, f8)")(wpmean)

if __name__ == "__main__":
    cc.compile()
    print("\nNumba AOT 컴파일 완료!")
    print("생성된 파일: _numba_hlsl_helpers.*.so")
    print("이제 numba import 없이 바로 초고속으로 사용 가능합니다!")
