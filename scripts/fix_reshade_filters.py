"""
filters/reshade 폴더 내 모든 필터 파일의 코딩 스타일 통일 스크립트

BaseFilter 상속 패턴 통일:
- super().__init__(name, description) 형식으로 변경
- 파라미터 처리는 params.get() 사용
"""

import re
from pathlib import Path

# 필터 이름과 설명 매핑
FILTER_METADATA = {
    "adaptive_sharpen_accurate.py": (
        "AdaptiveSharpen",
        "적응형 샤프닝 (정확한 2-pass 구현)",
    ),
    "artistic_vignette.py": ("ArtisticVignette", "예술적 비네팅 효과"),
    "ascii.py": ("ASCII", "ASCII 아트 효과"),
    "bloom.py": ("Bloom", "기본 블룸 효과"),
    "border.py": ("Border", "테두리 효과"),
    "cartoon.py": ("Cartoon", "카툰 효과"),
    "cas.py": ("CAS", "AMD Contrast Adaptive Sharpening"),
    "chromatic_aberration.py": ("ChromaticAberration", "렌즈 색수차 효과"),
    "color_matrix.py": ("ColorMatrix", "색상 행렬 변환"),
    "colourfulness.py": ("Colourfulness", "채도 강화"),
    "comic.py": ("Comic", "코믹 효과"),
    "deband.py": ("Deband", "디밴딩 (밴딩 제거)"),
    "deblur.py": ("Deblur", "디블러 효과"),
    "extended_levels.py": ("ExtendedLevels", "확장 레벨 조정"),
    "fake_hdr.py": ("FakeHDR", "가짜 HDR 효과"),
    "filmic_pass.py": ("FilmicPass", "시네마틱 패스"),
    "film_grain2.py": ("FilmGrain2", "필름 그레인 v2"),
    "fine_sharp.py": ("FineSharp", "파인 샤프"),
    "gaussian_bloom.py": ("GaussianBloom", "가우시안 블룸"),
    "gaussian_blur.py": ("GaussianBlur", "가우시안 블러"),
    "high_pass_sharpen.py": ("HighPassSharpen", "하이패스 샤프닝"),
    "hsl_shift.py": ("HSLShift", "HSL 색공간 시프트"),
    "hue_fx.py": ("HueFX", "색조 조정"),
    "lens_distort.py": ("LensDistort", "렌즈 왜곡"),
    "levels_accurate.py": ("Levels", "레벨 조정 (정확)"),
    "levels_plus.py": ("LevelsPlus", "고급 레벨 조정"),
    "level_io.py": ("LevelIO", "입출력 레벨 조정"),
    "lift_gamma_gain.py": ("LiftGammaGain", "리프트/감마/게인 조정"),
    "luma_sharpen_accurate.py": ("LumaSharpen", "루마 기반 언샤프 마스크 (정확)"),
    "magic_bloom.py": ("MagicBloom", "매직 블룸"),
    "monochrome.py": ("Monochrome", "흑백 변환"),
    "oilify.py": ("Oilify", "유화 효과"),
    "pd80_cbs.py": ("PD80CBS", "대비/밝기/채도 조정"),
    "pd80_color_balance.py": (
        "PD80ColorBalance",
        "색상 균형 (그림자/중간톤/하이라이트)",
    ),
    "pd80_color_gamut.py": ("PD80ColorGamut", "색역 조정"),
    "pd80_color_space_curves.py": ("PD80ColorSpaceCurves", "색공간 커브"),
    "pd80_color_temperature.py": ("PD80ColorTemperature", "색온도 조정"),
    "pd80_correct_color.py": ("PD80CorrectColor", "색상 보정"),
    "pd80_correct_contrast.py": ("PD80CorrectContrast", "대비 보정"),
    "pd80_posterize_pixelate.py": ("PD80PosterizePixelate", "포스터화/픽셀화"),
    "pd80_smh.py": ("PD80SMH", "그림자/중간톤/하이라이트 RGB 조정"),
    "pd80_technicolor.py": ("PD80Technicolor", "PD80 테크니컬러"),
    "remove_tint.py": ("RemoveTint", "틴트 제거"),
    "simple_bloom.py": ("SimpleBloom", "심플 블룸"),
    "simple_filters_accurate.py": ("SimpleFilters", "심플 필터 (정확)"),
    "simple_grain.py": ("SimpleGrain", "심플 그레인"),
    "sketch.py": ("Sketch", "스케치 효과"),
    "surface_blur.py": ("SurfaceBlur", "표면 블러 (엣지 보존)"),
    "swirl.py": ("Swirl", "스월 효과"),
    "technicolor.py": ("Technicolor", "테크니컬러 2-strip"),
    "technicolor2.py": ("Technicolor2", "테크니컬러 3-strip"),
    "unsharp.py": ("Unsharp", "언샤프 마스크"),
    "vibrance_accurate.py": ("Vibrance", "지능형 채도 부스트 (정확)"),
    "zigzag.py": ("ZigZag", "지그재그 왜곡"),
}


def fix_init_method(file_path: Path):
    """__init__ 메서드를 표준 패턴으로 수정"""
    filename = file_path.name

    if filename not in FILTER_METADATA:
        print(f"⚠️  건너뜀: {filename} (메타데이터 없음)")
        return False

    filter_name, description = FILTER_METADATA[filename]

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    original_content = content

    # 패턴 1: super().__init__(params) 수정
    pattern1 = r"def __init__\(self, params=None\):\s*\n\s*super\(\).__init__\(params\)"
    replacement1 = f'def __init__(self):\n        super().__init__("{filter_name}", "{description}")'
    content = re.sub(pattern1, replacement1, content)

    # 패턴 2: super().__init__() + self.name 수정
    pattern2 = r'def __init__\(self, params=None\):\s*\n\s*super\(\).__init__\(\)\s*\n\s*self\.name = "[^"]*"\s*\n\s*self\.description = "[^"]*"'
    content = re.sub(pattern2, replacement1, content)

    # 패턴 3: __init__(self, params: dict) 수정
    pattern3 = r'def __init__\(self, params: dict\):\s*\n\s*super\(\).__init__\("[^"]*", "[^"]*"\)'
    content = re.sub(pattern3, replacement1, content)

    if content != original_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ 수정됨: {filename}")
        return True
    else:
        print(f"ℹ️  변경 없음: {filename}")
        return False


def main():
    """메인 실행 함수"""
    reshade_dir = Path("filters/reshade")

    if not reshade_dir.exists():
        print(f"❌ 오류: {reshade_dir} 폴더를 찾을 수 없습니다.")
        return

    py_files = list(reshade_dir.glob("*.py"))
    py_files = [f for f in py_files if f.name not in ["__init__.py", "hlsl_helpers.py"]]

    print(f"📁 총 {len(py_files)}개 파일 검사 중...\n")

    modified_count = 0
    for py_file in sorted(py_files):
        if fix_init_method(py_file):
            modified_count += 1

    print(f"\n✨ 완료: {modified_count}개 파일 수정됨")


if __name__ == "__main__":
    main()
