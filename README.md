# MagicPick

Python 기반 Windows 이미지 처리 및 편집 애플리케이션

## 프로젝트 개요

화면 캡처 기능과 다양한 필터 및 편집 도구를 통합한 사용자 친화적인 이미지 처리 프로그램입니다.

### 주요 특징

- 직관적이고 접근 가능한 **Ribbon 스타일** 사용자 인터페이스
- 다양한 이미지 처리 필터 및 효과 제공 (70개+ ReShade 필터 포함)
- Windows 환경에 최적화된 독립 실행형 애플리케이션
- 실시간 이미지 미리보기 및 Undo/Redo 기능
- 시스템 트레이 백그라운드 실행 및 전역 단축키 지원
- 다국어 지원 (한국어/영어)

## 기술 스택

### 개발 환경
- Python 3.13 이상
- Windows 11 25H2

### 핵심 라이브러리
| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| **PyQt5** | 5.15+ | GUI 인터페이스 |
| **Pillow** | 10.0+ | 기본 이미지 처리 |
| **NumPy** | 1.24+ | 픽셀 레벨 연산 및 필터 구현 |
| **OpenCV** | 4.8+ | 고급 이미지 처리 |
| **mss** | 9.0+ | 화면 캡처 |
| **pywin32** | - | Windows API 연동 |
| **keyboard** | - | 전역 단축키 |
| **Numba** | 0.60+ | JIT 컴파일 최적화 |

### 배포
- **PyInstaller** - 독립 실행형 .exe 변환

## 주요 기능

### 🖼️ 화면 캡처
| 기능 | 단축키 | 설명 |
|------|--------|------|
| 전체 화면 캡처 | `Ctrl+Shift+F1` | 모든 모니터 전체 영역 캡처 |
| 영역 지정 캡처 | `Ctrl+Shift+F2` | 드래그로 사용자 정의 영역 캡처 |
| 윈도우 캡처 | `Ctrl+Shift+F3` | 특정 윈도우 선택 캡처 |
| 모니터 캡처 | `Ctrl+Shift+F4` | 선택한 모니터 캡처 (다중 모니터 지원) |

### 🎨 필터 및 효과

#### 기본 필터 (8종)
- 부드러운 (Soft), 선명한 (Sharp)
- 따뜻한 (Warm), 차가운 (Cool)
- 회색조 (Grayscale), 세피아 (Sepia)
- 반전 (Invert), 비네팅 (Vignette)

#### Photoshop 스타일 Photo Filter
- 10가지 프리셋: Warming 85/81, Cooling 80/82, Underwater, Sepia, Deep 색상들
- 필터 강도 조절 (0-100%)
- Luminosity 보존 옵션

#### 픽셀 기반 효과 (6종)
- 모자이크, 가우시안 블러, 평균 블러, 중앙값 블러
- 샤프닝, 엠보싱

#### 예술적 효과 (5종)
- 카툰 효과, 스케치 효과, 유화 효과
- 필름 그레인, 빈티지 효과

#### ReShade 스타일 효과 (70개+)
- ReShade 설정 파일(.ini) 파싱 및 적용
- Bloom 계열: NeoBloom, MagicBloom, OrtonBloom, HDRBloom 등
- Sharpen 계열: AdaptiveSharpen, FilmicAnamorphSharpen, QuintSharp 등
- Color 계열: DPX, Technicolor, Curves, Levels, ColorMatrix 등
- Effect 계열: ASCII, Comic, LiquidLens, PerfectPerspective 등
- Numba JIT 최적화로 고성능 처리

### ✏️ 기본 편집

#### 이미지 변형
- **크기 조절**: 7가지 모드 (비율 유지, 폭맞춤, 높이맞춤, 여백 붙이기, 여백 자르기, 꽉차게 늘리기)
- **회전**: 90°, 180°, 270°, 임의 각도
- **반전**: 좌우/상하 대칭 변환
- **자르기**: 드래그를 통한 사용자 정의 영역 자르기

#### 이미지 조정
- 밝기 조절 (-100 ~ +100)
- 대비 조절 (-100 ~ +100)
- 채도 조절 (0 ~ 200%)
- 감마 보정

### 📁 파일 관리
- **지원 형식**: JPG, PNG, BMP, GIF, TIFF, WebP
- **Undo/Redo**: 최근 20단계 작업 복구
- **실시간 미리보기**: 필터 및 조정 효과 미리보기

## 설치 및 실행

### 요구사항
```
pillow>=11.3.0
numpy>=2.2.0
PyQt5>=5.15.0
mss>=10.1.0
opencv-python-headless>=4.12.0
pywin32>=311
pygetwindow>=0.0.9
ruff>=0.14.0
numba>=0.62.0
pyinstaller>=6.17.0
rich
```

### 설치
```bash
# 저장소 클론
git clone https://github.com/Runedia/MagicPick.git
cd MagicPick

# 의존성 설치
pip install -r requirements.txt
```

### 실행
```bash
python main.py
```

### 빌드 (exe 생성)
```bash
pyinstaller build_config.spec
```

## 프로젝트 구조

```
MagicPick/
├── main.py                 # 애플리케이션 진입점
├── requirements.txt        # 의존성 목록
├── build_config.spec       # PyInstaller 설정
│
├── assets/                 # 리소스 (아이콘, 폰트, 사운드)
│
├── config/                 # 설정 관리
│   ├── settings.py         # 애플리케이션 설정
│   ├── translations.py     # 다국어 번역 (한국어/영어)
│   └── reshade_config.py   # ReShade 프리셋 관리
│
├── services/               # 백그라운드 서비스
│   ├── singleton.py        # 중복 실행 방지
│   └── tray_service.py     # 시스템 트레이 서비스
│
├── ui/                     # 사용자 인터페이스
│   ├── main_window.py      # 메인 윈도우 (Mixin 패턴)
│   ├── menu_bar.py         # Ribbon 메뉴바
│   ├── toolbar.py          # 자동 숨김 툴바
│   ├── mixins/             # MainWindow Mixin 모듈 (6개)
│   ├── dialogs/            # 모달 다이얼로그 (16개)
│   └── widgets/            # 커스텀 위젯
│
├── capture/                # 화면 캡처
│   ├── fullscreen.py       # 전체 화면 캡처
│   ├── region.py           # 영역 지정 캡처
│   ├── window.py           # 윈도우 캡처
│   ├── monitor.py          # 모니터 캡처
│   └── screen_capture.py   # 캡처 통합 관리
│
├── filters/                # 이미지 필터
│   ├── base_filter.py      # 필터 베이스 클래스
│   ├── basic_filters.py    # 기본 필터 (8종)
│   ├── photo_filter.py     # Photo Filter
│   ├── pixel_effects.py    # 픽셀 효과 (6종)
│   ├── artistic.py         # 예술적 효과 (5종)
│   └── reshade/            # ReShade 필터 (70개+)
│
├── editor/                 # 이미지 편집
│   ├── adjustments.py      # 이미지 조정
│   └── transform.py        # 이미지 변형
│
└── utils/                  # 유틸리티
    ├── file_manager.py     # 파일 관리
    ├── history.py          # Undo/Redo 관리
    └── global_hotkey.py    # 전역 단축키
```

## UI 특징

### Ribbon 메뉴바
- 7가지 메뉴 카테고리: 파일, 편집, 캡처, 필터, 색조, 스타일, 셰이더
- 메뉴 선택 시 해당 도구 목록이 툴바에 표시
- 설정 버튼으로 언어, 단축키, 캡처 옵션 변경

### 오버레이 툴바
- 선택한 메뉴에 따라 동적으로 변경
- 오버레이 방식으로 작업 영역을 가리지 않음
- 3초 후 자동 숨김 (마우스 오버 시 재표시)
- SHIFT + 휠 스크롤을 통한 가로 스크롤 지원

### 시스템 트레이
- 백그라운드 실행 지원
- 트레이 아이콘에서 빠른 캡처 접근
- 전역 단축키로 어디서든 캡처 가능

## 성능 최적화

- **Numba JIT**: ReShade 필터에 JIT 컴파일 적용
- **NumPy 벡터화**: 필터 연산 최적화
- **AOT 컴파일**: 초기 로딩 시간 단축
- 필터 처리 시간: 1920x1080 이미지 기준 5초 이내

## 시스템 요구사항

| 항목 | 요구사항 |
|------|----------|
| OS | Windows 11 이상 |
| 프로세서 | Intel/AMD 듀얼코어 이상 |
| 메모리 | 4GB 이상 권장 |
| 디스크 공간 | 약 100MB |

## 라이선스

- [MIT License](./LICENSE)

# 추가 구현 계획

## pyfxr 라이브러리를 활용한 fx 파일 실행
- CPU기반 코드에서 GPU 사용 가능