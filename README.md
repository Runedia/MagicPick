# MagicPick

Python 기반 Windows 이미지 처리 및 편집 애플리케이션

## 프로젝트 개요

화면 캡처 기능과 다양한 필터 및 편집 도구를 통합한 사용자 친화적인 이미지 처리 프로그램입니다.

### 주요 특징

- 직관적이고 접근 가능한 Ribbon 스타일 사용자 인터페이스
- 다양한 이미지 처리 필터 및 효과 제공
- Windows 환경에 최적화된 독립 실행형 애플리케이션
- 실시간 이미지 미리보기 및 Undo/Redo 기능

## 기술 스택

### 개발 환경
- Python 3.11 이상
- Windows 11 25H2

### 핵심 라이브러리
- **Pillow (PIL)** - 기본 이미지 처리
- **NumPy** - 픽셀 레벨 연산 및 필터 구현
- **PyQt5** - GUI 인터페이스
- **mss** - 화면 캡처 기능

### 배포
- **PyInstaller** - 독립 실행형 .exe 변환

## 주요 기능

### 화면 캡처
- **전체 화면 캡처** - 화면 전체 영역 한 번에 캡처
- **영역 지정 캡처** - 드래그를 통한 사용자 정의 영역 캡처
- **활성 윈도우 캡처** - 활성화된 윈도우를 캡처
- **모니터 캡처** - 선택한 모니터를 캡처 (다중 모니터 지원)

### 필터 및 효과

#### 기본 필터
- 부드러운, 선명한, 따뜻한, 차가운
- 회색조, 세피아, 반전, 비네팅

#### Photoshop 스타일 Photo Filter
- 색온도 조정 (Warming/Cooling 필터)
- 필터 강도 조절 (0-100%)
- Luminosity 보존 옵션

#### 픽셀 기반 효과
- 모자이크 - 지정된 픽셀 크기로 이미지 픽셀화
- 블러 효과 - 가우시안 블러, 평균 블러, 중앙값 블러
- 샤프닝 - 이미지 선명도 향상
- 엠보싱 - 3D 입체 효과

#### 예술적 효과
- 카툰 효과 - 만화 스타일 변환
- 스케치 효과 - 손으로 그린 스케치 느낌
- 유화 효과 - Oil Painting 스타일 변환
- 빈티지 효과 - 오래된 사진 스타일

#### ReShade 스타일 효과
- ReShade 설정 파일(.ini) 적용 지원
- 정적 이미지에서 구현 가능한 필터 적용

### 기본 편집

#### 이미지 변형
- 크기 조정 - 픽셀 단위로 이미지 해상도 변경
- 회전 - 90도, 180도, 270도 회전
- 좌우/상하 반전 - 수평/수직 대칭 변환
- 자르기 - 드래그를 통한 사용자 정의 영역 자르기
- 형식 변환 - JPG, PNG, BMP 상호 변경

#### 이미지 조정
- 밝기 조절 (-100 ~ +100)
- 대비 조절 (-100 ~ +100)
- 채도 조절 (0 ~ 200%)
- 감마 보정 - 중간톤 보정

### 파일 관리
- 이미지 불러오기 - JPG, PNG, BMP, GIF, TIFF, WebP 지원
- 이미지 저장 - 다양한 포맷으로 내보내기
- 실행 취소/재실행 - 최근 20단계 작업 복구 (Undo/Redo)
- 실시간 미리보기

## 설치 및 실행

### 요구사항
```
pillow>=10.0.0
numpy>=1.24.0
PyQt5>=5.15.0
mss>=9.0.0
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

## 프로젝트 구조

```
MagicPick/
├── main.py                 # 애플리케이션 진입점
├── requirements.txt        # 의존성 목록
├── config/
│   └── settings.py        # 설정 관리
├── ui/
│   ├── main_window.py     # 메인 윈도우
│   ├── menu_bar.py        # Ribbon 메뉴바
│   ├── toolbar.py         # 도구바
│   ├── dialogs/           # 모달 창들
│   └── widgets/           # 커스텀 위젯들
├── capture/
│   ├── fullscreen.py      # 전체 화면 캡처
│   ├── region.py          # 영역 지정 캡처
│   ├── window.py          # 윈도우 캡처
│   └── monitor.py         # 모니터 캡처
├── filters/
│   ├── basic_filters.py   # 기본 필터
│   ├── color_filters.py   # 색온도 필터
│   ├── photo_filter.py    # Photoshop Photo Filter
│   ├── pixel_effects.py   # 픽셀 기반 효과
│   ├── artistic.py        # 예술적 효과
│   └── reshade.py         # ReShade 스타일 효과
├── editor/
│   ├── adjustments.py     # 이미지 조정
│   ├── transform.py       # 변형
│   └── file_handler.py    # 파일 입출력
└── utils/
    ├── history.py         # Undo/Redo 관리
    ├── file_manager.py    # 파일 관리
    └── image_utils.py     # 이미지 유틸리티
```

## UI 특징

### Ribbon 메뉴바
- 7가지 메뉴 카테고리: 파일, 편집, 캡처, 필터, 색조, 스타일, 셰이더
- 메뉴 선택 시 해당 도구 목록이 툴바에 표시

### 툴바 (Toolbar)
- 선택한 메뉴에 따라 동적으로 변경
- 오버레이 방식으로 작업 영역을 가리지 않음
- 3초 후 자동 숨김 (옵션 설정 가능)
- 마우스 오버 시 재표시
- SHIFT + 휠 스크롤을 통한 가로 스크롤 지원
- 텍스트 길이에 따른 버튼 크기 자동 조정 (최소 100px)

### 이미지 뷰어
- 중앙 이미지 표시 영역
- 실시간 처리 결과 미리보기

### 상태표시줄
- 처리 시간, 파일 정보 표시

## 성능 목표

- 필터 처리 시간: 일반적인 이미지(1920x1080) 기준 5초 이내
- 메모리 효율적 관리
- UI 반응 지연 최소화

## 시스템 요구사항

- OS: Windows 7 이상
- 프로세서: Intel/AMD 듀얼코어 이상
- 메모리: 4GB 이상 권장
- 디스크 공간: 약 50-100MB

## 개발자

- 20210948 김지호

## 라이선스
- [MIT License](./LICENSE)