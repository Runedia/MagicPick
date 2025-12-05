"""
다국어 번역 시스템

한국어/영어 UI 전환을 지원합니다.
"""

from typing import Dict

# 번역 데이터
TRANSLATIONS: Dict[str, Dict[str, str]] = {
    # === 메뉴 ===
    "menu.file": {"ko": "파일", "en": "File"},
    "menu.edit": {"ko": "편집", "en": "Edit"},
    "menu.capture": {"ko": "캡처", "en": "Capture"},
    "menu.filter": {"ko": "필터", "en": "Filter"},
    "menu.tone": {"ko": "색조", "en": "Tone"},
    "menu.style": {"ko": "스타일", "en": "Style"},
    "menu.shader": {"ko": "셰이더", "en": "Shader"},
    "menu.settings": {"ko": "설정", "en": "Settings"},
    # === 파일 메뉴 ===
    "file.open": {"ko": "열기", "en": "Open"},
    "file.save": {"ko": "저장", "en": "Save"},
    "file.save_as": {"ko": "다른 이름으로 저장", "en": "Save As"},
    "file.exit": {"ko": "끝내기", "en": "Exit"},
    # === 편집 메뉴 ===
    "edit.undo": {"ko": "실행 취소", "en": "Undo"},
    "edit.redo": {"ko": "다시 실행", "en": "Redo"},
    "edit.reset": {"ko": "초기화", "en": "Reset"},
    "edit.rotate": {"ko": "회전", "en": "Rotate"},
    "edit.flip_horizontal": {"ko": "좌우 반전", "en": "Flip Horizontal"},
    "edit.flip_vertical": {"ko": "상하 반전", "en": "Flip Vertical"},
    "edit.crop": {"ko": "자르기", "en": "Crop"},
    # 자르기 메뉴 탭
    "menu.crop": {"ko": "✂ 자르기", "en": "✂ Crop"},
    # === 자르기 다이얼로그 ===
    "crop.title": {"ko": "이미지 자르기", "en": "Crop Image"},
    "crop.instruction": {
        "ko": "핸들을 드래그하여 자르기 영역을 조정하세요",
        "en": "Drag handles to adjust the crop area",
    },
    "crop.size_info": {
        "ko": "선택 영역: {width} x {height}",
        "en": "Selection: {width} x {height}",
    },
    "crop.no_image": {"ko": "자를 이미지가 없습니다", "en": "No image to crop"},
    "crop.cancelled": {"ko": "자르기 취소됨", "en": "Crop cancelled"},
    "crop.applied": {"ko": "자르기 적용됨", "en": "Crop applied"},
    "crop.reset": {"ko": "영역 초기화", "en": "Reset Area"},
    # === 캡처 메뉴 ===
    "capture.fullscreen": {"ko": "전체화면", "en": "Fullscreen"},
    "capture.region": {"ko": "영역 지정", "en": "Region"},
    "capture.window": {"ko": "윈도우", "en": "Window"},
    "capture.monitor": {"ko": "모니터", "en": "Monitor"},
    # === 필터 메뉴 ===
    "filter.soft": {"ko": "부드러운", "en": "Soft"},
    "filter.sharp": {"ko": "선명한", "en": "Sharp"},
    "filter.warm": {"ko": "따뜻한", "en": "Warm"},
    "filter.cool": {"ko": "차가운", "en": "Cool"},
    "filter.grayscale": {"ko": "회색조", "en": "Grayscale"},
    "filter.sepia": {"ko": "세피아", "en": "Sepia"},
    "filter.photo_filter": {"ko": "Photo Filter", "en": "Photo Filter"},
    # === 색조 메뉴 ===
    "tone.brightness": {"ko": "밝기", "en": "Brightness"},
    "tone.contrast": {"ko": "대비", "en": "Contrast"},
    "tone.saturation": {"ko": "채도", "en": "Saturation"},
    "tone.gamma": {"ko": "감마", "en": "Gamma"},
    # === 스타일 메뉴 ===
    "style.cartoon": {"ko": "카툰", "en": "Cartoon"},
    "style.sketch": {"ko": "스케치", "en": "Sketch"},
    "style.oil_painting": {"ko": "유화", "en": "Oil Painting"},
    "style.film_grain": {"ko": "필름 그레인", "en": "Film Grain"},
    "style.vintage": {"ko": "빈티지", "en": "Vintage"},
    "style.mosaic": {"ko": "모자이크", "en": "Mosaic"},
    "style.gaussian_blur": {"ko": "가우시안 블러", "en": "Gaussian Blur"},
    "style.average_blur": {"ko": "평균 블러", "en": "Average Blur"},
    "style.median_blur": {"ko": "중앙값 블러", "en": "Median Blur"},
    "style.sharpen": {"ko": "샤프닝", "en": "Sharpen"},
    "style.emboss": {"ko": "엠보싱", "en": "Emboss"},
    # === 셰이더 메뉴 ===
    "shader.reshade_load": {"ko": "ReShade 불러오기", "en": "Load ReShade"},
    "shader.performance": {"ko": "성능 측정", "en": "Performance"},
    # === 설정 다이얼로그 ===
    "settings.title": {"ko": "설정", "en": "Settings"},
    "settings.tab.general": {"ko": "일반", "en": "General"},
    "settings.tab.capture": {"ko": "캡처", "en": "Capture"},
    "settings.tab.advanced": {"ko": "고급", "en": "Advanced"},
    # 일반 탭
    "settings.language": {"ko": "언어", "en": "Language"},
    "settings.language.korean": {"ko": "한국어", "en": "Korean"},
    "settings.language.english": {"ko": "English", "en": "English"},
    "settings.start_with_windows": {
        "ko": "Windows 시작 시 실행",
        "en": "Start with Windows",
    },
    "settings.start_with_windows.note": {
        "ko": "(개발 중에는 비활성화됨)",
        "en": "(Disabled during development)",
    },
    # 캡처 탭
    "settings.hotkeys": {"ko": "단축키 설정", "en": "Hotkey Settings"},
    "settings.hotkey.fullscreen": {"ko": "전체화면 캡처", "en": "Fullscreen Capture"},
    "settings.hotkey.region": {"ko": "영역 지정 캡처", "en": "Region Capture"},
    "settings.hotkey.window": {"ko": "윈도우 캡처", "en": "Window Capture"},
    "settings.hotkey.monitor": {"ko": "모니터 캡처", "en": "Monitor Capture"},
    "settings.save_location": {"ko": "저장 위치", "en": "Save Location"},
    "settings.browse": {"ko": "찾아보기...", "en": "Browse..."},
    "settings.filename_format": {"ko": "파일 이름 형식", "en": "Filename Format"},
    "settings.filename_format.hint": {
        "ko": "{datetime}, {date}, {time} 사용 가능",
        "en": "Use {datetime}, {date}, {time}",
    },
    "settings.default_format": {"ko": "기본 파일 형식", "en": "Default File Format"},
    "settings.auto_save": {"ko": "캡처 후 자동 저장", "en": "Auto-save after capture"},
    # 고급 탭
    "settings.capture_options": {"ko": "캡처 옵션", "en": "Capture Options"},
    "settings.sound_enabled": {"ko": "알림음 활성화", "en": "Enable Sound"},
    "settings.clipboard_copy": {
        "ko": "클립보드 자동 복사",
        "en": "Auto-copy to Clipboard",
    },
    # 버튼
    "button.ok": {"ko": "확인", "en": "OK"},
    "button.cancel": {"ko": "취소", "en": "Cancel"},
    "button.apply": {"ko": "적용", "en": "Apply"},
    "button.reset": {"ko": "초기화", "en": "Reset"},
    # === 트레이 ===
    "tray.show_main_window": {"ko": "창 열기", "en": "Open Window"},
    "tray.quit": {"ko": "종료", "en": "Quit"},
    # === 상태바 ===
    "status.ready": {"ko": "준비", "en": "Ready"},
    "status.no_image": {"ko": "이미지 없음", "en": "No Image"},
    "status.processing": {"ko": "처리 중...", "en": "Processing..."},
    "status.saved": {"ko": "저장됨", "en": "Saved"},
    "status.capture_saved": {
        "ko": "캡처가 저장되었습니다: {path}",
        "en": "Capture saved: {path}",
    },
    "status.applied": {"ko": "{action} 적용됨", "en": "{action} applied"},
    "status.error": {"ko": "오류: {message}", "en": "Error: {message}"},
    "status.unknown_transform": {
        "ko": "알 수 없는 변형: {type}",
        "en": "Unknown transform: {type}",
    },
    # === 편집 상태 메시지 ===
    "edit.adjustment_cancelled": {"ko": "조정 취소됨", "en": "Adjustment cancelled"},
    "edit.rotation_cancelled": {"ko": "회전 취소됨", "en": "Rotation cancelled"},
    "edit.reset_complete": {
        "ko": "원본 이미지로 초기화됨",
        "en": "Reset to original image",
    },
    "edit.undo_unavailable": {
        "ko": "실행 취소할 작업이 없습니다",
        "en": "Nothing to undo",
    },
    "edit.redo_unavailable": {
        "ko": "다시 실행할 작업이 없습니다",
        "en": "Nothing to redo",
    },
    "edit.rotation_preview": {
        "ko": "미리보기: {angle}도 회전",
        "en": "Preview: {angle}° rotation",
    },
    "edit.rotation_zero": {"ko": "회전 각도가 0도입니다", "en": "Rotation angle is 0°"},
    # === 크기 조절 ===
    "edit.resize": {"ko": "크기 조절", "en": "Resize"},
    "resize.title": {"ko": "크기 조절", "en": "Resize"},
    "resize.no_resize": {"ko": "크기 변경 안함", "en": "No Resize"},
    "resize.maintain_aspect": {"ko": "비율 유지", "en": "Maintain Aspect"},
    "resize.fit_width": {"ko": "폭맞춤", "en": "Fit Width"},
    "resize.fit_height": {"ko": "높이맞춤", "en": "Fit Height"},
    "resize.add_padding": {"ko": "여백 붙이기", "en": "Add Padding"},
    "resize.crop_to_fit": {"ko": "여백 자르기", "en": "Crop to Fit"},
    "resize.stretch": {"ko": "꽉차게 늘리기", "en": "Stretch"},
    "resize.upscale_small": {
        "ko": "작은 이미지도 지정한 크기로 리사이징",
        "en": "Upscale small images to target size",
    },
    "resize.original_size": {"ko": "원본 크기:", "en": "Original Size:"},
    "resize.result_size": {"ko": "결과 크기:", "en": "Result Size:"},
    # === 다이얼로그 ===
    "dialog.about.title": {"ko": "프로그램 정보", "en": "About"},
    "dialog.select_folder": {"ko": "폴더 선택", "en": "Select Folder"},
}


class Translator:
    """
    번역 관리자

    현재 언어 설정에 따라 문자열을 번역합니다.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        # 저장된 언어 설정 불러오기
        self._current_language = self._load_saved_language()

    def _load_saved_language(self) -> str:
        """저장된 언어 설정을 불러옵니다."""
        try:
            from config.settings import settings

            lang = settings.get("general/language")
            if lang in ("ko", "en"):
                return lang
        except Exception:
            pass
        return "ko"  # 기본값

    def set_language(self, lang: str) -> None:
        """
        현재 언어를 설정합니다.

        Args:
            lang: 언어 코드 ("ko" 또는 "en")
        """
        if lang in ("ko", "en"):
            self._current_language = lang

    def get_language(self) -> str:
        """현재 언어를 반환합니다."""
        return self._current_language

    def tr(self, key: str, **kwargs) -> str:
        """
        키에 해당하는 번역된 문자열을 반환합니다.

        Args:
            key: 번역 키 (예: "menu.file")
            **kwargs: 문자열 포맷팅에 사용할 인자

        Returns:
            번역된 문자열. 키가 없으면 키 자체를 반환
        """
        if key in TRANSLATIONS:
            text = TRANSLATIONS[key].get(self._current_language, key)
            if kwargs:
                try:
                    text = text.format(**kwargs)
                except KeyError:
                    pass
            return text
        return key


# 편의를 위한 전역 인스턴스 및 함수
translator = Translator()


def tr(key: str, **kwargs) -> str:
    """번역 함수 (단축형)"""
    return translator.tr(key, **kwargs)


def set_language(lang: str) -> None:
    """언어 설정 함수 (단축형)"""
    translator.set_language(lang)


def get_language() -> str:
    """현재 언어 반환 함수 (단축형)"""
    return translator.get_language()
