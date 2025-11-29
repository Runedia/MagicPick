import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import lerp, shift_image_approx


class DenoiseFilter(BaseFilter):
    """
    Denoise 필터 (KNN 알고리즘)

    NVIDIA Denoise.fx의 KNearestNeighbors 기법을 구현했습니다.
    이미지의 노이즈를 줄이면서 엣지를 보존합니다.
    """

    def __init__(self):
        super().__init__("Denoise", "노이즈 제거 (KNN)")
        # 기본값 설정 (Denoise.fx 기준)
        self.noise_level = 0.15
        self.lerp_coefficient = 0.8
        self.weight_threshold = 0.03
        self.counter_threshold = 0.05
        self.gaussian_sigma = 50.0
        self.window_radius = 3

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Denoise 필터 적용

        Args:
            image: 입력 이미지 (H, W, 3)
            **params: 필터 파라미터
                - NoiseLevel: 노이즈 레벨 (0.01 ~ 1.0)
                - LerpCoefficeint: 원본 혼합 비율 (0.0 ~ 1.0)
                - WeightThreshold: 가중치 임계값 (0.0 ~ 1.0)
                - CounterThreshold: 카운터 임계값 (0.0 ~ 1.0)
                - GaussianSigma: 가우시안 시그마 (1.0 ~ 100.0)
                - WindowRadius: 탐색 반경 (1 ~ 10)
        """
        # 파라미터 로드 (오타가 포함된 원본 파라미터 이름 지원)
        self.noise_level = params.get("NoiseLevel", self.noise_level)
        # Denoise.fx의 오타 "LerpCoefficeint" 지원
        self.lerp_coefficient = params.get(
            "LerpCoefficeint", params.get("LerpCoefficient", self.lerp_coefficient)
        )
        self.weight_threshold = params.get("WeightThreshold", self.weight_threshold)
        self.counter_threshold = params.get("CounterThreshold", self.counter_threshold)
        self.gaussian_sigma = params.get("GaussianSigma", self.gaussian_sigma)
        self.window_radius = int(params.get("WindowRadius", self.window_radius))

        # 이미지 정규화
        img_float = image.astype(np.float32) / 255.0

        # 최적화를 위한 상수 계산
        # 0으로 나누기 방지
        inv_noise = 1.0 / max(self.noise_level, 0.001)
        inv_sigma = 1.0 / max(self.gaussian_sigma, 0.001)

        # 결과 누적을 위한 버퍼 초기화
        result_accum = np.zeros_like(img_float)
        weight_sum = np.zeros(img_float.shape[:2], dtype=np.float32)
        counter = np.zeros(img_float.shape[:2], dtype=np.float32)

        # 윈도우 루프 (벡터화된 연산)
        # -WindowRadius ~ +WindowRadius
        for i in range(-self.window_radius, self.window_radius + 1):
            for j in range(-self.window_radius, self.window_radius + 1):
                # 이미지 시프트 (이웃 픽셀 가져오기)
                tex_ij = shift_image_approx(img_float, i, j)

                # 색상 차이 계산 (Squared Euclidean Distance)
                # weight = dot(orig - texIJ, orig - texIJ);
                diff = img_float - tex_ij
                dot_diff = np.sum(diff * diff, axis=2)

                # 거리 제곱
                dist_sq = float(i * i + j * j)

                # 가중치 계산
                # weight = exp(-(weight * rcp(NoiseLevel) + (i * i + j * j) * rcp(GaussianSigma)));
                exponent = -(dot_diff * inv_noise + dist_sq * inv_sigma)
                weight = np.exp(exponent)

                # 카운터 업데이트
                # counter += weight > WeightThreshold;
                counter += (weight > self.weight_threshold).astype(np.float32)

                # 가중치 합산
                # sum += weight;
                weight_sum += weight

                # 결과 누적 (브로드캐스팅)
                # result.rgb += texIJ * weight;
                result_accum += tex_ij * weight[:, :, np.newaxis]

        # 정규화
        # result /= sum;
        weight_sum = np.maximum(weight_sum, 1e-6)  # 0 나누기 방지
        result_normalized = result_accum / weight_sum[:, :, np.newaxis]

        # 보간 계수 결정
        # float iWindowArea = 2.0 * WindowRadius + 1.0;
        # iWindowArea *= iWindowArea;
        window_area = (2.0 * self.window_radius + 1.0) ** 2

        # float lerpQ = (counter > (CounterThreshold * iWindowArea)) ? 1.0 - LerpCoefficeint : LerpCoefficeint;
        condition = counter > (self.counter_threshold * window_area)
        lerp_q = np.where(condition, 1.0 - self.lerp_coefficient, self.lerp_coefficient)

        # 최종 혼합
        # result = lerp(result, orig, lerpQ);
        final_result = lerp(result_normalized, img_float, lerp_q[:, :, np.newaxis])

        # 결과 반환 (0~255 uint8)
        return (np.clip(final_result, 0, 1) * 255).astype(np.uint8)
