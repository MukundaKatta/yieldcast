"""GradientBoosting yield prediction model."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from yieldcast.models import (
    CROP_YIELD_RANGES,
    CropType,
    Field,
    SatelliteReading,
    SoilSample,
    WeatherData,
    YieldPrediction,
)
from yieldcast.predictor.satellite import NDVITracker
from yieldcast.predictor.soil import SoilAnalyzer
from yieldcast.predictor.weather import WeatherFeatureExtractor


# Feature order expected by the model
FEATURE_NAMES = [
    "gdd_total",
    "gdd_fulfillment",
    "precip_total_in",
    "precip_distribution",
    "drought_index",
    "heat_stress_days",
    "cold_stress_days",
    "heat_stress_frac",
    "avg_solar_mj",
    "soil_ph",
    "soil_ph_score",
    "soil_om_pct",
    "soil_n_ppm",
    "soil_p_ppm",
    "soil_k_ppm",
    "soil_moisture_pct",
    "soil_cec",
    "soil_n_score",
    "soil_p_score",
    "soil_k_score",
    "soil_overall",
    "ndvi_peak",
    "ndvi_mean",
    "ndvi_trend",
    "ndvi_green_frac",
    "ndvi_vigor",
]


class YieldPredictor:
    """Crop yield predictor using GradientBoosting on weather, soil, and satellite features.

    The model is trained per crop type on extracted feature vectors.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ):
        self.weather_extractor = WeatherFeatureExtractor()
        self.soil_analyzer = SoilAnalyzer()
        self.ndvi_tracker = NDVITracker()
        self._models: dict[CropType, GradientBoostingRegressor] = {}
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._random_state = random_state
        self._is_trained: dict[CropType, bool] = {}

    def _make_model(self) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            random_state=self._random_state,
            loss="squared_error",
            subsample=0.8,
        )

    def build_feature_vector(
        self,
        crop: CropType,
        weather: list[WeatherData],
        soil: SoilSample,
        satellite: list[SatelliteReading],
    ) -> np.ndarray:
        """Build a feature vector from raw inputs."""
        weather_feats = self.weather_extractor.extract_features(weather, crop)
        soil_feats = self.soil_analyzer.extract_features(soil)
        sat_feats = self.ndvi_tracker.extract_features(satellite)

        combined = {**weather_feats, **soil_feats, **sat_feats}
        return np.array([combined[name] for name in FEATURE_NAMES])

    def train(
        self,
        crop: CropType,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Train a model for a specific crop type.

        Args:
            crop: The crop type.
            X: Feature matrix of shape (n_samples, n_features).
            y: Yield values of shape (n_samples,).
        """
        model = self._make_model()
        model.fit(X, y)
        self._models[crop] = model
        self._is_trained[crop] = True

    def predict_yield(
        self,
        field: Field,
        weather: list[WeatherData],
        satellite: list[SatelliteReading],
    ) -> YieldPrediction:
        """Predict yield for a field using trained model or heuristic fallback.

        If no trained model exists for the crop, uses a score-based heuristic
        that maps component scores to the crop's known yield range.
        """
        if field.crop is None:
            raise ValueError(f"Field {field.field_id} has no crop assigned")
        if field.soil is None:
            raise ValueError(f"Field {field.field_id} has no soil data")

        crop = field.crop.crop_type
        features = self.build_feature_vector(crop, weather, field.soil, satellite)

        # Component scores for the prediction result
        weather_feats = self.weather_extractor.extract_features(weather, crop)
        soil_score = self.soil_analyzer.overall_score(field.soil)
        ndvi_score = self.ndvi_tracker.vigor_score(satellite)

        # Weather score: combine GDD fulfillment, precip distribution, inverse drought
        weather_score = (
            0.4 * min(weather_feats["gdd_fulfillment"], 1.0)
            + 0.3 * weather_feats["precip_distribution"]
            + 0.3 * (1.0 - weather_feats["drought_index"])
        )
        weather_score = round(min(max(weather_score, 0.0), 1.0), 4)

        if crop in self._is_trained and self._is_trained[crop]:
            predicted = float(self._models[crop].predict(features.reshape(1, -1))[0])
            low_range, _, high_range = CROP_YIELD_RANGES[crop]
            predicted = max(low_range * 0.5, min(predicted, high_range * 1.2))
        else:
            # Heuristic fallback
            predicted = self._heuristic_yield(crop, weather_score, soil_score, ndvi_score)

        # Confidence interval: +/- based on combined uncertainty
        avg_score = (weather_score + soil_score + ndvi_score) / 3.0
        spread = predicted * (0.20 - 0.10 * avg_score)  # 10-20% spread
        confidence_low = round(max(0.0, predicted - spread), 2)
        confidence_high = round(predicted + spread, 2)

        # Determine unit
        unit = "bu/acre"
        if crop == CropType.RICE:
            unit = "lbs/acre"
        elif crop == CropType.COTTON:
            unit = "lbs lint/acre"
        elif crop == CropType.SUNFLOWER:
            unit = "lbs/acre"
        elif crop == CropType.ALFALFA:
            unit = "tons/acre"

        return YieldPrediction(
            field_id=field.field_id,
            crop_type=crop,
            predicted_yield=round(predicted, 2),
            confidence_low=confidence_low,
            confidence_high=confidence_high,
            unit=unit,
            weather_score=weather_score,
            soil_score=round(soil_score, 4),
            ndvi_score=round(ndvi_score, 4),
        )

    def _heuristic_yield(
        self,
        crop: CropType,
        weather_score: float,
        soil_score: float,
        ndvi_score: float,
    ) -> float:
        """Estimate yield from component scores without a trained ML model."""
        low, typical, high = CROP_YIELD_RANGES[crop]
        composite = 0.35 * weather_score + 0.35 * soil_score + 0.30 * ndvi_score
        # Map composite 0-1 to low-high range
        yield_est = low + composite * (high - low)
        return round(yield_est, 2)
