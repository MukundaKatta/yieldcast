"""Tests for the yield prediction model."""

from datetime import date, timedelta

import numpy as np
import pytest

from yieldcast.models import (
    CROP_YIELD_RANGES,
    Crop,
    CropType,
    Field,
    SatelliteReading,
    SoilSample,
    WeatherData,
)
from yieldcast.predictor.yield_model import FEATURE_NAMES, YieldPredictor


def _make_field(crop_type: CropType = CropType.CORN) -> Field:
    return Field(
        field_id="T001",
        name="Test Field",
        area_acres=100.0,
        latitude=40.0,
        longitude=-90.0,
        crop=Crop(crop_type=crop_type, planting_date=date(2025, 4, 15)),
        soil=SoilSample(
            ph=6.5, organic_matter_pct=4.0, nitrogen_ppm=40.0,
            phosphorus_ppm=35.0, potassium_ppm=200.0, moisture_pct=28.0,
            cec=15.0, sand_pct=40.0, silt_pct=40.0, clay_pct=20.0,
        ),
    )


def _make_weather(n_days: int = 180) -> list[WeatherData]:
    start = date(2025, 4, 1)
    return [
        WeatherData(
            date=start + timedelta(days=i),
            temp_max_f=80.0,
            temp_min_f=60.0,
            precipitation_in=0.1,
            humidity_pct=65.0,
            solar_radiation_mj=18.0,
            wind_speed_mph=8.0,
        )
        for i in range(n_days)
    ]


def _make_satellite(n: int = 10) -> list[SatelliteReading]:
    return [
        SatelliteReading(
            date=date(2025, 4, 1) + timedelta(days=i * 15),
            red_band=0.1,
            nir_band=0.6,
            field_id="T001",
        )
        for i in range(n)
    ]


class TestYieldPredictor:
    def setup_method(self):
        self.predictor = YieldPredictor()

    def test_feature_vector_shape(self):
        field = _make_field()
        weather = _make_weather()
        satellite = _make_satellite()
        vec = self.predictor.build_feature_vector(
            CropType.CORN, weather, field.soil, satellite
        )
        assert vec.shape == (len(FEATURE_NAMES),)

    def test_heuristic_prediction(self):
        field = _make_field()
        weather = _make_weather()
        satellite = _make_satellite()
        pred = self.predictor.predict_yield(field, weather, satellite)
        low, _, high = CROP_YIELD_RANGES[CropType.CORN]
        assert low <= pred.predicted_yield <= high
        assert pred.confidence_low < pred.predicted_yield < pred.confidence_high
        assert pred.crop_type == CropType.CORN

    def test_prediction_scores_bounded(self):
        field = _make_field()
        pred = self.predictor.predict_yield(field, _make_weather(), _make_satellite())
        assert 0.0 <= pred.weather_score <= 1.0
        assert 0.0 <= pred.soil_score <= 1.0
        assert 0.0 <= pred.ndvi_score <= 1.0

    def test_train_and_predict(self):
        field = _make_field()
        weather = _make_weather()
        satellite = _make_satellite()

        # Generate training data
        n_samples = 50
        X = np.random.default_rng(42).random((n_samples, len(FEATURE_NAMES)))
        y = np.random.default_rng(42).uniform(120, 250, n_samples)

        self.predictor.train(CropType.CORN, X, y)
        pred = self.predictor.predict_yield(field, weather, satellite)
        assert pred.predicted_yield > 0

    def test_no_crop_raises(self):
        field = Field(
            field_id="T001", name="Empty", area_acres=100.0,
            latitude=40.0, longitude=-90.0,
        )
        with pytest.raises(ValueError, match="no crop"):
            self.predictor.predict_yield(field, _make_weather(), _make_satellite())

    def test_no_soil_raises(self):
        field = Field(
            field_id="T001", name="No Soil", area_acres=100.0,
            latitude=40.0, longitude=-90.0,
            crop=Crop(crop_type=CropType.CORN, planting_date=date(2025, 4, 15)),
        )
        with pytest.raises(ValueError, match="no soil"):
            self.predictor.predict_yield(field, _make_weather(), _make_satellite())

    def test_multiple_crop_types(self):
        for crop_type in [CropType.SOYBEANS, CropType.RICE, CropType.COTTON]:
            field = _make_field(crop_type)
            pred = self.predictor.predict_yield(field, _make_weather(), _make_satellite())
            assert pred.crop_type == crop_type
            assert pred.predicted_yield > 0
