"""Tests for weather feature extraction."""

from datetime import date, timedelta

import pytest

from yieldcast.models import CropType, WeatherData
from yieldcast.predictor.weather import WeatherFeatureExtractor


def _make_day(d: date, high: float = 80.0, low: float = 60.0, precip: float = 0.0) -> WeatherData:
    return WeatherData(
        date=d,
        temp_max_f=high,
        temp_min_f=low,
        precipitation_in=precip,
        humidity_pct=65.0,
        solar_radiation_mj=18.0,
        wind_speed_mph=8.0,
    )


def _make_season(n_days: int = 180, high: float = 80.0, low: float = 60.0, precip: float = 0.1) -> list[WeatherData]:
    start = date(2025, 4, 1)
    return [_make_day(start + timedelta(days=i), high, low, precip) for i in range(n_days)]


class TestWeatherFeatureExtractor:
    def setup_method(self):
        self.ext = WeatherFeatureExtractor()

    def test_gdd_basic(self):
        days = _make_season(10, high=80.0, low=60.0)
        # avg temp = 70F, GDD per day = 70 - 50 = 20
        gdd = self.ext.compute_gdd(days)
        assert gdd == pytest.approx(200.0)

    def test_gdd_cold_days_zero(self):
        days = _make_season(5, high=45.0, low=35.0)
        # avg temp = 40F < 50F base, so GDD = 0
        gdd = self.ext.compute_gdd(days)
        assert gdd == 0.0

    def test_total_precipitation(self):
        days = _make_season(10, precip=0.5)
        total = self.ext.total_precipitation(days)
        assert total == pytest.approx(5.0)

    def test_drought_index_no_drought(self):
        days = _make_season(30, precip=0.2)
        idx = self.ext.drought_index(days)
        assert idx == 0.0

    def test_drought_index_severe(self):
        days = _make_season(30, precip=0.0)
        idx = self.ext.drought_index(days)
        assert idx > 0.5

    def test_heat_stress_days(self):
        hot_days = _make_season(5, high=100.0, low=75.0)
        cool_days = _make_season(5, high=80.0, low=60.0)
        all_days = hot_days + cool_days
        count = self.ext.heat_stress_days(all_days)
        assert count == 5

    def test_cold_stress_days(self):
        cold_days = _make_season(3, high=40.0, low=28.0)
        warm_days = _make_season(7, high=80.0, low=60.0)
        count = self.ext.cold_stress_days(cold_days + warm_days)
        assert count == 3

    def test_gdd_fulfillment_ratio(self):
        # 180 days at avg 70F => GDD = 180 * 20 = 3600
        days = _make_season(180, high=80.0, low=60.0)
        ratio = self.ext.gdd_fulfillment_ratio(days, CropType.CORN)
        assert ratio > 1.0  # exceeds corn midpoint of 2600

    def test_extract_features_keys(self):
        days = _make_season(30)
        features = self.ext.extract_features(days, CropType.CORN)
        expected_keys = {
            "gdd_total", "gdd_fulfillment", "precip_total_in",
            "precip_distribution", "drought_index", "heat_stress_days",
            "cold_stress_days", "heat_stress_frac", "avg_solar_mj",
        }
        assert set(features.keys()) == expected_keys

    def test_empty_weather(self):
        assert self.ext.compute_gdd([]) == 0.0
        assert self.ext.drought_index([]) == 0.0
        assert self.ext.avg_solar_radiation([]) == 0.0
