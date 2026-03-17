"""Tests for planting and fertilizer optimizers."""

from datetime import date, timedelta

import pytest

from yieldcast.models import CropType, SoilSample, WeatherData
from yieldcast.optimizer.fertilizer import FertilizerRecommender
from yieldcast.optimizer.planting import PlantingOptimizer


def _make_weather_history(year: int = 2025) -> list[WeatherData]:
    """Generate a year of weather history."""
    days = []
    start = date(year, 1, 1)
    for i in range(365):
        d = start + timedelta(days=i)
        doy = d.timetuple().tm_yday
        import math
        season = math.sin(math.pi * (doy - 80) / 200)
        high = 50 + 35 * season
        low = high - 20
        precip = 0.1 if i % 3 == 0 else 0.0
        days.append(WeatherData(
            date=d,
            temp_max_f=round(high, 1),
            temp_min_f=round(low, 1),
            precipitation_in=precip,
            humidity_pct=60.0,
            solar_radiation_mj=15.0,
            wind_speed_mph=8.0,
        ))
    return days


class TestPlantingOptimizer:
    def setup_method(self):
        self.optimizer = PlantingOptimizer()
        self.history = _make_weather_history()

    def test_corn_planting_window(self):
        rec = self.optimizer.recommend(CropType.CORN, 2025, self.history)
        assert rec.window_start.month == 4
        assert rec.window_end.month == 5
        assert rec.optimal_date >= rec.window_start
        assert rec.optimal_date <= rec.window_end

    def test_winter_wheat_fall_planting(self):
        history = _make_weather_history(2024) + _make_weather_history(2025)
        rec = self.optimizer.recommend(CropType.WINTER_WHEAT, 2025, history)
        assert rec.optimal_date.month == 9

    def test_frost_risk_bounded(self):
        rec = self.optimizer.recommend(CropType.SOYBEANS, 2025, self.history)
        assert 0.0 <= rec.frost_risk <= 1.0

    def test_soil_temp_check(self):
        rec = self.optimizer.recommend(CropType.CORN, 2025, self.history)
        assert isinstance(rec.soil_temp_ready, bool)

    def test_notes_populated(self):
        rec = self.optimizer.recommend(CropType.CORN, 2025, self.history)
        assert len(rec.notes) > 0

    def test_last_frost_date(self):
        frost_date = self.optimizer.estimate_last_frost_date(self.history, 2025)
        assert frost_date.year == 2025
        assert frost_date.month <= 6


class TestFertilizerRecommender:
    def setup_method(self):
        self.recommender = FertilizerRecommender()

    def test_deficient_soil_gets_recommendations(self):
        soil = SoilSample(
            ph=5.0, organic_matter_pct=1.5, nitrogen_ppm=10.0,
            phosphorus_ppm=10.0, potassium_ppm=80.0, moisture_pct=20.0,
            cec=10.0, sand_pct=50.0, silt_pct=30.0, clay_pct=20.0,
        )
        rec = self.recommender.recommend(soil, CropType.CORN, "F001")
        assert rec.nitrogen_lbs_per_acre > 0
        assert rec.phosphorus_lbs_per_acre > 0
        assert rec.potassium_lbs_per_acre > 0
        assert rec.lime_tons_per_acre > 0

    def test_good_soil_minimal_recommendations(self):
        soil = SoilSample(
            ph=6.5, organic_matter_pct=5.0, nitrogen_ppm=55.0,
            phosphorus_ppm=45.0, potassium_ppm=220.0, moisture_pct=28.0,
            cec=18.0, sand_pct=40.0, silt_pct=40.0, clay_pct=20.0,
        )
        rec = self.recommender.recommend(soil, CropType.CORN, "F001")
        assert rec.nitrogen_lbs_per_acre == 0.0
        assert rec.lime_tons_per_acre == 0.0

    def test_soybeans_no_nitrogen(self):
        soil = SoilSample(
            ph=6.5, organic_matter_pct=3.0, nitrogen_ppm=15.0,
            phosphorus_ppm=35.0, potassium_ppm=200.0, moisture_pct=25.0,
            cec=15.0, sand_pct=40.0, silt_pct=40.0, clay_pct=20.0,
        )
        rec = self.recommender.recommend(soil, CropType.SOYBEANS, "F001")
        assert rec.nitrogen_lbs_per_acre == 0.0
        assert any("fix nitrogen" in n.lower() for n in rec.notes)

    def test_alfalfa_no_nitrogen(self):
        soil = SoilSample(
            ph=6.8, organic_matter_pct=4.0, nitrogen_ppm=10.0,
            phosphorus_ppm=40.0, potassium_ppm=250.0, moisture_pct=25.0,
            cec=15.0, sand_pct=40.0, silt_pct=40.0, clay_pct=20.0,
        )
        rec = self.recommender.recommend(soil, CropType.ALFALFA, "F001")
        assert rec.nitrogen_lbs_per_acre == 0.0

    def test_all_crops_supported(self):
        soil = SoilSample(
            ph=6.0, organic_matter_pct=3.0, nitrogen_ppm=25.0,
            phosphorus_ppm=20.0, potassium_ppm=150.0, moisture_pct=25.0,
            cec=12.0, sand_pct=40.0, silt_pct=40.0, clay_pct=20.0,
        )
        for crop in CropType:
            rec = self.recommender.recommend(soil, crop, "F001")
            assert isinstance(rec.notes, list)
