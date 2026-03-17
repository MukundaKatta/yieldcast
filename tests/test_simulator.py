"""Tests for synthetic data generation."""

import pytest

from yieldcast.models import CropType
from yieldcast.simulator import FarmSimulator


class TestFarmSimulator:
    def setup_method(self):
        self.sim = FarmSimulator(seed=42)

    def test_generate_soil_quality_levels(self):
        for quality in ["poor", "medium", "good"]:
            soil = self.sim.generate_soil(quality)
            assert 3.0 <= soil.ph <= 10.0
            assert soil.nitrogen_ppm >= 0
            assert soil.sand_pct + soil.silt_pct + soil.clay_pct == pytest.approx(100.0, abs=0.5)

    def test_generate_weather_season_length(self):
        weather = self.sim.generate_weather_season(2025)
        # March through October = roughly 240 days
        assert len(weather) > 200

    def test_weather_temperatures_reasonable(self):
        weather = self.sim.generate_weather_season(2025)
        for day in weather:
            assert day.temp_max_f > day.temp_min_f
            assert day.precipitation_in >= 0
            assert 0 <= day.humidity_pct <= 100

    def test_generate_satellite_readings(self):
        readings = self.sim.generate_satellite_readings("F001", 2025, n_readings=10)
        assert len(readings) == 10
        for r in readings:
            assert 0.0 <= r.red_band <= 1.0
            assert 0.0 <= r.nir_band <= 1.0
            assert r.field_id == "F001"

    def test_generate_field_complete(self):
        field, weather, satellite = self.sim.generate_field(
            "F001", CropType.CORN, 2025, "good"
        )
        assert field.field_id == "F001"
        assert field.crop is not None
        assert field.crop.crop_type == CropType.CORN
        assert field.soil is not None
        assert len(weather) > 0
        assert len(satellite) > 0

    def test_generate_farm_multiple_fields(self):
        farm = self.sim.generate_farm(n_fields=8, year=2025)
        assert len(farm) == 8
        crop_types = {f.crop.crop_type for f, _, _ in farm}
        assert len(crop_types) >= 3  # variety of crops

    def test_reproducible_with_seed(self):
        sim1 = FarmSimulator(seed=123)
        sim2 = FarmSimulator(seed=123)
        field1, _, _ = sim1.generate_field("F001", CropType.CORN, 2025)
        field2, _, _ = sim2.generate_field("F001", CropType.CORN, 2025)
        assert field1.soil.ph == field2.soil.ph
        assert field1.area_acres == field2.area_acres
