"""Tests for pydantic data models."""

from datetime import date

import pytest

from yieldcast.models import (
    CROP_GDD_REQUIREMENTS,
    CROP_PLANTING_WINDOWS,
    CROP_YIELD_RANGES,
    Crop,
    CropType,
    Field,
    FertilizerRecommendation,
    PlantingRecommendation,
    SatelliteReading,
    SoilSample,
    WeatherData,
    YieldPrediction,
)


class TestCropType:
    def test_all_crop_types_have_yield_ranges(self):
        for crop in CropType:
            assert crop in CROP_YIELD_RANGES
            low, typical, high = CROP_YIELD_RANGES[crop]
            assert low < typical < high

    def test_all_crop_types_have_gdd_requirements(self):
        for crop in CropType:
            assert crop in CROP_GDD_REQUIREMENTS
            low, high = CROP_GDD_REQUIREMENTS[crop]
            assert low < high

    def test_all_crop_types_have_planting_windows(self):
        for crop in CropType:
            assert crop in CROP_PLANTING_WINDOWS

    def test_at_least_12_crop_types(self):
        assert len(CropType) >= 12


class TestWeatherData:
    def test_valid_weather(self):
        w = WeatherData(
            date=date(2025, 6, 15),
            temp_max_f=85.0,
            temp_min_f=62.0,
            precipitation_in=0.5,
            humidity_pct=70.0,
            solar_radiation_mj=18.0,
            wind_speed_mph=8.0,
        )
        assert w.temp_max_f == 85.0

    def test_negative_precipitation_rejected(self):
        with pytest.raises(Exception):
            WeatherData(
                date=date(2025, 6, 15),
                temp_max_f=85.0,
                temp_min_f=62.0,
                precipitation_in=-1.0,
                humidity_pct=70.0,
                solar_radiation_mj=18.0,
                wind_speed_mph=8.0,
            )


class TestSoilSample:
    def test_valid_soil(self):
        s = SoilSample(
            ph=6.5,
            organic_matter_pct=3.5,
            nitrogen_ppm=40.0,
            phosphorus_ppm=30.0,
            potassium_ppm=200.0,
            moisture_pct=25.0,
            cec=15.0,
            sand_pct=40.0,
            silt_pct=40.0,
            clay_pct=20.0,
        )
        assert s.ph == 6.5

    def test_ph_out_of_range(self):
        with pytest.raises(Exception):
            SoilSample(
                ph=12.0,
                organic_matter_pct=3.5,
                nitrogen_ppm=40.0,
                phosphorus_ppm=30.0,
                potassium_ppm=200.0,
                moisture_pct=25.0,
                cec=15.0,
                sand_pct=40.0,
                silt_pct=40.0,
                clay_pct=20.0,
            )


class TestField:
    def test_field_with_crop_and_soil(self):
        field = Field(
            field_id="F001",
            name="Test Field",
            area_acres=100.0,
            latitude=40.0,
            longitude=-90.0,
            crop=Crop(crop_type=CropType.CORN, planting_date=date(2025, 4, 15)),
            soil=SoilSample(
                ph=6.5, organic_matter_pct=3.5, nitrogen_ppm=40.0,
                phosphorus_ppm=30.0, potassium_ppm=200.0, moisture_pct=25.0,
                cec=15.0, sand_pct=40.0, silt_pct=40.0, clay_pct=20.0,
            ),
        )
        assert field.crop.crop_type == CropType.CORN
        assert field.area_acres == 100.0


class TestYieldPrediction:
    def test_valid_prediction(self):
        pred = YieldPrediction(
            field_id="F001",
            crop_type=CropType.CORN,
            predicted_yield=180.0,
            confidence_low=160.0,
            confidence_high=200.0,
            unit="bu/acre",
            weather_score=0.75,
            soil_score=0.80,
            ndvi_score=0.70,
        )
        assert pred.predicted_yield == 180.0
        assert pred.confidence_low < pred.confidence_high


class TestSatelliteReading:
    def test_valid_reading(self):
        r = SatelliteReading(
            date=date(2025, 6, 1),
            red_band=0.1,
            nir_band=0.6,
            field_id="F001",
        )
        assert r.nir_band == 0.6
