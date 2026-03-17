"""Pydantic data models for YIELDCAST."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field as PydanticField


class CropType(str, Enum):
    """Supported crop types with realistic yield ranges (bushels/acre)."""

    CORN = "corn"
    SOYBEANS = "soybeans"
    WINTER_WHEAT = "winter_wheat"
    SPRING_WHEAT = "spring_wheat"
    RICE = "rice"
    COTTON = "cotton"
    BARLEY = "barley"
    SORGHUM = "sorghum"
    OATS = "oats"
    CANOLA = "canola"
    SUNFLOWER = "sunflower"
    ALFALFA = "alfalfa"


# Realistic yield ranges in bushels per acre (low, typical, high)
CROP_YIELD_RANGES: dict[CropType, tuple[float, float, float]] = {
    CropType.CORN: (120.0, 180.0, 250.0),
    CropType.SOYBEANS: (35.0, 50.0, 70.0),
    CropType.WINTER_WHEAT: (40.0, 60.0, 85.0),
    CropType.SPRING_WHEAT: (30.0, 48.0, 70.0),
    CropType.RICE: (6000.0, 7500.0, 9500.0),  # lbs/acre
    CropType.COTTON: (700.0, 900.0, 1200.0),  # lbs lint/acre
    CropType.BARLEY: (50.0, 75.0, 100.0),
    CropType.SORGHUM: (50.0, 75.0, 110.0),
    CropType.OATS: (55.0, 75.0, 100.0),
    CropType.CANOLA: (30.0, 45.0, 65.0),
    CropType.SUNFLOWER: (1200.0, 1600.0, 2100.0),  # lbs/acre
    CropType.ALFALFA: (3.0, 5.0, 8.0),  # tons/acre
}

# Optimal growing degree day requirements (base 50F)
CROP_GDD_REQUIREMENTS: dict[CropType, tuple[float, float]] = {
    CropType.CORN: (2400.0, 2800.0),
    CropType.SOYBEANS: (2200.0, 2600.0),
    CropType.WINTER_WHEAT: (1800.0, 2200.0),
    CropType.SPRING_WHEAT: (1600.0, 2000.0),
    CropType.RICE: (2800.0, 3400.0),
    CropType.COTTON: (2200.0, 2800.0),
    CropType.BARLEY: (1400.0, 1800.0),
    CropType.SORGHUM: (2400.0, 3000.0),
    CropType.OATS: (1200.0, 1600.0),
    CropType.CANOLA: (1800.0, 2200.0),
    CropType.SUNFLOWER: (2000.0, 2600.0),
    CropType.ALFALFA: (1800.0, 2400.0),
}

# Optimal planting windows as (earliest_month, latest_month)
CROP_PLANTING_WINDOWS: dict[CropType, tuple[int, int]] = {
    CropType.CORN: (4, 5),
    CropType.SOYBEANS: (5, 6),
    CropType.WINTER_WHEAT: (9, 10),
    CropType.SPRING_WHEAT: (3, 4),
    CropType.RICE: (4, 5),
    CropType.COTTON: (4, 6),
    CropType.BARLEY: (3, 4),
    CropType.SORGHUM: (5, 6),
    CropType.OATS: (3, 4),
    CropType.CANOLA: (4, 5),
    CropType.SUNFLOWER: (5, 6),
    CropType.ALFALFA: (4, 5),
}


class WeatherData(BaseModel):
    """Daily weather observations for a location."""

    date: date
    temp_max_f: float = PydanticField(description="Daily high temperature (F)")
    temp_min_f: float = PydanticField(description="Daily low temperature (F)")
    precipitation_in: float = PydanticField(
        ge=0.0, description="Daily precipitation (inches)"
    )
    humidity_pct: float = PydanticField(
        ge=0.0, le=100.0, description="Relative humidity %"
    )
    solar_radiation_mj: float = PydanticField(
        ge=0.0, description="Solar radiation (MJ/m2)"
    )
    wind_speed_mph: float = PydanticField(ge=0.0, description="Wind speed (mph)")


class SoilSample(BaseModel):
    """Soil test results for a field."""

    ph: float = PydanticField(ge=3.0, le=10.0, description="Soil pH")
    organic_matter_pct: float = PydanticField(
        ge=0.0, le=15.0, description="Organic matter %"
    )
    nitrogen_ppm: float = PydanticField(ge=0.0, description="Available nitrogen (ppm)")
    phosphorus_ppm: float = PydanticField(
        ge=0.0, description="Available phosphorus (ppm)"
    )
    potassium_ppm: float = PydanticField(
        ge=0.0, description="Available potassium (ppm)"
    )
    moisture_pct: float = PydanticField(
        ge=0.0, le=100.0, description="Soil moisture %"
    )
    cec: float = PydanticField(
        ge=0.0, description="Cation exchange capacity (meq/100g)"
    )
    sand_pct: float = PydanticField(ge=0.0, le=100.0)
    silt_pct: float = PydanticField(ge=0.0, le=100.0)
    clay_pct: float = PydanticField(ge=0.0, le=100.0)


class Crop(BaseModel):
    """Crop planted in a field."""

    crop_type: CropType
    variety: str = "generic"
    planting_date: date
    expected_harvest_date: Optional[date] = None


class Field(BaseModel):
    """A farm field with its properties."""

    field_id: str
    name: str
    area_acres: float = PydanticField(gt=0.0)
    latitude: float = PydanticField(ge=-90.0, le=90.0)
    longitude: float = PydanticField(ge=-180.0, le=180.0)
    elevation_ft: float = 0.0
    crop: Optional[Crop] = None
    soil: Optional[SoilSample] = None


class SatelliteReading(BaseModel):
    """Satellite imagery bands for NDVI computation."""

    date: date
    red_band: float = PydanticField(ge=0.0, le=1.0, description="Red reflectance")
    nir_band: float = PydanticField(
        ge=0.0, le=1.0, description="Near-infrared reflectance"
    )
    field_id: str


class YieldPrediction(BaseModel):
    """Predicted yield for a field."""

    field_id: str
    crop_type: CropType
    predicted_yield: float = PydanticField(description="Predicted yield per acre")
    confidence_low: float = PydanticField(description="Lower bound (90% CI)")
    confidence_high: float = PydanticField(description="Upper bound (90% CI)")
    unit: str = "bu/acre"
    weather_score: float = PydanticField(
        ge=0.0, le=1.0, description="Weather favorability 0-1"
    )
    soil_score: float = PydanticField(
        ge=0.0, le=1.0, description="Soil health score 0-1"
    )
    ndvi_score: float = PydanticField(
        ge=0.0, le=1.0, description="Vegetation vigor 0-1"
    )


class FertilizerRecommendation(BaseModel):
    """Fertilizer application recommendation."""

    field_id: str
    nitrogen_lbs_per_acre: float = 0.0
    phosphorus_lbs_per_acre: float = 0.0
    potassium_lbs_per_acre: float = 0.0
    lime_tons_per_acre: float = 0.0
    notes: list[str] = PydanticField(default_factory=list)


class PlantingRecommendation(BaseModel):
    """Optimal planting date recommendation."""

    crop_type: CropType
    optimal_date: date
    window_start: date
    window_end: date
    soil_temp_ready: bool
    frost_risk: float = PydanticField(
        ge=0.0, le=1.0, description="Frost risk probability"
    )
    notes: list[str] = PydanticField(default_factory=list)
