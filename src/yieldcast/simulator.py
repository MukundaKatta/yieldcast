"""Synthetic farm data generator for testing and demonstration."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np

from yieldcast.models import (
    CROP_PLANTING_WINDOWS,
    CROP_YIELD_RANGES,
    Crop,
    CropType,
    Field,
    SatelliteReading,
    SoilSample,
    WeatherData,
)


class FarmSimulator:
    """Generate realistic synthetic farm data for demonstrations and testing.

    Produces correlated weather, soil, and satellite data that approximate
    real growing conditions for the US Corn Belt region.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_soil(self, quality: str = "medium") -> SoilSample:
        """Generate a synthetic soil sample.

        Args:
            quality: One of 'poor', 'medium', 'good'. Controls the
                     distribution of nutrient values.
        """
        profiles = {
            "poor": {"ph_mu": 5.2, "n_mu": 12.0, "p_mu": 15.0, "k_mu": 100.0, "om_mu": 1.5},
            "medium": {"ph_mu": 6.3, "n_mu": 35.0, "p_mu": 30.0, "k_mu": 180.0, "om_mu": 3.5},
            "good": {"ph_mu": 6.6, "n_mu": 50.0, "p_mu": 42.0, "k_mu": 220.0, "om_mu": 5.0},
        }
        p = profiles.get(quality, profiles["medium"])

        sand = self.rng.uniform(25, 55)
        clay = self.rng.uniform(10, 35)
        silt = 100.0 - sand - clay
        if silt < 0:
            silt = 0.0
            clay = 100.0 - sand

        return SoilSample(
            ph=round(float(self.rng.normal(p["ph_mu"], 0.4)), 2),
            organic_matter_pct=round(float(max(0.5, self.rng.normal(p["om_mu"], 0.8))), 2),
            nitrogen_ppm=round(float(max(1.0, self.rng.normal(p["n_mu"], 8.0))), 1),
            phosphorus_ppm=round(float(max(1.0, self.rng.normal(p["p_mu"], 8.0))), 1),
            potassium_ppm=round(float(max(10.0, self.rng.normal(p["k_mu"], 30.0))), 1),
            moisture_pct=round(float(self.rng.uniform(15.0, 40.0)), 1),
            cec=round(float(max(3.0, self.rng.normal(15.0, 5.0))), 1),
            sand_pct=round(sand, 1),
            silt_pct=round(silt, 1),
            clay_pct=round(clay, 1),
        )

    def generate_weather_season(
        self,
        year: int,
        latitude: float = 40.0,
        start_month: int = 3,
        end_month: int = 10,
    ) -> list[WeatherData]:
        """Generate a full growing season of daily weather data.

        Produces realistic temperature curves, stochastic precipitation,
        and correlated humidity/solar radiation.
        """
        start = date(year, start_month, 1)
        end = date(year, end_month, 28)
        n_days = (end - start).days + 1
        days: list[WeatherData] = []

        for i in range(n_days):
            current = start + timedelta(days=i)
            # Seasonal temperature curve (peaks in July)
            doy = current.timetuple().tm_yday
            season_factor = np.sin(np.pi * (doy - 80) / 200)
            base_high = 55 + 35 * season_factor
            base_low = base_high - self.rng.uniform(15, 25)

            temp_high = float(base_high + self.rng.normal(0, 5))
            temp_low = float(base_low + self.rng.normal(0, 4))
            temp_low = min(temp_low, temp_high - 5)

            # Precipitation: 30% chance of rain, gamma-distributed amounts
            has_rain = self.rng.random() < 0.30
            precip = 0.0
            if has_rain:
                precip = float(self.rng.gamma(1.5, 0.3))

            humidity = float(np.clip(self.rng.normal(60 + precip * 10, 12), 20, 100))
            solar = float(np.clip(
                self.rng.normal(15 + 10 * season_factor - precip * 3, 3), 2, 32
            ))
            wind = float(max(0, self.rng.normal(8, 4)))

            days.append(
                WeatherData(
                    date=current,
                    temp_max_f=round(temp_high, 1),
                    temp_min_f=round(temp_low, 1),
                    precipitation_in=round(precip, 2),
                    humidity_pct=round(humidity, 1),
                    solar_radiation_mj=round(solar, 1),
                    wind_speed_mph=round(wind, 1),
                )
            )
        return days

    def generate_satellite_readings(
        self,
        field_id: str,
        year: int,
        n_readings: int = 12,
        crop_health: float = 0.7,
    ) -> list[SatelliteReading]:
        """Generate synthetic satellite NDVI readings across a season.

        Simulates a vegetation growth curve that peaks mid-season.
        """
        readings = []
        start = date(year, 4, 1)
        interval = 15  # roughly every 2 weeks

        for i in range(n_readings):
            current = start + timedelta(days=i * interval)
            # NDVI growth curve: rises to peak then declines
            progress = i / max(1, n_readings - 1)
            # Bell curve peaking at 60% through season
            ndvi_base = crop_health * np.exp(-((progress - 0.6) ** 2) / 0.08)
            ndvi_base = max(0.1, ndvi_base)

            noise = float(self.rng.normal(0, 0.03))
            ndvi = float(np.clip(ndvi_base + noise, 0.05, 0.95))

            # Convert NDVI back to red/NIR bands: NDVI = (NIR-R)/(NIR+R)
            # Choose NIR, solve for R: R = NIR*(1-NDVI)/(1+NDVI)
            nir = float(np.clip(0.4 + 0.5 * ndvi + self.rng.normal(0, 0.02), 0.1, 0.95))
            red = float(np.clip(nir * (1 - ndvi) / (1 + ndvi), 0.01, 0.95))

            readings.append(
                SatelliteReading(
                    date=current,
                    red_band=round(red, 4),
                    nir_band=round(nir, 4),
                    field_id=field_id,
                )
            )
        return readings

    def generate_field(
        self,
        field_id: str,
        crop_type: Optional[CropType] = None,
        year: int = 2025,
        soil_quality: str = "medium",
    ) -> tuple[Field, list[WeatherData], list[SatelliteReading]]:
        """Generate a complete field with all associated data.

        Returns:
            Tuple of (Field, weather_data, satellite_readings).
        """
        if crop_type is None:
            crop_type = self.rng.choice(list(CropType))

        early_month, _ = CROP_PLANTING_WINDOWS[crop_type]
        planting_date = date(year, early_month, self.rng.integers(5, 25))

        lat = float(self.rng.uniform(35.0, 45.0))
        lon = float(self.rng.uniform(-100.0, -85.0))
        area = float(self.rng.uniform(40.0, 320.0))

        soil = self.generate_soil(soil_quality)
        weather = self.generate_weather_season(year, lat)
        crop_health = {"poor": 0.4, "medium": 0.65, "good": 0.85}.get(
            soil_quality, 0.65
        )
        satellite = self.generate_satellite_readings(
            field_id, year, crop_health=crop_health
        )

        field = Field(
            field_id=field_id,
            name=f"Field {field_id}",
            area_acres=round(area, 1),
            latitude=round(lat, 4),
            longitude=round(lon, 4),
            elevation_ft=float(self.rng.uniform(500, 1200)),
            crop=Crop(
                crop_type=crop_type,
                planting_date=planting_date,
            ),
            soil=soil,
        )

        return field, weather, satellite

    def generate_farm(
        self,
        n_fields: int = 5,
        year: int = 2025,
    ) -> list[tuple[Field, list[WeatherData], list[SatelliteReading]]]:
        """Generate a complete farm with multiple fields."""
        qualities = ["poor", "medium", "medium", "good", "good"]
        farm = []
        for i in range(n_fields):
            quality = qualities[i % len(qualities)]
            crop = list(CropType)[i % len(CropType)]
            field_data = self.generate_field(
                field_id=f"F{i + 1:03d}",
                crop_type=crop,
                year=year,
                soil_quality=quality,
            )
            farm.append(field_data)
        return farm
