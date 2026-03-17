"""Weather feature extraction for yield prediction."""

from __future__ import annotations

import numpy as np

from yieldcast.models import CropType, CROP_GDD_REQUIREMENTS, WeatherData


class WeatherFeatureExtractor:
    """Extract agronomic weather features from daily observations.

    Computes growing degree days, precipitation accumulations,
    drought indices, and stress day counts.
    """

    BASE_TEMP_F = 50.0  # GDD base temperature
    DROUGHT_THRESHOLD_DAYS = 7  # consecutive dry days for drought flag
    HEAT_STRESS_THRESHOLD_F = 95.0
    COLD_STRESS_THRESHOLD_F = 32.0

    def compute_gdd(self, weather: list[WeatherData]) -> float:
        """Compute accumulated growing degree days (base 50F).

        GDD = sum of max(0, (Tmax + Tmin) / 2 - Tbase) for each day.
        """
        total = 0.0
        for day in weather:
            avg = (day.temp_max_f + day.temp_min_f) / 2.0
            total += max(0.0, avg - self.BASE_TEMP_F)
        return total

    def total_precipitation(self, weather: list[WeatherData]) -> float:
        """Total precipitation across the season (inches)."""
        return sum(day.precipitation_in for day in weather)

    def precipitation_distribution_score(
        self, weather: list[WeatherData]
    ) -> float:
        """Score how evenly distributed rainfall is (0=clustered, 1=even).

        Splits the season into weekly buckets and measures uniformity
        using coefficient of variation.
        """
        if not weather:
            return 0.0
        week_size = 7
        n_weeks = max(1, len(weather) // week_size)
        weekly_precip = []
        for i in range(n_weeks):
            chunk = weather[i * week_size : (i + 1) * week_size]
            weekly_precip.append(sum(d.precipitation_in for d in chunk))

        arr = np.array(weekly_precip)
        mean = arr.mean()
        if mean < 0.01:
            return 0.0
        cv = arr.std() / mean
        # Invert: lower CV = better distribution. Cap at 2.0 for normalization.
        return float(max(0.0, 1.0 - cv / 2.0))

    def drought_index(self, weather: list[WeatherData]) -> float:
        """Compute a drought stress index (0=no drought, 1=severe).

        Measures the fraction of the season spent in drought conditions,
        defined as consecutive dry-day streaks >= DROUGHT_THRESHOLD_DAYS.
        """
        if not weather:
            return 0.0

        drought_days = 0
        streak = 0
        for day in weather:
            if day.precipitation_in < 0.01:
                streak += 1
            else:
                if streak >= self.DROUGHT_THRESHOLD_DAYS:
                    drought_days += streak
                streak = 0
        if streak >= self.DROUGHT_THRESHOLD_DAYS:
            drought_days += streak

        return min(1.0, drought_days / max(1, len(weather)))

    def heat_stress_days(self, weather: list[WeatherData]) -> int:
        """Count days where max temperature exceeds heat stress threshold."""
        return sum(
            1 for d in weather if d.temp_max_f >= self.HEAT_STRESS_THRESHOLD_F
        )

    def cold_stress_days(self, weather: list[WeatherData]) -> int:
        """Count days where min temperature drops below freezing."""
        return sum(
            1 for d in weather if d.temp_min_f <= self.COLD_STRESS_THRESHOLD_F
        )

    def avg_solar_radiation(self, weather: list[WeatherData]) -> float:
        """Average daily solar radiation (MJ/m2)."""
        if not weather:
            return 0.0
        return sum(d.solar_radiation_mj for d in weather) / len(weather)

    def gdd_fulfillment_ratio(
        self, weather: list[WeatherData], crop: CropType
    ) -> float:
        """Ratio of actual GDD to the crop's optimal GDD midpoint.

        Values near 1.0 indicate the season warmth matched crop needs.
        """
        gdd = self.compute_gdd(weather)
        low, high = CROP_GDD_REQUIREMENTS[crop]
        midpoint = (low + high) / 2.0
        return min(gdd / midpoint, 1.5) if midpoint > 0 else 0.0

    def extract_features(
        self, weather: list[WeatherData], crop: CropType
    ) -> dict[str, float]:
        """Extract all weather features as a flat dictionary."""
        n_days = len(weather)
        return {
            "gdd_total": self.compute_gdd(weather),
            "gdd_fulfillment": self.gdd_fulfillment_ratio(weather, crop),
            "precip_total_in": self.total_precipitation(weather),
            "precip_distribution": self.precipitation_distribution_score(weather),
            "drought_index": self.drought_index(weather),
            "heat_stress_days": float(self.heat_stress_days(weather)),
            "cold_stress_days": float(self.cold_stress_days(weather)),
            "heat_stress_frac": self.heat_stress_days(weather) / max(1, n_days),
            "avg_solar_mj": self.avg_solar_radiation(weather),
        }
