"""Optimal planting date recommendation engine."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from yieldcast.models import (
    CROP_GDD_REQUIREMENTS,
    CROP_PLANTING_WINDOWS,
    CropType,
    PlantingRecommendation,
    WeatherData,
)


class PlantingOptimizer:
    """Recommend optimal planting dates based on weather history and crop requirements.

    Analyzes historical weather to estimate soil temperature readiness,
    last frost dates, and accumulated GDD windows.
    """

    SOIL_READY_TEMP_F = 50.0  # minimum soil temp for most crops
    FROST_TEMP_F = 32.0

    def estimate_last_frost_date(
        self, weather_history: list[WeatherData], year: int
    ) -> date:
        """Estimate the last spring frost date from historical weather.

        Scans January-June for the latest date with min temp <= 32F.
        """
        spring_days = [
            d
            for d in weather_history
            if d.date.year == year and d.date.month <= 6
        ]
        spring_days.sort(key=lambda d: d.date)

        last_frost = date(year, 3, 15)  # default
        for day in spring_days:
            if day.temp_min_f <= self.FROST_TEMP_F:
                last_frost = day.date

        return last_frost

    def estimate_soil_temp_ready_date(
        self, weather_history: list[WeatherData], year: int
    ) -> date:
        """Estimate when soil temperature is warm enough for planting.

        Uses a 5-day rolling average of daily mean temps as a proxy
        for soil temperature at planting depth.
        """
        spring_days = sorted(
            [d for d in weather_history if d.date.year == year and 2 <= d.date.month <= 6],
            key=lambda d: d.date,
        )

        if len(spring_days) < 5:
            return date(year, 4, 15)  # default

        means = [(d.temp_max_f + d.temp_min_f) / 2.0 for d in spring_days]
        for i in range(4, len(means)):
            window_avg = np.mean(means[i - 4 : i + 1])
            if window_avg >= self.SOIL_READY_TEMP_F:
                return spring_days[i].date

        return date(year, 4, 15)

    def frost_risk_at_date(
        self, target_date: date, weather_history: list[WeatherData]
    ) -> float:
        """Estimate frost probability at a given date from historical data.

        Looks at the same calendar date (+/- 7 days) across available years.
        """
        doy = target_date.timetuple().tm_yday
        frost_count = 0
        total_count = 0
        for day in weather_history:
            day_doy = day.date.timetuple().tm_yday
            if abs(day_doy - doy) <= 7:
                total_count += 1
                if day.temp_min_f <= self.FROST_TEMP_F:
                    frost_count += 1

        if total_count == 0:
            return 0.1
        return round(frost_count / total_count, 3)

    def recommend(
        self,
        crop: CropType,
        year: int,
        weather_history: list[WeatherData],
        latitude: float = 40.0,
    ) -> PlantingRecommendation:
        """Generate optimal planting date recommendation.

        Combines crop planting window, frost risk, and soil temperature
        readiness to find the best planting date.
        """
        early_month, late_month = CROP_PLANTING_WINDOWS[crop]

        # For winter wheat, planting is in fall of the prior year
        planting_year = year if early_month <= 6 else year - 1

        window_start = date(planting_year, early_month, 1)
        window_end = date(planting_year, late_month, 28)

        last_frost = self.estimate_last_frost_date(weather_history, planting_year)
        soil_ready = self.estimate_soil_temp_ready_date(weather_history, planting_year)

        notes: list[str] = []

        if crop == CropType.WINTER_WHEAT:
            # Winter wheat: plant in fall, target 6 weeks before first fall frost
            optimal = date(planting_year, early_month, 20)
            soil_temp_ready = True
            frost_risk = 0.0
            notes.append("Winter wheat: target 6 weeks before first hard freeze")
        else:
            # Spring-planted: wait for soil warmth and frost safety
            safe_date = max(last_frost + timedelta(days=10), soil_ready)
            optimal = max(window_start, safe_date)
            optimal = min(optimal, window_end)

            soil_temp_ready = soil_ready <= optimal
            frost_risk = self.frost_risk_at_date(optimal, weather_history)

            if frost_risk > 0.2:
                notes.append(
                    f"Elevated frost risk ({frost_risk:.0%}); consider delaying 5-7 days"
                )
            if not soil_temp_ready:
                notes.append("Soil may not be warm enough; monitor soil temperature")

        gdd_low, gdd_high = CROP_GDD_REQUIREMENTS[crop]
        notes.append(
            f"Crop requires {gdd_low:.0f}-{gdd_high:.0f} GDD (base 50F) to mature"
        )

        return PlantingRecommendation(
            crop_type=crop,
            optimal_date=optimal,
            window_start=window_start,
            window_end=window_end,
            soil_temp_ready=soil_temp_ready,
            frost_risk=frost_risk,
            notes=notes,
        )
