"""NDVI tracking from satellite imagery bands."""

from __future__ import annotations

import numpy as np

from yieldcast.models import SatelliteReading


class NDVITracker:
    """Compute and analyze Normalized Difference Vegetation Index from satellite data.

    NDVI = (NIR - Red) / (NIR + Red)
    Range: -1 to 1, with healthy vegetation typically 0.3-0.8.
    """

    HEALTHY_NDVI_MIN = 0.3
    HEALTHY_NDVI_MAX = 0.8
    PEAK_NDVI = 0.75

    def compute_ndvi(self, reading: SatelliteReading) -> float:
        """Compute NDVI from a single satellite reading."""
        denominator = reading.nir_band + reading.red_band
        if denominator < 1e-6:
            return 0.0
        return (reading.nir_band - reading.red_band) / denominator

    def compute_ndvi_series(
        self, readings: list[SatelliteReading]
    ) -> list[float]:
        """Compute NDVI time series from multiple readings."""
        return [self.compute_ndvi(r) for r in readings]

    def peak_ndvi(self, readings: list[SatelliteReading]) -> float:
        """Maximum NDVI observed during the season."""
        if not readings:
            return 0.0
        series = self.compute_ndvi_series(readings)
        return max(series)

    def mean_ndvi(self, readings: list[SatelliteReading]) -> float:
        """Average NDVI across the season."""
        if not readings:
            return 0.0
        series = self.compute_ndvi_series(readings)
        return float(np.mean(series))

    def ndvi_trend(self, readings: list[SatelliteReading]) -> float:
        """Linear trend slope of NDVI over the season.

        Positive values indicate improving vegetation; negative indicates decline.
        """
        if len(readings) < 2:
            return 0.0
        series = np.array(self.compute_ndvi_series(readings))
        x = np.arange(len(series), dtype=float)
        # Linear regression slope
        slope = np.polyfit(x, series, 1)[0]
        return float(slope)

    def green_fraction(self, readings: list[SatelliteReading]) -> float:
        """Fraction of readings with NDVI in the healthy range."""
        if not readings:
            return 0.0
        series = self.compute_ndvi_series(readings)
        count = sum(
            1 for v in series if self.HEALTHY_NDVI_MIN <= v <= self.HEALTHY_NDVI_MAX
        )
        return count / len(series)

    def vigor_score(self, readings: list[SatelliteReading]) -> float:
        """Overall vegetation vigor score (0-1).

        Combines peak NDVI, mean NDVI, and green fraction.
        """
        if not readings:
            return 0.0
        peak = self.peak_ndvi(readings)
        mean = self.mean_ndvi(readings)
        green_frac = self.green_fraction(readings)

        peak_score = min(peak / self.PEAK_NDVI, 1.0)
        mean_score = min(mean / 0.5, 1.0)

        return round(0.4 * peak_score + 0.3 * mean_score + 0.3 * green_frac, 4)

    def extract_features(
        self, readings: list[SatelliteReading]
    ) -> dict[str, float]:
        """Extract all satellite/NDVI features as a flat dictionary."""
        return {
            "ndvi_peak": self.peak_ndvi(readings),
            "ndvi_mean": self.mean_ndvi(readings),
            "ndvi_trend": self.ndvi_trend(readings),
            "ndvi_green_frac": self.green_fraction(readings),
            "ndvi_vigor": self.vigor_score(readings),
        }
