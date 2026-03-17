"""Tests for NDVI satellite tracking."""

from datetime import date, timedelta

import pytest

from yieldcast.models import SatelliteReading
from yieldcast.predictor.satellite import NDVITracker


def _make_reading(red: float, nir: float, day_offset: int = 0) -> SatelliteReading:
    return SatelliteReading(
        date=date(2025, 5, 1) + timedelta(days=day_offset),
        red_band=red,
        nir_band=nir,
        field_id="F001",
    )


class TestNDVITracker:
    def setup_method(self):
        self.tracker = NDVITracker()

    def test_ndvi_healthy_vegetation(self):
        # High NIR, low red = healthy
        reading = _make_reading(red=0.1, nir=0.6)
        ndvi = self.tracker.compute_ndvi(reading)
        assert ndvi == pytest.approx(0.7143, abs=0.01)

    def test_ndvi_bare_soil(self):
        # Similar red and NIR = bare soil
        reading = _make_reading(red=0.3, nir=0.35)
        ndvi = self.tracker.compute_ndvi(reading)
        assert ndvi < 0.15

    def test_ndvi_zero_bands(self):
        reading = _make_reading(red=0.0, nir=0.0)
        assert self.tracker.compute_ndvi(reading) == 0.0

    def test_peak_ndvi(self):
        readings = [
            _make_reading(0.2, 0.5, 0),
            _make_reading(0.1, 0.7, 15),
            _make_reading(0.15, 0.55, 30),
        ]
        peak = self.tracker.peak_ndvi(readings)
        assert peak == pytest.approx(0.75, abs=0.01)

    def test_mean_ndvi(self):
        readings = [_make_reading(0.1, 0.6, i * 15) for i in range(5)]
        mean = self.tracker.mean_ndvi(readings)
        assert 0.0 < mean < 1.0

    def test_ndvi_trend_increasing(self):
        # Gradually improving vegetation
        readings = [
            _make_reading(0.3, 0.4, 0),
            _make_reading(0.2, 0.5, 15),
            _make_reading(0.1, 0.6, 30),
            _make_reading(0.08, 0.7, 45),
        ]
        trend = self.tracker.ndvi_trend(readings)
        assert trend > 0

    def test_vigor_score_bounded(self):
        readings = [_make_reading(0.1, 0.6, i * 15) for i in range(8)]
        score = self.tracker.vigor_score(readings)
        assert 0.0 <= score <= 1.0

    def test_empty_readings(self):
        assert self.tracker.peak_ndvi([]) == 0.0
        assert self.tracker.mean_ndvi([]) == 0.0
        assert self.tracker.vigor_score([]) == 0.0

    def test_extract_features_keys(self):
        readings = [_make_reading(0.1, 0.6, i * 15) for i in range(6)]
        features = self.tracker.extract_features(readings)
        assert "ndvi_peak" in features
        assert "ndvi_vigor" in features
        assert len(features) == 5
