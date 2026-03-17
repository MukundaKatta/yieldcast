"""Tests for soil analysis."""

import pytest

from yieldcast.models import SoilSample
from yieldcast.predictor.soil import SoilAnalyzer, _range_score


def _good_soil() -> SoilSample:
    return SoilSample(
        ph=6.5, organic_matter_pct=4.5, nitrogen_ppm=45.0,
        phosphorus_ppm=35.0, potassium_ppm=200.0, moisture_pct=28.0,
        cec=18.0, sand_pct=40.0, silt_pct=40.0, clay_pct=20.0,
    )


def _poor_soil() -> SoilSample:
    return SoilSample(
        ph=4.5, organic_matter_pct=1.0, nitrogen_ppm=5.0,
        phosphorus_ppm=5.0, potassium_ppm=50.0, moisture_pct=8.0,
        cec=5.0, sand_pct=80.0, silt_pct=10.0, clay_pct=10.0,
    )


class TestRangeScore:
    def test_within_range(self):
        assert _range_score(6.5, 6.0, 7.0) == 1.0

    def test_below_range(self):
        score = _range_score(4.0, 6.0, 7.0)
        assert 0.0 < score < 1.0

    def test_above_range(self):
        score = _range_score(9.0, 6.0, 7.0)
        assert 0.0 <= score < 1.0

    def test_at_boundary(self):
        assert _range_score(6.0, 6.0, 7.0) == 1.0
        assert _range_score(7.0, 6.0, 7.0) == 1.0


class TestSoilAnalyzer:
    def setup_method(self):
        self.analyzer = SoilAnalyzer()

    def test_good_soil_high_scores(self):
        soil = _good_soil()
        assert self.analyzer.ph_score(soil) == 1.0
        assert self.analyzer.overall_score(soil) > 0.7

    def test_poor_soil_low_scores(self):
        soil = _poor_soil()
        assert self.analyzer.ph_score(soil) < 0.5
        assert self.analyzer.overall_score(soil) < 0.4

    def test_overall_score_bounded(self):
        for soil in [_good_soil(), _poor_soil()]:
            score = self.analyzer.overall_score(soil)
            assert 0.0 <= score <= 1.0

    def test_extract_features_keys(self):
        features = self.analyzer.extract_features(_good_soil())
        assert "soil_ph" in features
        assert "soil_overall" in features
        assert "soil_n_score" in features
        assert len(features) == 12

    def test_texture_score_loam_is_best(self):
        loam = SoilSample(
            ph=6.5, organic_matter_pct=4.0, nitrogen_ppm=40.0,
            phosphorus_ppm=30.0, potassium_ppm=200.0, moisture_pct=25.0,
            cec=15.0, sand_pct=40.0, silt_pct=40.0, clay_pct=20.0,
        )
        sandy = SoilSample(
            ph=6.5, organic_matter_pct=4.0, nitrogen_ppm=40.0,
            phosphorus_ppm=30.0, potassium_ppm=200.0, moisture_pct=25.0,
            cec=15.0, sand_pct=85.0, silt_pct=10.0, clay_pct=5.0,
        )
        assert self.analyzer.texture_score(loam) > self.analyzer.texture_score(sandy)
