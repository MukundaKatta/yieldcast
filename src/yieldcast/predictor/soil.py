"""Soil analysis for yield prediction."""

from __future__ import annotations

import numpy as np

from yieldcast.models import CropType, SoilSample


# Optimal ranges per nutrient for major grain crops
_OPTIMAL_PH = (6.0, 7.0)
_OPTIMAL_OM = (3.0, 6.0)  # organic matter %
_OPTIMAL_N = (20.0, 60.0)  # ppm
_OPTIMAL_P = (25.0, 50.0)  # ppm
_OPTIMAL_K = (150.0, 250.0)  # ppm
_OPTIMAL_MOISTURE = (20.0, 35.0)  # %
_OPTIMAL_CEC = (10.0, 25.0)  # meq/100g


def _range_score(value: float, low: float, high: float) -> float:
    """Score how well a value falls within an optimal range.

    Returns 1.0 if within range, decaying linearly outside.
    """
    if low <= value <= high:
        return 1.0
    if value < low:
        return max(0.0, 1.0 - (low - value) / low) if low > 0 else 0.0
    # value > high
    overshoot = value - high
    return max(0.0, 1.0 - overshoot / high) if high > 0 else 0.0


class SoilAnalyzer:
    """Analyze soil samples and produce scores for yield prediction."""

    def ph_score(self, sample: SoilSample) -> float:
        """Score pH suitability (0-1). Optimal range 6.0-7.0."""
        return _range_score(sample.ph, *_OPTIMAL_PH)

    def organic_matter_score(self, sample: SoilSample) -> float:
        """Score organic matter content (0-1)."""
        return _range_score(sample.organic_matter_pct, *_OPTIMAL_OM)

    def nitrogen_score(self, sample: SoilSample) -> float:
        """Score available nitrogen level."""
        return _range_score(sample.nitrogen_ppm, *_OPTIMAL_N)

    def phosphorus_score(self, sample: SoilSample) -> float:
        """Score available phosphorus level."""
        return _range_score(sample.phosphorus_ppm, *_OPTIMAL_P)

    def potassium_score(self, sample: SoilSample) -> float:
        """Score available potassium level."""
        return _range_score(sample.potassium_ppm, *_OPTIMAL_K)

    def moisture_score(self, sample: SoilSample) -> float:
        """Score soil moisture level."""
        return _range_score(sample.moisture_pct, *_OPTIMAL_MOISTURE)

    def cec_score(self, sample: SoilSample) -> float:
        """Score cation exchange capacity."""
        return _range_score(sample.cec, *_OPTIMAL_CEC)

    def texture_score(self, sample: SoilSample) -> float:
        """Score soil texture. Loamy soils (balanced sand/silt/clay) score highest."""
        # Ideal loam: ~40% sand, 40% silt, 20% clay
        ideal = np.array([40.0, 40.0, 20.0])
        actual = np.array([sample.sand_pct, sample.silt_pct, sample.clay_pct])
        distance = np.linalg.norm(actual - ideal)
        # Max possible distance is roughly 100
        return float(max(0.0, 1.0 - distance / 80.0))

    def overall_score(self, sample: SoilSample) -> float:
        """Weighted composite soil health score (0-1)."""
        weights = {
            "ph": 0.15,
            "om": 0.10,
            "n": 0.20,
            "p": 0.15,
            "k": 0.15,
            "moisture": 0.10,
            "cec": 0.05,
            "texture": 0.10,
        }
        scores = {
            "ph": self.ph_score(sample),
            "om": self.organic_matter_score(sample),
            "n": self.nitrogen_score(sample),
            "p": self.phosphorus_score(sample),
            "k": self.potassium_score(sample),
            "moisture": self.moisture_score(sample),
            "cec": self.cec_score(sample),
            "texture": self.texture_score(sample),
        }
        total = sum(scores[k] * weights[k] for k in weights)
        return round(total, 4)

    def extract_features(self, sample: SoilSample) -> dict[str, float]:
        """Extract all soil features as a flat dictionary."""
        return {
            "soil_ph": sample.ph,
            "soil_ph_score": self.ph_score(sample),
            "soil_om_pct": sample.organic_matter_pct,
            "soil_n_ppm": sample.nitrogen_ppm,
            "soil_p_ppm": sample.phosphorus_ppm,
            "soil_k_ppm": sample.potassium_ppm,
            "soil_moisture_pct": sample.moisture_pct,
            "soil_cec": sample.cec,
            "soil_n_score": self.nitrogen_score(sample),
            "soil_p_score": self.phosphorus_score(sample),
            "soil_k_score": self.potassium_score(sample),
            "soil_overall": self.overall_score(sample),
        }
