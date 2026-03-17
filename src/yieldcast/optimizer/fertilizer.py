"""Fertilizer recommendation engine based on soil analysis."""

from __future__ import annotations

from yieldcast.models import CropType, FertilizerRecommendation, SoilSample


# Target nutrient levels by crop type (N ppm, P ppm, K ppm, ideal pH)
CROP_NUTRIENT_TARGETS: dict[CropType, dict[str, float]] = {
    CropType.CORN: {"n": 50.0, "p": 40.0, "k": 200.0, "ph": 6.5},
    CropType.SOYBEANS: {"n": 15.0, "p": 35.0, "k": 200.0, "ph": 6.5},
    CropType.WINTER_WHEAT: {"n": 45.0, "p": 30.0, "k": 180.0, "ph": 6.3},
    CropType.SPRING_WHEAT: {"n": 45.0, "p": 30.0, "k": 180.0, "ph": 6.3},
    CropType.RICE: {"n": 55.0, "p": 30.0, "k": 150.0, "ph": 6.0},
    CropType.COTTON: {"n": 50.0, "p": 35.0, "k": 200.0, "ph": 6.2},
    CropType.BARLEY: {"n": 40.0, "p": 30.0, "k": 170.0, "ph": 6.5},
    CropType.SORGHUM: {"n": 45.0, "p": 35.0, "k": 190.0, "ph": 6.3},
    CropType.OATS: {"n": 35.0, "p": 25.0, "k": 160.0, "ph": 6.2},
    CropType.CANOLA: {"n": 50.0, "p": 30.0, "k": 180.0, "ph": 6.5},
    CropType.SUNFLOWER: {"n": 40.0, "p": 35.0, "k": 200.0, "ph": 6.5},
    CropType.ALFALFA: {"n": 10.0, "p": 40.0, "k": 250.0, "ph": 6.8},
}

# Conversion factors: ppm deficit -> lbs/acre recommendation
N_PPM_TO_LBS = 3.5
P_PPM_TO_LBS = 2.0
K_PPM_TO_LBS = 1.5


class FertilizerRecommender:
    """Generate fertilizer recommendations based on soil analysis and crop needs.

    Compares current soil nutrient levels to crop-specific targets and
    calculates application rates to close the gap.
    """

    def recommend(
        self,
        soil: SoilSample,
        crop: CropType,
        field_id: str = "unknown",
    ) -> FertilizerRecommendation:
        """Produce fertilizer recommendation for a field.

        Args:
            soil: Current soil test results.
            crop: Target crop type.
            field_id: Identifier for the field.
        """
        targets = CROP_NUTRIENT_TARGETS[crop]
        notes: list[str] = []

        # Nitrogen
        n_deficit = max(0.0, targets["n"] - soil.nitrogen_ppm)
        n_lbs = round(n_deficit * N_PPM_TO_LBS, 1)
        if crop == CropType.SOYBEANS and soil.nitrogen_ppm >= 10:
            n_lbs = 0.0
            notes.append("Soybeans fix nitrogen; supplemental N not recommended")
        elif crop == CropType.ALFALFA and soil.nitrogen_ppm >= 5:
            n_lbs = 0.0
            notes.append("Alfalfa fixes nitrogen; supplemental N not needed")
        if n_lbs > 0:
            notes.append(f"Nitrogen deficit: {n_deficit:.0f} ppm below target")

        # Phosphorus
        p_deficit = max(0.0, targets["p"] - soil.phosphorus_ppm)
        p_lbs = round(p_deficit * P_PPM_TO_LBS, 1)
        if p_lbs > 0:
            notes.append(f"Phosphorus deficit: {p_deficit:.0f} ppm below target")

        # Potassium
        k_deficit = max(0.0, targets["k"] - soil.potassium_ppm)
        k_lbs = round(k_deficit * K_PPM_TO_LBS, 1)
        if k_lbs > 0:
            notes.append(f"Potassium deficit: {k_deficit:.0f} ppm below target")

        # Lime recommendation for pH correction
        lime_tons = 0.0
        if soil.ph < targets["ph"] - 0.3:
            ph_gap = targets["ph"] - soil.ph
            # Rough rule: ~1 ton lime per 0.5 pH unit on typical soils
            lime_tons = round(ph_gap / 0.5, 1)
            notes.append(
                f"Soil pH {soil.ph:.1f} is below target {targets['ph']:.1f}; "
                f"lime recommended"
            )
        elif soil.ph > 7.5:
            notes.append(
                f"Soil pH {soil.ph:.1f} is high; consider sulfur amendment"
            )

        # Organic matter advisory
        if soil.organic_matter_pct < 2.0:
            notes.append(
                "Low organic matter; consider cover crops or compost application"
            )

        if not notes:
            notes.append("Soil nutrient levels are adequate for this crop")

        return FertilizerRecommendation(
            field_id=field_id,
            nitrogen_lbs_per_acre=n_lbs,
            phosphorus_lbs_per_acre=p_lbs,
            potassium_lbs_per_acre=k_lbs,
            lime_tons_per_acre=lime_tons,
            notes=notes,
        )
