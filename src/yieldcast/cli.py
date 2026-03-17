"""CLI entry point for YIELDCAST."""

from __future__ import annotations

import click
from rich.console import Console

from yieldcast.models import CropType
from yieldcast.optimizer.fertilizer import FertilizerRecommender
from yieldcast.optimizer.planting import PlantingOptimizer
from yieldcast.predictor.yield_model import YieldPredictor
from yieldcast.report import (
    render_fertilizer_report,
    render_planting_report,
    render_yield_report,
)
from yieldcast.simulator import FarmSimulator

console = Console()


@click.group()
@click.version_option(package_name="yieldcast")
def main() -> None:
    """YIELDCAST -- AI Crop Yield Predictor."""


@main.command()
@click.option("--fields", default=5, help="Number of fields to simulate.")
@click.option("--seasons", default=1, help="Number of growing seasons.")
@click.option("--seed", default=42, help="Random seed for reproducibility.")
def simulate(fields: int, seasons: int, seed: int) -> None:
    """Run a full simulation with synthetic farm data."""
    sim = FarmSimulator(seed=seed)
    predictor = YieldPredictor()
    fert = FertilizerRecommender()

    for year_offset in range(seasons):
        year = 2025 + year_offset
        console.print(f"\n[bold cyan]=== Season {year} ===[/bold cyan]")

        farm = sim.generate_farm(n_fields=fields, year=year)
        predictions = []
        fert_recs = []

        for field, weather, satellite in farm:
            pred = predictor.predict_yield(field, weather, satellite)
            predictions.append((field, pred))

            if field.crop and field.soil:
                rec = fert.recommend(field.soil, field.crop.crop_type, field.field_id)
                fert_recs.append((field, rec))

        render_yield_report(predictions, console)
        render_fertilizer_report(fert_recs, console)


@main.command()
@click.option("--crop", type=click.Choice([c.value for c in CropType]), default="corn")
@click.option("--area", default=100.0, help="Field area in acres.")
@click.option("--seed", default=42, help="Random seed.")
def predict(crop: str, area: float, seed: int) -> None:
    """Predict yield for a single field."""
    crop_type = CropType(crop)
    sim = FarmSimulator(seed=seed)
    predictor = YieldPredictor()

    field, weather, satellite = sim.generate_field(
        field_id="CMD01",
        crop_type=crop_type,
        year=2025,
        soil_quality="medium",
    )
    # Override area
    field.area_acres = area

    pred = predictor.predict_yield(field, weather, satellite)
    render_yield_report([(field, pred)], console)

    total = pred.predicted_yield * area
    console.print(
        f"[bold]Total estimated production: {total:,.0f} {pred.unit.split('/')[0]}[/bold]"
    )


@main.command()
@click.option("--fields", default=10, help="Number of fields in report.")
@click.option("--seed", default=42, help="Random seed.")
def report(fields: int, seed: int) -> None:
    """Generate a comprehensive farm report."""
    sim = FarmSimulator(seed=seed)
    predictor = YieldPredictor()
    fert = FertilizerRecommender()
    planting = PlantingOptimizer()

    farm = sim.generate_farm(n_fields=fields, year=2025)

    predictions = []
    fert_recs = []

    for field, weather, satellite in farm:
        pred = predictor.predict_yield(field, weather, satellite)
        predictions.append((field, pred))

        if field.crop and field.soil:
            rec = fert.recommend(field.soil, field.crop.crop_type, field.field_id)
            fert_recs.append((field, rec))

    render_yield_report(predictions, console)
    render_fertilizer_report(fert_recs, console)

    # Planting recommendations for all crop types
    weather_history = farm[0][1] if farm else []
    planting_recs = []
    for crop_type in list(CropType)[:6]:
        rec = planting.recommend(crop_type, 2025, weather_history)
        planting_recs.append(rec)

    render_planting_report(planting_recs, console)


if __name__ == "__main__":
    main()
