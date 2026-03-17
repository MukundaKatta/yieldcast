"""Rich terminal report formatting for YIELDCAST."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from yieldcast.models import (
    FertilizerRecommendation,
    Field,
    PlantingRecommendation,
    YieldPrediction,
)


def render_yield_report(
    predictions: list[tuple[Field, YieldPrediction]],
    console: Console | None = None,
) -> None:
    """Render a formatted yield prediction report to the terminal."""
    if console is None:
        console = Console()

    table = Table(
        title="YIELDCAST Yield Predictions",
        show_lines=True,
        title_style="bold cyan",
    )
    table.add_column("Field", style="bold")
    table.add_column("Crop", style="green")
    table.add_column("Area (ac)")
    table.add_column("Yield/ac", justify="right", style="bold yellow")
    table.add_column("90% CI", justify="right")
    table.add_column("Unit")
    table.add_column("Weather", justify="center")
    table.add_column("Soil", justify="center")
    table.add_column("NDVI", justify="center")

    for field, pred in predictions:
        ci = f"{pred.confidence_low:.1f} - {pred.confidence_high:.1f}"
        table.add_row(
            field.name,
            pred.crop_type.value.replace("_", " ").title(),
            f"{field.area_acres:.0f}",
            f"{pred.predicted_yield:.1f}",
            ci,
            pred.unit,
            _score_bar(pred.weather_score),
            _score_bar(pred.soil_score),
            _score_bar(pred.ndvi_score),
        )

    console.print()
    console.print(table)
    console.print()

    # Summary stats
    total_fields = len(predictions)
    if total_fields > 0:
        avg_weather = sum(p.weather_score for _, p in predictions) / total_fields
        avg_soil = sum(p.soil_score for _, p in predictions) / total_fields
        avg_ndvi = sum(p.ndvi_score for _, p in predictions) / total_fields
        console.print(
            Panel(
                f"Fields analyzed: {total_fields}  |  "
                f"Avg weather score: {avg_weather:.2f}  |  "
                f"Avg soil score: {avg_soil:.2f}  |  "
                f"Avg NDVI score: {avg_ndvi:.2f}",
                title="Summary",
                style="dim",
            )
        )


def render_fertilizer_report(
    recommendations: list[tuple[Field, FertilizerRecommendation]],
    console: Console | None = None,
) -> None:
    """Render fertilizer recommendations as a rich table."""
    if console is None:
        console = Console()

    table = Table(
        title="Fertilizer Recommendations",
        show_lines=True,
        title_style="bold green",
    )
    table.add_column("Field", style="bold")
    table.add_column("N (lbs/ac)", justify="right")
    table.add_column("P (lbs/ac)", justify="right")
    table.add_column("K (lbs/ac)", justify="right")
    table.add_column("Lime (tons/ac)", justify="right")
    table.add_column("Notes", max_width=50)

    for field, rec in recommendations:
        table.add_row(
            field.name,
            f"{rec.nitrogen_lbs_per_acre:.1f}",
            f"{rec.phosphorus_lbs_per_acre:.1f}",
            f"{rec.potassium_lbs_per_acre:.1f}",
            f"{rec.lime_tons_per_acre:.1f}",
            "; ".join(rec.notes[:2]),
        )

    console.print()
    console.print(table)


def render_planting_report(
    recommendations: list[PlantingRecommendation],
    console: Console | None = None,
) -> None:
    """Render planting date recommendations."""
    if console is None:
        console = Console()

    table = Table(
        title="Planting Date Recommendations",
        show_lines=True,
        title_style="bold magenta",
    )
    table.add_column("Crop", style="bold")
    table.add_column("Optimal Date", style="green")
    table.add_column("Window")
    table.add_column("Soil Ready", justify="center")
    table.add_column("Frost Risk", justify="center")
    table.add_column("Notes", max_width=50)

    for rec in recommendations:
        frost_style = "red" if rec.frost_risk > 0.2 else "green"
        table.add_row(
            rec.crop_type.value.replace("_", " ").title(),
            str(rec.optimal_date),
            f"{rec.window_start} to {rec.window_end}",
            "[green]Yes[/green]" if rec.soil_temp_ready else "[red]No[/red]",
            f"[{frost_style}]{rec.frost_risk:.0%}[/{frost_style}]",
            "; ".join(rec.notes[:2]),
        )

    console.print()
    console.print(table)


def _score_bar(score: float) -> str:
    """Render a score as a colored text indicator."""
    pct = int(score * 100)
    if score >= 0.7:
        return f"[green]{pct}%[/green]"
    elif score >= 0.4:
        return f"[yellow]{pct}%[/yellow]"
    else:
        return f"[red]{pct}%[/red]"
