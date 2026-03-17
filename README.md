# YIELDCAST

AI-powered crop yield predictor that combines weather patterns, soil analysis, and satellite imagery to forecast agricultural output and optimize farming decisions.

## Features

- **Yield Prediction** -- GradientBoosting model trained on weather, soil, and satellite features
- **Weather Analysis** -- Growing degree days (GDD), precipitation accumulation, and drought indices
- **Soil Profiling** -- pH balance, macro/micronutrient levels, and moisture content scoring
- **Satellite NDVI** -- Normalized Difference Vegetation Index tracking from imagery bands
- **Planting Optimization** -- Recommends optimal planting windows per crop and region
- **Fertilizer Recommendations** -- Nutrient-specific dosing based on soil deficiencies
- **Synthetic Simulation** -- Generate realistic farm datasets for 12 crop types
- **Rich Reports** -- Terminal-formatted yield reports with confidence intervals

## Supported Crops

Corn, Soybeans, Wheat (Winter/Spring), Rice, Cotton, Barley, Sorghum, Oats, Canola, Sunflower, Alfalfa.

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Run a full simulation with synthetic data
yieldcast simulate --fields 5 --seasons 3

# Predict yield for a single field
yieldcast predict --crop corn --area 100

# Generate a formatted report
yieldcast report --fields 10
```

## Project Structure

```
src/yieldcast/
  cli.py              CLI entry point (click + rich)
  models.py           Pydantic data models
  simulator.py        Synthetic farm data generator
  report.py           Rich terminal report formatting
  predictor/
    yield_model.py    GradientBoosting yield predictor
    weather.py        Weather feature extraction (GDD, drought)
    soil.py           Soil nutrient and moisture analysis
    satellite.py      NDVI computation from satellite bands
  optimizer/
    planting.py       Optimal planting date recommender
    fertilizer.py     Fertilizer dosing engine
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
