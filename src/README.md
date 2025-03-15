# March Madness Prediction - Modular Codebase

This directory contains a modular implementation of the March Madness prediction pipeline. The code has been organized into separate modules for better maintainability, readability, and extensibility.

## Directory Structure

- `main.py` - Main entry point that orchestrates the entire pipeline
- `data_loader.py` - Functions for loading and preprocessing data
- `feature_engineering.py` - Functions for feature engineering and transformation
- `models.py` - Model definition, training, and prediction logic
- `evaluation.py` - Functions for evaluation and submission generation

## Usage

To run the pipeline with default settings:

```bash
python main.py
```
'''
'''

For more options:

```bash
python main.py --help
```

Available command line arguments:
- `--data_path`: Path to the data directory
- `--start_year`: Only include seasons from this year onwards
- `--output_file`: Output file name for predictions
- `--verbose`: Verbosity level (0=minimal, 1=normal)

Example:
```bash
python main.py --data_path "../data/" --start_year 2019 --output_file "my_predictions.csv"
```

## Pipeline Flow

1. Data loading (`data_loader.py`)
2. Feature engineering (`feature_engineering.py`)
3. Model training and prediction (`models.py`)
4. Evaluation and submission generation (`evaluation.py`)

## Key Features

- Time-series cross-validation approach
- Advanced feature engineering including head-to-head and recent performance metrics
- Stacking ensemble with diverse base models
- Robust evaluation with detailed statistics

## Extension Points

The modular structure makes it easy to extend the pipeline:

- Add new feature engineering functions in `feature_engineering.py`
- Implement new models or ensemble strategies in `models.py`
- Enhance evaluation metrics in `evaluation.py`
- Modify the data loading process in `data_loader.py` 