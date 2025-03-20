# March Machine Learning Mania 2025 Prediction Model

This project is a prediction model developed for the Kaggle competition "March Machine Learning Mania 2025," aimed at predicting the outcomes of the NCAA basketball tournament.

## Project Overview

This project builds a prediction model using historical NCAA basketball game data, analyzing various features such as team statistics, season performance, and historical matchup records to predict the winning probabilities of each game in the tournament.

## Main Features

- Data loading and processing
- Feature engineering (Elo ratings, Strength of Schedule, KenPom metrics, etc.)
- Multi-model ensemble (XGBoost, CatBoost, LightGBM, etc.)
- Monte Carlo simulation optimization
- Game evaluation and validation

## Project Structure

```
├── src/                   # Source code
│   ├── main.py            # Main program entry
│   ├── data_loader.py     # Data loading module
│   ├── feature_engineering.py  # Feature engineering module
│   ├── models.py          # Model definition and training
│   ├── evaluation.py      # Model evaluation
│   └── monte_carlo.py     # Monte Carlo simulation
├── input/                 # Input data directory
├── requirements.txt       # Project dependencies
├── LICENSE                # License
└── README.md              # Project description
```

## Installation and Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run predictions:
```bash
python src/main.py --data_path [data_path] --output_file submission.csv
```

## Parameter Descriptions

- `--data_path`: Path to the data directory
- `--start_year`: Starting year (default: 2021)
- `--output_file`: Output file name for predictions (default: submission.csv)
- `--verbose`: Verbosity level (0=minimal, 1=normal)
- `--stage`: Competition stage (1 or 2)
- `--test_mode`: Enable test mode (only use 10 games per year for quick testing)
- `--simulation_mode`: Enable simulation mode
- `--use_extended_models`: Use extended model set
- `--use_monte_carlo`: Enable Monte Carlo simulation to optimize predictions
- `--num_simulations`: Number of Monte Carlo simulations to run
- `--simulation_weight`: Weight to give to simulation results (0-1)

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details. 