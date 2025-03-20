import os
import argparse
import data_loader
import feature_engineering
import models
import evaluation
import numpy as np
from sklearn.metrics import brier_score_loss
import pandas as pd

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='March Mania Prediction')
    parser.add_argument('--data_path', type=str, default="Z:\\kaggle\\MMLM2025\\March-Machine-Learning-Mania-2025\\input\\march-machine-learning-mania-2025",
                        help='Path to the data directory')
    parser.add_argument('--start_year', type=int, default=2021,
                        help='Only include seasons starting from this year')
    parser.add_argument('--output_file', type=str, default="submission.csv",
                        help='Output file name for predictions')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level (0=minimal, 1=normal)')
    parser.add_argument('--stage', type=int, default=2,
                        help='Competition stage (1 or 2)')
    parser.add_argument('--test_mode', action='store_true',
                        help='Enable test mode, only use 10 games per year for quick testing')
    parser.add_argument('--simulation_mode', action='store_true',
                        help='Enable simulation mode, train on 2021-2023 data + 2024 regular season data, evaluate on 2024 tournament data')
    parser.add_argument('--use_extended_models', action='store_true',
                        help='Use extended model set, include more model variants to increase diversity')
    # New Monte Carlo simulation parameters
    parser.add_argument('--use_monte_carlo', action='store_true',
                        help='Enable Monte Carlo simulation to optimize predictions')
    parser.add_argument('--num_simulations', type=int, default=10000,
                        help='Number of Monte Carlo simulations to run')
    parser.add_argument('--simulation_weight', type=float, default=0.3,
                        help='Weight to give to simulation results (0-1)')
    return parser.parse_args()

def run_pipeline(data_path, start_year=2021, output_file="submission.csv", verbose=1, stage=2, test_mode=False, 
                simulation_mode=False, use_extended_models=False, use_monte_carlo=False, num_simulations=10000, 
                simulation_weight=0.3):
    """
    Run the complete March Mania prediction process

    Parameters:
        data_path: Path to the data directory
        start_year: Only include seasons starting from this year
        output_file: Output file name for predictions
        verbose: Verbosity level (0=minimal, 1=normal)
        stage: Competition stage (1 or 2)
        test_mode: If True, only use 10 games per year for quick testing
        simulation_mode: If True, train on 2021-2023 data + 2024 regular season data, evaluate on 2024 tournament data
        use_extended_models: If True, use extended model set, include more model variants to increase diversity
        use_monte_carlo: If True, use Monte Carlo simulation to optimize predictions
        num_simulations: Number of Monte Carlo simulations to run
        simulation_weight: Weight to give to simulation results (0-1)
    """
    print(f"Running March Mania prediction process starting from {start_year} for {stage} stage")
    if test_mode:
        print("Test mode enabled: Only use 10 games per year for quick testing")
        print("This will significantly reduce dataset size and accelerate process")
        print("Note: Results will be less accurate but useful for debugging full process")
    
    if simulation_mode:
        print("Simulation mode enabled: Train on 2021-2023 data + 2024 regular season data")
        print("Model performance will be evaluated on 2024 tournament data")
    
    if use_extended_models:
        print("Extended models enabled: Use larger model set with more model variants")
        print("This will significantly increase training time but may improve prediction accuracy")
    
    if use_monte_carlo:
        print("Monte Carlo simulation enabled: Use tournament simulation to optimize predictions")
        print(f"Simulation runs: {num_simulations}, Simulation weight: {simulation_weight}")
    
    # Step 1: Load data
    season_detail, tourney_detail, seeds, teams, submission = data_loader.load_data(data_path, start_year, stage, test_mode)
    
    # Load merged KenPom data
    merged_kenpom_df = data_loader.load_merged_kenpom_data(data_path, teams)
    
    # Create empty DataFrame for original KenPom data as it has been disabled
    kenpom_df = pd.DataFrame()
    print("Original KenPom data loading disabled, using merged_kenpom.csv instead.")
    
    # Step 2: Merge and prepare game data
    games = data_loader.merge_and_prepare_games(season_detail, tourney_detail)
    
    # Step 3: Create seed dictionary
    seed_dict = data_loader.prepare_seed_dict(seeds)
    
    # Step 4: Basic feature engineering
    print("Executing feature engineering...")
    games = feature_engineering.feature_engineering(games, seed_dict)
    
    # Step 5: Add team statistics features (cumulative, to avoid target leakage)
    games, team_stats_cum = feature_engineering.add_team_features(games)
    
    # Step 6: Add advanced features: Head-to-head history and recent performance
    games = feature_engineering.add_head_to_head_features(games)
    games = feature_engineering.add_recent_performance_features(games, window=5)
    
    # Step 6.5: Add new features: Elo ratings, schedule strength, and key stat differentials
    print("Adding advanced prediction features...")
    games = feature_engineering.add_elo_ratings(games, k_factor=20, initial_elo=1500, reset_each_season=True)
    games = feature_engineering.add_strength_of_schedule(games, team_stats_cum)
    games = feature_engineering.add_key_stat_differentials(games)
    games = feature_engineering.enhance_key_stat_differentials(games)
    games = feature_engineering.add_historical_tournament_performance(games, seed_dict)
    
    # --- Merge KenPom features into training data ---
    if not merged_kenpom_df.empty:
        print("Adding merged KenPom features...")
        games = feature_engineering.merge_merged_kenpom_features(games, merged_kenpom_df)
    else:
        print("No available merged KenPom data.")
    

    
    # Step 7: Aggregate features
    agg_features = feature_engineering.aggregate_features(games)
    
    # --- Filter training data based on mode ---
    current_season = games['Season'].max()
    print(f"Current season: {current_season}")
    
    # In simulation mode, reserve 2024 tournament data for evaluation
    eval_data = None
    if simulation_mode and current_season >= 2024:
        print("Extracting 2024 tournament data for evaluation...")
        eval_data = games[(games['Season'] == 2024) & (games['GameType'] == 'Tournament')].copy()
        print(f"Evaluation data size: {len(eval_data)} games")
    
    # Filter training data
    if simulation_mode:
        # Simulation mode: Train on all data from 2021-2023 + 2024 regular season data
        games = games[
            ((games['Season'] < 2024)) |
            ((games['Season'] == 2024) & (games['GameType'] == 'Regular'))
        ]
    else:
        # Normal mode: Use regular season and previous season tournament data
        games = games[
            ((games['Season'] == current_season) & (games['GameType'] == 'Regular')) |
            ((games['Season'] < current_season) & (games['GameType'].isin(['Regular', 'Tournament'])))
        ]
    
    print(f"Filtered training data row count: {len(games)}")
    
    # Step 8: Prepare submission features (keep submission data unchanged)
    submission_df = feature_engineering.prepare_submission_features(
        submission, seed_dict, team_stats_cum, data_loader.extract_game_info
    )
    
    # Front-end validation of submission data integrity
    if 'original_index' in submission_df.columns:
        expected_rows = len(submission_df['original_index'].unique())
        actual_rows = len(submission_df)
        if expected_rows != actual_rows:
            print(f"Warning: Detected submission data integrity issue!")
            print(f"Expected {expected_rows} unique rows, but found {actual_rows} total rows")
            print("This indicates duplicate or missing rows, which will cause misalignment")
    
    # Apply same new features to submission data
    print("Adding advanced prediction features to submission data...")
    submission_df = feature_engineering.add_elo_ratings(submission_df, k_factor=20, initial_elo=1500, reset_each_season=True)
    submission_df = feature_engineering.add_strength_of_schedule(submission_df, team_stats_cum)
    # Note: Key stat differentials require game details, which may not apply to future games
    # But they will be used if they exist in submission data
    submission_df = feature_engineering.add_key_stat_differentials(submission_df)
    submission_df = feature_engineering.enhance_key_stat_differentials(submission_df)
    submission_df = feature_engineering.add_historical_tournament_performance(submission_df, seed_dict)
    
    # Try adding KenPom features to submission data
    if not merged_kenpom_df.empty:
        print("Adding merged KenPom features to submission data...")
        submission_df = feature_engineering.merge_merged_kenpom_features(submission_df, merged_kenpom_df)
    else:
        print("Submission data has no available merged KenPom data.")
    
    # If in simulation mode, apply new features to evaluation data
    if simulation_mode and eval_data is not None and not eval_data.empty:
        print("Adding advanced prediction features to evaluation data...")
        eval_data = feature_engineering.add_elo_ratings(eval_data, k_factor=20, initial_elo=1500, reset_each_season=True)
        eval_data = feature_engineering.add_strength_of_schedule(eval_data, team_stats_cum)
        eval_data = feature_engineering.add_key_stat_differentials(eval_data)
        eval_data = feature_engineering.enhance_key_stat_differentials(eval_data)
        eval_data = feature_engineering.add_historical_tournament_performance(eval_data, seed_dict)
        
        # Try adding KenPom features to evaluation data
        if not merged_kenpom_df.empty and simulation_mode:
            print("Adding merged KenPom features to evaluation data...")
            eval_data = feature_engineering.merge_merged_kenpom_features(eval_data, merged_kenpom_df)
        

    
    # Validate submission data integrity after all feature engineering
    if 'original_index' in submission_df.columns:
        expected_rows = len(submission_df['original_index'].unique())
        actual_rows = len(submission_df)
        if expected_rows != actual_rows:
            print(f"Warning: Detected submission data integrity issue after processing!")
            print(f"Expected {expected_rows} unique rows, but found {actual_rows} total rows")
            print("This indicates duplicate or missing rows, which may cause misalignment")
            # Attempt to fix this problem before continuing
            print("Attempting to fix by removing possible duplicate rows...")
            # Maintain original index order
            submission_df = submission_df.drop_duplicates(subset=['original_index'], keep='first')
            # Reindex to ensure correct order
            submission_df = submission_df.sort_values('original_index').reset_index(drop=True)
            print(f"Fixed row count: {len(submission_df)}")
    
    # Step 9: Prepare training and evaluation data
    # Here add new feature selection or engineering steps
    combined_features = feature_engineering.select_best_features(games, is_tournament=False)
    print(f"Selected {len(combined_features)} features for model training")
    
    X_train = games[combined_features].copy()
    y_train = games['WTeamWins'].copy()
    
    # If available, get same features from submission data
    X_submission = submission_df[combined_features].copy()
    
    if verbose >= 1:
        print(f"Training data shape: {X_train.shape}")
        print(f"Submission data shape: {X_submission.shape}")
    
    # Display percentage of missing values in training set
    missing_train = (X_train.isnull().sum() / len(X_train)) * 100
    missing_train = missing_train[missing_train > 0].sort_values(ascending=False)
    if len(missing_train) > 0 and verbose >= 1:
        print("\nMissing features in training set (percentage):")
        for feature, pct in missing_train.items():
            print(f"{feature}: {pct:.2f}%")
    
    # Step 10: Handle missing values
    X_train = feature_engineering.handle_missing_values(X_train)
    X_submission = feature_engineering.handle_missing_values(X_submission)
    
    # Apply same missing values handling to evaluation data (if in simulation mode)
    if simulation_mode and eval_data is not None and not eval_data.empty:
        X_eval = eval_data[combined_features].copy()
        y_eval = eval_data['WTeamWins'].copy()
        X_eval = feature_engineering.handle_missing_values(X_eval)
        print(f"Evaluation data shape: {X_eval.shape}")
        
        # Step 11.5: If in simulation mode, evaluate model performance
        evaluate_model(X_train, y_train, X_eval, y_eval, combined_features, use_extended_models)
    
    # Step 11: Train model and generate predictions
    # Use model set for training
    print("Training model set...")
    
    # Select best threshold (if not in simulation mode)
    best_threshold = 0.5  # Default value
    if not simulation_mode:
        best_threshold = find_best_threshold(X_train, y_train, combined_features, use_extended_models)
        print(f"Using best threshold: {best_threshold:.4f}")
    
    # Select base or extended model set based on configuration
    if use_extended_models:
        print("Using extended model set...")
        models_list = models.get_extended_models()
    else:
        models_list = models.get_base_models()
    
    # Train all models
    trained_models = models.train_all_models(X_train, y_train, models_list, verbose=verbose)
    
    # Generate predictions
    predictions = models.predict_with_all_models(trained_models, X_submission, verbose=verbose)
    
    # Step 12: If enabled, use Monte Carlo simulation optimization
    if use_monte_carlo:
        print(f"Running {num_simulations} Monte Carlo simulations...")
        import monte_carlo
        
        # Use unadjusted model predictions for submission as baseline
        base_predictions = predictions.copy()
        
        # Execute Monte Carlo simulations
        simulation_predictions = monte_carlo.run_monte_carlo_simulations(
            submission, seed_dict, trained_models, X_submission, 
            num_simulations=num_simulations,
            verbose=verbose
        )
        
        # Blend base predictions and simulation predictions
        for i in range(len(predictions)):
            predictions[i] = (1 - simulation_weight) * base_predictions[i] + simulation_weight * simulation_predictions[i]
        
        print(f"Completed Monte Carlo simulations, using {simulation_weight:.2f} weight blending")
    
    # Fix edge cases: Ensure predictions are in [0.01, 0.99] range
    predictions = np.clip(predictions, 0.01, 0.99)
    
    # Prepare final submission
    submission_output = pd.read_csv(os.path.join(data_path, "SampleSubmissionStage2.csv" if stage == 2 else "SampleSubmissionStage1.csv"))
    submission_output['Pred'] = predictions
    
    # Save predictions
    submission_output.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    return submission_output

def evaluate_model(X_train, y_train, X_eval, y_eval, features, use_extended_models=False):
    """
    Evaluate model performance on real evaluation data
    
    Parameters:
        X_train: Training features
        y_train: Training labels
        X_eval: Evaluation features
        y_eval: Evaluation labels
        features: List of features
        use_extended_models: Whether to use extended model set
    """
    print("\nEvaluating model performance on 2024 tournament data...")
    
    # Select base or extended model set based on configuration
    if use_extended_models:
        print("Evaluating with extended model set...")
        models_list = models.get_extended_models()
    else:
        models_list = models.get_base_models()
    
    # Train all models
    trained_models = models.train_all_models(X_train, y_train, models_list, verbose=1)
    
    # Predict on evaluation set
    eval_predictions = models.predict_with_all_models(trained_models, X_eval, verbose=1)
    
    # Evaluation metrics: Accuracy and log loss
    accuracy = (eval_predictions > 0.5).astype(int) == y_eval
    accuracy_pct = accuracy.mean() * 100
    
    # Calculate log loss
    from sklearn.metrics import log_loss
    log_loss_score = log_loss(y_eval, eval_predictions)
    
    # Calculate Brier score
    brier_score = brier_score_loss(y_eval, eval_predictions)
    
    print(f"Evaluation set size: {len(X_eval)} games")
    print(f"Accuracy: {accuracy_pct:.2f}%")
    print(f"Log loss: {log_loss_score:.4f}")
    print(f"Brier score: {brier_score:.4f}")
    print("Note: Tournament predictions are usually more challenging than regular season predictions")

def find_best_threshold(X_train, y_train, features, use_extended_models=False):
    """
    Find the best threshold to convert probabilities to binary predictions
    
    Parameters:
        X_train: Training features
        y_train: Training labels
        features: List of features
        use_extended_models: Whether to use extended model set
        
    Returns:
        Best threshold value
    """
    from sklearn.model_selection import StratifiedKFold
    
    print("Finding best prediction threshold...")
    
    # Select base or extended model set based on configuration
    if use_extended_models:
        models_list = models.get_extended_models()
    else:
        models_list = models.get_base_models()
    
    # Use cross validation to find best threshold
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_preds = []
    all_targets = []
    
    for train_idx, val_idx in kf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Train all models
        trained_models = models.train_all_models(X_train_fold, y_train_fold, models_list, verbose=0)
        
        # Get validation set predictions
        val_preds = models.predict_with_all_models(trained_models, X_val_fold, verbose=0)
        
        all_preds.extend(val_preds)
        all_targets.extend(y_val_fold)
    
    # Convert list to array
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Try different thresholds
    thresholds = np.arange(0.3, 0.7, 0.01)
    best_accuracy = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        accuracy = np.mean((all_preds > threshold) == all_targets)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    print(f"Best threshold: {best_threshold:.4f}, Cross-validation accuracy: {best_accuracy*100:.2f}%")
    return best_threshold

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        args.data_path,
        args.start_year,
        args.output_file,
        args.verbose,
        args.stage,
        args.test_mode,
        args.simulation_mode,
        args.use_extended_models,
        args.use_monte_carlo,
        args.num_simulations,
        args.simulation_weight
    )

# Solution 1:
# python main.py --use_monte_carlo --num_simulations=20000 --simulation_weight=0.4