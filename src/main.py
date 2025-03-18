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
                        help='Only include seasons from this year onwards')
    parser.add_argument('--output_file', type=str, default="submission.csv",
                        help='Output file name for predictions')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level (0=minimal, 1=normal)')
    parser.add_argument('--stage', type=int, default=2,
                        help='Competition stage (1 or 2)')
    parser.add_argument('--test_mode', action='store_true',
                        help='Enable test mode to only use 10 games per year for quick testing')
    parser.add_argument('--simulation_mode', action='store_true',
                        help='Enable simulation mode to train on 2021-2023 data + 2024 regular season data and evaluate on 2024 tournament data')
    parser.add_argument('--use_extended_models', action='store_true',
                        help='Use extended model set with model variants for greater diversity')
    # Add new arguments for Monte Carlo simulation
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
    Run the entire March Mania prediction pipeline

    Args:
        data_path: Path to the data directory
        start_year: Only include seasons from this year onwards
        output_file: Output file name for predictions
        verbose: Verbosity level (0=minimal, 1=normal)
        stage: Competition stage (1 or 2)
        test_mode: If True, only use 10 games per year for quick testing
        simulation_mode: If True, train on 2021-2023 data + 2024 regular season data and evaluate on 2024 tournament data
        use_extended_models: If True, use extended model set with model variants for greater diversity
        use_monte_carlo: If True, use Monte Carlo simulation to optimize predictions
        num_simulations: Number of Monte Carlo simulations to run
        simulation_weight: Weight to give to simulation results (0-1)
    """
    print(f"Running March Mania prediction pipeline from {start_year} onwards for Stage {stage}")
    if test_mode:
        print("TEST MODE ENABLED: Only using 10 games per year for quick testing")
        print("This will significantly reduce the dataset size and speed up the pipeline")
        print("Note: Results will be less accurate but useful for debugging the full pipeline flow")
    
    if simulation_mode:
        print("SIMULATION MODE ENABLED: Training on 2021-2023 data + 2024 regular season data")
        print("Will evaluate model performance on 2024 tournament data")
    
    if use_extended_models:
        print("EXTENDED MODELS ENABLED: Using a larger set of models with varied hyperparameters")
        print("This will significantly increase training time but may improve prediction accuracy")
    
    if use_monte_carlo:
        print("MONTE CARLO SIMULATION ENABLED: Will optimize predictions using tournament simulation")
        print(f"Number of simulations: {num_simulations}, Simulation weight: {simulation_weight}")
    
    # Step 1: Load data
    season_detail, tourney_detail, seeds, teams, submission = data_loader.load_data(data_path, start_year, stage, test_mode)
    
    # Load KenPom external data
    kenpom_df = data_loader.load_kenpom_data(data_path, teams)
    
    # Step 2: Merge and prepare games data
    games = data_loader.merge_and_prepare_games(season_detail, tourney_detail)
    
    # Step 3: Create seed dictionary
    seed_dict = data_loader.prepare_seed_dict(seeds)
    
    # Step 4: Basic feature engineering
    print("Performing feature engineering...")
    games = feature_engineering.feature_engineering(games, seed_dict)
    
    # Step 5: Add team statistics features (cumulative, avoiding target leakage)
    games, team_stats_cum = feature_engineering.add_team_features(games)
    
    # Step 6: Add advanced features: head-to-head and recent performance
    games = feature_engineering.add_head_to_head_features(games)
    games = feature_engineering.add_recent_performance_features(games, window=5)
    
    # Step 6.5: Add new features: Elo ratings, Strength of Schedule, and key statistical differentials
    print("Adding advanced predictive features...")
    games = feature_engineering.add_elo_ratings(games, k_factor=20, initial_elo=1500, reset_each_season=True)
    games = feature_engineering.add_strength_of_schedule(games, team_stats_cum)
    games = feature_engineering.add_key_stat_differentials(games)
    games = feature_engineering.enhance_key_stat_differentials(games)
    games = feature_engineering.add_historical_tournament_performance(games, seed_dict)
    
    # --- Merge KenPom features into training data ---
    try:
        if not kenpom_df.empty:
            print("Merging KenPom features...")
            games = feature_engineering.enhance_kenpom_features(games, kenpom_df)
        else:
            print("KenPom data is empty. Skipping KenPom feature merge.")
    except Exception as e:
        print(f"Error merging KenPom features: {e}")
        print("Continuing without KenPom features...")
    
    # Step 7: Aggregate features
    agg_features = feature_engineering.aggregate_features(games)
    
    # --- Filter training data based on mode ---
    current_season = games['Season'].max()
    print(f"Current season: {current_season}")
    
    # Set aside 2024 tournament data for evaluation in simulation mode
    eval_data = None
    if simulation_mode and current_season >= 2024:
        print("Extracting 2024 tournament data for evaluation...")
        eval_data = games[(games['Season'] == 2024) & (games['GameType'] == 'Tournament')].copy()
        print(f"Evaluation data size: {len(eval_data)} games")
    
    # Filter training data
    if simulation_mode:
        # For simulation mode: train on 2021-2023 all data + 2024 regular season data
        games = games[
            ((games['Season'] < 2024)) |
            ((games['Season'] == 2024) & (games['GameType'] == 'Regular'))
        ]
    else:
        # Normal mode: use current season Regular games and previous seasons Tournament games
        games = games[
            ((games['Season'] == current_season) & (games['GameType'] == 'Regular')) |
            ((games['Season'] < current_season) & (games['GameType'].isin(['Regular', 'Tournament'])))
        ]
    
    print(f"Training data rows after filtering: {len(games)}")
    
    # Step 8: Prepare submission features (submission games remain unchanged)
    submission_df = feature_engineering.prepare_submission_features(
        submission.copy(), seed_dict, team_stats_cum, data_loader.extract_game_info
    )
    
    # Verify submission data integrity before further processing
    if 'original_index' in submission_df.columns:
        expected_rows = len(submission_df['original_index'].unique())
        actual_rows = len(submission_df)
        if expected_rows != actual_rows:
            print(f"WARNING: Submission data integrity issue detected!")
            print(f"Expected {expected_rows} unique rows, but found {actual_rows} total rows")
            print("This suggests duplicated or missing rows which will cause misalignment")
    
    # Apply the same new features to submission data
    print("Adding advanced predictive features to submission data...")
    submission_df = feature_engineering.add_elo_ratings(submission_df, k_factor=20, initial_elo=1500, reset_each_season=True)
    submission_df = feature_engineering.add_strength_of_schedule(submission_df, team_stats_cum)
    # Note: Key stat differentials require game details which may not be available for future games
    # But if they are in the submission data, they will be used
    submission_df = feature_engineering.add_key_stat_differentials(submission_df)
    submission_df = feature_engineering.enhance_key_stat_differentials(submission_df)
    submission_df = feature_engineering.add_historical_tournament_performance(submission_df, seed_dict)
    
    # Try to add KenPom features to submission data
    try:
        if not kenpom_df.empty:
            print("Merging KenPom features to submission data...")
            submission_df = feature_engineering.enhance_kenpom_features(submission_df, kenpom_df)
        else:
            print("KenPom data is empty. Skipping KenPom feature merge for submission.")
    except Exception as e:
        print(f"Error merging KenPom features to submission: {e}")
    
    # Apply new features to evaluation data if in simulation mode
    if simulation_mode and eval_data is not None and not eval_data.empty:
        print("Adding advanced predictive features to evaluation data...")
        eval_data = feature_engineering.add_elo_ratings(eval_data, k_factor=20, initial_elo=1500, reset_each_season=True)
        eval_data = feature_engineering.add_strength_of_schedule(eval_data, team_stats_cum)
        eval_data = feature_engineering.add_key_stat_differentials(eval_data)
        eval_data = feature_engineering.enhance_key_stat_differentials(eval_data)
        eval_data = feature_engineering.add_historical_tournament_performance(eval_data, seed_dict)
        
        # Try to add KenPom features to eval data
        try:
            if not kenpom_df.empty:
                print("Merging KenPom features to evaluation data...")
                eval_data = feature_engineering.enhance_kenpom_features(eval_data, kenpom_df)
        except Exception as e:
            print(f"Error merging KenPom features to evaluation data: {e}")
    
    # Verify submission data integrity after all feature engineering
    if 'original_index' in submission_df.columns:
        expected_rows = len(submission_df['original_index'].unique())
        actual_rows = len(submission_df)
        if expected_rows != actual_rows:
            print(f"WARNING: Submission data integrity issue detected after feature engineering!")
            print(f"Expected {expected_rows} unique rows, but found {actual_rows} total rows")
            print("Attempting to fix by removing duplicates and sorting by original index...")
            submission_df = submission_df.drop_duplicates(subset=['original_index']).sort_values('original_index').reset_index(drop=True)
            print(f"After fixing: {len(submission_df)} rows")
    
    # Step 10: Prepare final datasets
    X_train, y_train, X_test, features = evaluation.prepare_dataset(games, submission_df)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Prediction data shape: {X_test.shape}")
    print(f"Submission data shape before modeling: {submission_df.shape}")
    
    # Find the best threshold based on Brier Score using time-series cross-validation on training data
    best_threshold = find_best_threshold(X_train, y_train, features, use_extended_models)
    print(f"Best threshold based on Brier Score: {best_threshold:.4f}")
    
    # Step 11: Train models and predict
    print("Starting training and prediction...")
    test_pred = models.stacking_ensemble_cv(X_train, y_train, X_test, features, verbose=verbose, use_extended_models=use_extended_models)
    
    # Step 12: Generate initial submission file (create a temporary version first if using Monte Carlo)
    print("Generating initial submission file...")
    print(f"Test predictions shape: {test_pred.shape}")
    
    if use_monte_carlo:
        # Save initial predictions to a temporary file
        initial_submission = evaluation.generate_submission(
            submission_df, test_pred, output_file.replace('.csv', '_pre_monte_carlo.csv')
        )
    else:
        # Generate the final submission directly if not using Monte Carlo
        final_submission = evaluation.generate_submission(submission_df, test_pred, output_file)
    
    # Step 12.5: Monte Carlo simulation (if enabled)
    if use_monte_carlo:
        print("\n--- Running Monte Carlo Tournament Simulation ---")
        import monte_carlo
        
        # Make sure initial_submission has the necessary columns
        initial_submission = initial_submission.copy()
        
        # Run Monte Carlo simulation and optimize predictions
        optimized_submission = monte_carlo.simulate_and_optimize(
            initial_submission, 
            seed_dict,
            team_stats_cum,  # Pass team statistics for better probability estimation
            num_simulations=num_simulations,
            simulation_weight=simulation_weight
        )
        
        # Generate final submission file with optimized predictions
        print("Generating final submission file with Monte Carlo optimized predictions...")
        final_submission = optimized_submission.copy()
        final_submission.to_csv(output_file, index=False)
        
        # Print comparison of original vs. optimized predictions
        print("\nComparing original vs. optimized predictions:")
        original_mean = initial_submission['Pred'].mean()
        optimized_mean = final_submission['Pred'].mean()
        print(f"  Original mean: {original_mean:.4f}")
        print(f"  Optimized mean: {optimized_mean:.4f}")
        print(f"  Mean difference: {abs(optimized_mean - original_mean):.4f}")
        
        # Calculate and print the average absolute change in predictions
        common_ids = set(initial_submission['ID']).intersection(final_submission['ID'])
        changes = [abs(final_submission.loc[final_submission['ID'] == id, 'Pred'].iloc[0] - 
                      initial_submission.loc[initial_submission['ID'] == id, 'Pred'].iloc[0])
                  for id in common_ids]
        avg_change = sum(changes) / len(changes) if changes else 0
        print(f"  Average absolute change in predictions: {avg_change:.4f}")
        
        # Count significantly changed predictions (e.g., >0.05)
        sig_changes = sum(1 for c in changes if c > 0.05)
        print(f"  Predictions with significant changes (>0.05): {sig_changes} ({sig_changes/len(changes)*100:.1f}%)")
    
    # Print final submission stats
    if use_monte_carlo:
        evaluation.print_submission_stats(final_submission)
    
    # Step 13: Evaluate on 2024 tournament data if in simulation mode
    if simulation_mode and eval_data is not None and not eval_data.empty:
        print("\n--- Simulation Mode Evaluation on 2024 Tournament Data ---")
        X_eval, y_eval, _, _ = evaluation.prepare_dataset(eval_data, None)
        
        print(f"Evaluation data shape: {X_eval.shape}")
        eval_metrics = evaluate_model(X_train, y_train, X_eval, y_eval, features, use_extended_models)
        
        # Save evaluation results
        eval_output = output_file.replace('.csv', '_eval_results.csv')
        evaluation.save_evaluation_results(eval_metrics, eval_output)
        print(f"Evaluation results saved to {eval_output}")
    
    print(f"Pipeline complete! Submission saved to {output_file}")
    if test_mode:
        print("Remember: This was run in TEST MODE with a reduced dataset.")
        print("For actual competition predictions, run without the --test_mode flag.")
    if use_monte_carlo:
        print("Monte Carlo simulation was used to optimize predictions.")
        print(f"Number of simulations: {num_simulations}")
        print(f"Simulation weight: {simulation_weight}")
    
    return final_submission

def evaluate_model(X_train, y_train, X_eval, y_eval, features, use_extended_models=False):
    """
    Evaluate model performance on the evaluation dataset
    
    Args:
        X_train: Training features
        y_train: Training target
        X_eval: Evaluation features
        y_eval: Evaluation target
        features: List of feature names
        use_extended_models: Whether to use the extended model set
        
    Returns:
        Dictionary with evaluation metrics and best threshold
    """
    from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
    import models  # Import here to avoid circular imports
    
    print("Evaluating model performance on 2024 tournament data...")
    
    # Train model using the same approach as for predictions
    pred_eval = models.stacking_ensemble_cv(X_train, y_train, X_eval, features, verbose=1, use_extended_models=use_extended_models)
    
    # Calculate metrics
    metrics = {
        'log_loss': log_loss(y_eval, pred_eval),
        'accuracy': accuracy_score(y_eval, pred_eval > 0.5),
        'roc_auc': roc_auc_score(y_eval, pred_eval)
    }
    
    # Calculate Brier Score
    brier_score = brier_score_loss(y_eval, pred_eval)
    print(f"Brier Score: {brier_score:.4f}")
    metrics['brier_score'] = brier_score
    
    # Use the best threshold found during training (passed from the main pipeline)
    print(f"Evaluation metrics: {metrics}")
    
    return metrics

def find_best_threshold(X_train, y_train, features, use_extended_models=False):
    """
    Find the best threshold based on Brier Score using time-series cross-validation on training data
    
    Args:
        X_train: Training features
        y_train: Training target
        features: List of feature names
        use_extended_models: Whether to use the extended model set
        
    Returns:
        Best threshold value
    """
    import models  # Import here to avoid circular imports
    import numpy as np
    
    print("Finding best threshold based on Brier Score using time-series cross-validation...")
    
    # 检查是否有Season列用于时间序列验证
    if 'Season' not in X_train.columns:
        print("警告: 'Season'列未找到，无法进行时间序列验证。将使用默认阈值0.5")
        return 0.5
    
    # 获取所有唯一的赛季并排序
    seasons = np.sort(X_train['Season'].unique())
    print(f"找到{len(seasons)}个赛季用于时间序列验证: {seasons}")
    
    if len(seasons) <= 1:
        print("警告: 只有一个赛季的数据，无法进行时间序列验证。将使用默认阈值0.5")
        return 0.5
    
    # 初始化预测数组
    train_pred = np.zeros(len(y_train))
    train_indices_used = np.zeros(len(y_train), dtype=bool)
    
    # 使用时间序列前向验证：对每个赛季，使用之前所有赛季的数据进行训练
    for i, season in enumerate(seasons[1:], 1):  # 从第二个赛季开始
        print(f"  验证赛季 {season}，使用赛季 {seasons[:i]} 的数据进行训练")
        
        # 训练集：当前赛季之前的所有数据
        train_idx = X_train[X_train['Season'] < season].index
        
        # 验证集：当前赛季的数据
        val_idx = X_train[X_train['Season'] == season].index
        
        if len(train_idx) == 0 or len(val_idx) == 0:
            print(f"  警告: 赛季 {season} 的训练或验证数据为空，跳过此折")
            continue
        
        X_train_fold = X_train.loc[train_idx]
        X_val_fold = X_train.loc[val_idx]
        y_train_fold = y_train.loc[train_idx]
        
        # 训练模型并在验证集上预测
        try:
            val_pred = models.stacking_ensemble_cv(X_train_fold, y_train_fold, X_val_fold, features, verbose=0, use_extended_models=use_extended_models)
            train_pred[val_idx] = val_pred
            train_indices_used[val_idx] = True
            print(f"  赛季 {season} 验证完成，验证集大小: {len(val_idx)}")
        except Exception as e:
            print(f"  警告: 赛季 {season} 验证时出错: {e}")
            continue
    
    # 只使用有预测的样本来计算最佳阈值
    if np.sum(train_indices_used) == 0:
        print("警告: 没有成功进行任何验证，将使用默认阈值0.5")
        return 0.5
    
    y_train_used = y_train[train_indices_used]
    train_pred_used = train_pred[train_indices_used]
    
    print(f"时间序列验证完成，共使用 {np.sum(train_indices_used)} 个样本进行阈值优化")
    
    # 计算不同阈值的Brier Score
    thresholds = np.arange(0.0, 1.01, 0.01)
    best_threshold = 0.5
    best_brier_score = float('inf')
    
    for thresh in thresholds:
        # 应用阈值到预测
        adjusted_preds = np.where(train_pred_used >= thresh, 1.0, 0.0)
        
        # 计算Brier Score
        brier = brier_score_loss(y_train_used, adjusted_preds)
        
        if brier < best_brier_score:
            best_brier_score = brier
            best_threshold = thresh
    
    print(f"最佳阈值: {best_threshold:.2f}，Brier Score: {best_brier_score:.4f}")
    
    return best_threshold

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        data_path=args.data_path,
        start_year=args.start_year,
        output_file=args.output_file,
        verbose=args.verbose,
        stage=args.stage,
        test_mode=args.test_mode,
        simulation_mode=args.simulation_mode,
        use_extended_models=args.use_extended_models,
        use_monte_carlo=args.use_monte_carlo,
        num_simulations=args.num_simulations,
        simulation_weight=args.simulation_weight
    )
    
# python main.py --use_monte_carlo --num_simulations=10000 --simulation_weight=0.3