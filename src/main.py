import os
import argparse
import data_loader
import feature_engineering
import models
import evaluation

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='March Mania Prediction')
    parser.add_argument('--data_path', type=str, default="Z:\\kaggle\\MMLM2025\\March-Machine-Learning-Mania-2025\\input\\march-machine-learning-mania-2025",
                        help='Path to the data directory')
    parser.add_argument('--start_year', type=int, default=2018,
                        help='Only include seasons from this year onwards')
    parser.add_argument('--output_file', type=str, default="submission.csv",
                        help='Output file name for predictions')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level (0=minimal, 1=normal)')
    parser.add_argument('--stage', type=int, default=2,
                        help='Competition stage (1 or 2)')
    parser.add_argument('--test_mode', action='store_true',
                        help='Enable test mode to only use 10 games per year for quick testing')
    return parser.parse_args()

def run_pipeline(data_path, start_year=2018, output_file="submission.csv", verbose=1, stage=2, test_mode=False):
    """
    Run the entire March Mania prediction pipeline

    Args:
        data_path: Path to the data directory
        start_year: Only include seasons from this year onwards
        output_file: Output file name for predictions
        verbose: Verbosity level (0=minimal, 1=normal)
        stage: Competition stage (1 or 2)
        test_mode: If True, only use 10 games per year for quick testing
    """
    print(f"Running March Mania prediction pipeline from {start_year} onwards for Stage {stage}")
    if test_mode:
        print("TEST MODE ENABLED: Only using 10 games per year for quick testing")
        print("This will significantly reduce the dataset size and speed up the pipeline")
        print("Note: Results will be less accurate but useful for debugging the full pipeline flow")
    
    # Step 1: Load data
    season_detail, tourney_detail, seeds, teams, submission = data_loader.load_data(data_path, start_year, stage, test_mode)
    
    # 加载 KenPom 外部数据
    kenpom_df = data_loader.load_kenpom_data(data_path, teams)
    
    # Step 2: Merge and prepare games data
    games = data_loader.merge_and_prepare_games(season_detail, tourney_detail)
    
    # Step 3: Create seed dictionary
    seed_dict = data_loader.prepare_seed_dict(seeds)
    
    # Step 4: Feature engineering
    print("Performing feature engineering...")
    games = feature_engineering.feature_engineering(games, seed_dict)
    
    # Step 5: Add team statistics features
    games, team_stats = feature_engineering.add_team_features(games)
    
    # Step 6: Add advanced features
    games = feature_engineering.add_head_to_head_features(games)
    games = feature_engineering.add_recent_performance_features(games, window=5)
    
    # --- 新增：合并 KenPom 外部数据到训练数据 ---
    try:
        if not kenpom_df.empty:
            print("Merging KenPom features...")
            games = feature_engineering.merge_kenpom_features(games, kenpom_df)
        else:
            print("KenPom data is empty. Skipping KenPom feature merge.")
    except Exception as e:
        print(f"Error merging KenPom features: {e}")
        print("Continuing without KenPom features...")
    
    # Step 7: Aggregate features
    agg_features = feature_engineering.aggregate_features(games)
    
    # Step 8: Prepare submission features
    submission_df = feature_engineering.prepare_submission_features(
        submission.copy(), seed_dict, team_stats, data_loader.extract_game_info
    )
    
    # Step 9: Merge aggregated features if they exist
    if not agg_features.empty and 'IDTeams_c_score' in agg_features.columns:
        print("Merging aggregated features...")
        games = games.merge(agg_features, how='left', left_on='IDTeams', right_on='IDTeams_c_score')
        submission_df = submission_df.merge(agg_features, how='left', left_on='IDTeams', right_on='IDTeams_c_score')
    
    # --- 新增：将 KenPom 外部特征也合并到提交数据 ---
    try:
        if not kenpom_df.empty:
            print("Merging KenPom features to submission data...")
            submission_df = feature_engineering.merge_kenpom_features(submission_df, kenpom_df)
        else:
            print("KenPom data is empty. Skipping KenPom feature merge for submission.")
    except Exception as e:
        print(f"Error merging KenPom features to submission: {e}")
        print("Continuing without KenPom features in submission...")
    
    # Step 10: Prepare final datasets
    X_train, y_train, X_test, features = evaluation.prepare_dataset(games, submission_df)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Prediction data shape: {X_test.shape}")
    print(f"Submission data shape before modeling: {submission_df.shape}")
    
    # Step 11: Train models and predict
    print("Starting training and prediction...")
    test_pred = models.stacking_ensemble_cv(X_train, y_train, X_test, features, verbose=verbose)
    
    # Step 12: Generate submission file
    print("Generating submission file...")
    print(f"Test predictions shape: {test_pred.shape}")
    final_submission = evaluation.generate_submission(submission_df, test_pred, output_file)
    
    print(f"Pipeline complete! Submission saved to {output_file}")
    if test_mode:
        print("Remember: This was run in TEST MODE with a reduced dataset.")
        print("For actual competition predictions, run without the --test_mode flag.")
    
    return final_submission

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        data_path=args.data_path,
        start_year=args.start_year,
        output_file=args.output_file,
        verbose=args.verbose,
        stage=args.stage,
        test_mode=args.test_mode
    )
