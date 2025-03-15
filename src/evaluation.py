import numpy as np
import pandas as pd

def generate_submission(submission_df, test_pred, output_file='submission.csv'):
    """
    Generate the final submission file with predictions
    
    Args:
        submission_df: DataFrame with submission format
        test_pred: Array of predictions
        output_file: Filename for output submission
        
    Returns:
        DataFrame with ID and predictions
    """
    final_submission = submission_df[['ID']].copy()
    final_submission['Pred'] = test_pred.clip(0.001, 0.999)
    final_submission.to_csv(output_file, index=False)
    
    print_submission_stats(final_submission)
    
    return final_submission

def print_submission_stats(submission_df):
    """
    Print statistics about the submission file
    
    Args:
        submission_df: DataFrame with ID and Pred columns
    """
    print("Sample submission:")
    print(submission_df.head())
    print(f"Total predictions: {len(submission_df)}")
    print("\nPrediction value distribution:")
    print(f"  Min: {submission_df['Pred'].min():.4f}")
    print(f"  Max: {submission_df['Pred'].max():.4f}")
    print(f"  Mean: {submission_df['Pred'].mean():.4f}")
    print(f"  Std: {submission_df['Pred'].std():.4f}")
    print(f"  25% quantile: {submission_df['Pred'].quantile(0.25):.4f}")
    print(f"  50% quantile: {submission_df['Pred'].quantile(0.5):.4f}")
    print(f"  75% quantile: {submission_df['Pred'].quantile(0.75):.4f}")
    
    # Display histogram summary
    print("\nPrediction value histogram:")
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, bin_edges = np.histogram(submission_df['Pred'], bins=bins)
    for i in range(len(hist)):
        print(f"  {bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}: {hist[i]} ({hist[i]/len(submission_df)*100:.2f}%)")
    
    print(f"\nFinal submission has {len(submission_df)} rows")
    if len(submission_df) == 264628:
        print("✅ Matches submission 64 format (264,628 rows)")
    elif len(submission_df) == 131407:
        print("✓ Matches sample submission format (131,407 rows)")
    else:
        print(f"⚠️ Unexpected number of rows: {len(submission_df)}")

def prepare_dataset(games, submission_df, extract_agg_features=True):
    """
    Prepare the final datasets for training and prediction
    
    Args:
        games: DataFrame with processed game data
        submission_df: DataFrame with processed submission data
        extract_agg_features: Whether to extract aggregated features
        
    Returns:
        Tuple of (X_train, y_train, X_test, features list)
    """
    # Exclude non-model input columns and raw statistics
    exclude_cols = ['ID', 'DayNum', 'ST', 'Team1', 'Team2', 'IDTeams',
                    'IDTeam1', 'IDTeam2', 'WTeamID', 'WScore', 'LTeamID',
                    'LScore', 'NumOT', 'WinA', 'ScoreDiff', 'ScoreDiffNorm',
                    'WLoc', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA',
                    'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF',
                    'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR',
                    'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 'IDTeams_c_score',
                    'Pred']
    
    features = [c for c in games.columns if c not in exclude_cols and c in submission_df.columns]
    print(f"Using {len(features)} features for training, sample features: {features[:5]}")
    
    X_train = games[features].fillna(0)
    y_train = games['WinA']
    X_sub = submission_df[features].fillna(0)
    
    return X_train, y_train, X_sub, features 