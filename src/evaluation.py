import numpy as np
import pandas as pd
import os

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
    # Fix submission format if needed
    final_submission = fix_submission_format(submission_df, test_pred)
    
    # Save to file
    final_submission.to_csv(output_file, index=False)
    
    print_submission_stats(final_submission)
    
    return final_submission

def fix_submission_format(submission_df, test_pred):
    """
    Fix submission format issues by adjusting the dataframe to match expected row counts
    
    Args:
        submission_df: DataFrame with submission data
        test_pred: Array of predictions
        
    Returns:
        DataFrame with corrected format
    """
    expected_counts = [264628, 131407]  # submission_64.csv and SampleSubmissionStage2.csv counts
    current_count = len(submission_df)
    
    if current_count in expected_counts:
        print(f"Submission format already correct with {current_count} rows")
        return submission_df
    
    print(f"Fixing submission format. Current count: {current_count}, Expected: {expected_counts}")
    
    # Check if we can find the sample submission file to use as reference
    data_path = "Z:\\kaggle\\MMLM2025\\March-Machine-Learning-Mania-2025\\input\\march-machine-learning-mania-2025"
    sample_file = None
    
    if os.path.exists(os.path.join(data_path, "SampleSubmissionStage2.csv")):
        sample_file = os.path.join(data_path, "SampleSubmissionStage2.csv")
        print(f"Using SampleSubmissionStage2.csv as reference")
    elif os.path.exists(os.path.join(data_path, "SampleSubmissionStage1.csv")):
        sample_file = os.path.join(data_path, "SampleSubmissionStage1.csv")
        print(f"Using SampleSubmissionStage1.csv as reference")
    
    if sample_file:
        print(f"Loading reference submission from {sample_file}")
        reference_df = pd.read_csv(sample_file)
        
        # Create a mapping from ID to prediction
        id_to_pred = dict(zip(submission_df['ID'], test_pred))
        
        # Create a new dataframe with the reference IDs
        fixed_df = pd.DataFrame({'ID': reference_df['ID']})
        
        # Map predictions to the reference IDs
        fixed_df['Pred'] = fixed_df['ID'].map(id_to_pred)
        
        # Fill missing predictions with 0.5 (random chance)
        missing_count = fixed_df['Pred'].isna().sum()
        if missing_count > 0:
            print(f"Warning: {missing_count} IDs in reference not found in submission. Using 0.5 as prediction.")
            fixed_df['Pred'] = fixed_df['Pred'].fillna(0.5)
        
        print(f"Final fixed dataframe has {len(fixed_df)} rows")
        return fixed_df
    
    print("Warning: Could not find reference submission file. Returning original dataframe.")
    return submission_df

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
                    'Pred']  # Removed 'Season' from exclude_cols
    
    features = [c for c in games.columns if c not in exclude_cols and c in submission_df.columns]
    print(f"Using {len(features)} features for training, sample features: {features[:5]}")
    
    # Ensure Season column is available for model training if it exists in games
    if 'Season' in games.columns and 'Season' not in features:
        # Create a copy of X_train with Season column added
        X_train = games[features + ['Season']].fillna(0)
        
        # If submission data doesn't have Season, add a default season (e.g., 2025)
        if 'Season' not in submission_df.columns:
            X_sub = submission_df[features].fillna(0)
            X_sub['Season'] = 2025  # Use the current competition year
        else:
            X_sub = submission_df[features + ['Season']].fillna(0)
    else:
        X_train = games[features].fillna(0)
        X_sub = submission_df[features].fillna(0)
    
    y_train = games['WinA']
    
    return X_train, y_train, X_sub, features + (['Season'] if 'Season' in X_train.columns else []) 