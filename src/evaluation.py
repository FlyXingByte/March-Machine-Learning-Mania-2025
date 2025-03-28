import numpy as np
import pandas as pd
import os

def generate_submission(submission_df, test_pred, output_file='submission.csv'):
    """
    Generate final submission file with predictions
    
    Parameters:
        submission_df: DataFrame with submission format
        test_pred: Array of predictions
        output_file: Output filename for submission
        
    Returns:
        DataFrame containing ID and predictions
    """
    # Check if tracking columns exist
    has_tracking = 'original_index' in submission_df.columns and 'original_ID' in submission_df.columns
    
    if has_tracking:
        # Create a new DataFrame with original IDs and predictions
        # Sort by original index to ensure correct order
        if len(test_pred) != len(submission_df):
            print(f"Warning: Prediction count ({len(test_pred)}) does not match submission DataFrame size ({len(submission_df)})")
            
        # Create temporary DataFrame with index and predictions
        temp_df = pd.DataFrame({
            'original_index': submission_df['original_index'],
            'Pred': test_pred
        })
        # Sort by original index
        temp_df = temp_df.sort_values('original_index')
        
        # Create final submission with sorted predictions
        final_submission = pd.DataFrame({
            'ID': submission_df.sort_values('original_index')['original_ID'].values,
            'Pred': temp_df['Pred'].values
        })
        
        print(f"Created submission with {len(final_submission)} rows using tracking columns")
    else:
        # If tracking columns are not available, use fallback alignment method
        print("Warning: Tracking columns not found. Using fallback alignment method.")
        final_submission = fix_submission_format(submission_df, test_pred)
    
    # Save to file
    final_submission.to_csv(output_file, index=False)
    
    print_submission_stats(final_submission)
    
    return final_submission

def fix_submission_format(submission_df, test_pred):
    """
    通过调整DataFrame以匹配预期的行数来修复提交格式问题
    
    参数:
        submission_df: 包含提交数据的DataFrame
        test_pred: 预测数组
        
    返回:
        格式已更正的DataFrame
    """
    expected_counts = [264628, 131407]  # submission_64.csv和SampleSubmissionStage2.csv计数
    current_count = len(submission_df)
    
    if current_count in expected_counts:
        print(f"提交格式已正确，有{current_count}行")
        return submission_df
    
    print(f"修复提交格式。当前计数：{current_count}，预期：{expected_counts}")
    
    # 检查是否可以找到样本提交文件作为参考
    data_path = "Z:\\kaggle\\MMLM2025\\March-Machine-Learning-Mania-2025\\input\\march-machine-learning-mania-2025"
    sample_file = None
    
    if os.path.exists(os.path.join(data_path, "SampleSubmissionStage2.csv")):
        sample_file = os.path.join(data_path, "SampleSubmissionStage2.csv")
        print(f"使用SampleSubmissionStage2.csv作为参考")
    elif os.path.exists(os.path.join(data_path, "SampleSubmissionStage1.csv")):
        sample_file = os.path.join(data_path, "SampleSubmissionStage1.csv")
        print(f"使用SampleSubmissionStage1.csv作为参考")
    
    if sample_file:
        print(f"从{sample_file}加载参考提交")
        reference_df = pd.read_csv(sample_file)
        
        # 创建从ID到预测的映射
        id_to_pred = dict(zip(submission_df['ID'], test_pred))
        
        # 创建一个包含参考ID的新DataFrame
        fixed_df = pd.DataFrame({'ID': reference_df['ID']})
        
        # 将预测映射到参考ID
        fixed_df['Pred'] = fixed_df['ID'].map(id_to_pred)
        
        # 使用0.5（随机机会）填充缺失的预测
        missing_count = fixed_df['Pred'].isna().sum()
        if missing_count > 0:
            print(f"警告：参考中有{missing_count}个ID在提交中未找到。使用0.5作为预测。")
            fixed_df['Pred'] = fixed_df['Pred'].fillna(0.5)
        
        print(f"最终修复的DataFrame有{len(fixed_df)}行")
        return fixed_df
    
    print("警告：找不到参考提交文件。返回原始DataFrame。")
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
    # Exclude non-model input columns and raw statistics, but keep tracking columns
    exclude_cols = ['ID', 'DayNum', 'ST', 'Team1', 'Team2', 'IDTeams',
                    'IDTeam1', 'IDTeam2', 'WTeamID', 'WScore', 'LTeamID',
                    'LScore', 'NumOT', 'WinA', 'ScoreDiff', 'ScoreDiffNorm',
                    'WLoc', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA',
                    'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF',
                    'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR',
                    'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 'IDTeams_c_score',
                    'Pred']  # Removed 'Season' from exclude_cols
                             # DO NOT exclude 'original_index' and 'original_ID'
    
    # Print columns that are in both dataframes for debugging
    if submission_df is not None:
        common_columns = set(games.columns) & set(submission_df.columns)
        print(f"Common columns count: {len(common_columns)}")
        print(f"Sample common columns: {list(common_columns)[:5]}")
    
    # Make sure both dataframes have the same GenderCode column if present
    if 'GenderCode' in games.columns:
        print("GenderCode column found in training data")
    
    if submission_df is not None and 'GenderCode' in submission_df.columns:
        print("GenderCode column found in submission data")
    
    # Get common features that are present in both dataframes
    features = [c for c in games.columns if c not in exclude_cols and (submission_df is None or c in submission_df.columns)]
    
    if submission_df is not None:
        print(f"Using {len(features)} features for training, sample features: {features[:5]}")
    
    # Check for tracking columns
    tracking_cols = []
    if submission_df is not None:
        tracking_cols = [col for col in ['original_index', 'original_ID'] if col in submission_df.columns]
        if tracking_cols:
            print(f"Found tracking columns: {tracking_cols}")
    
    # Ensure Season column is available for model training if it exists in games
    if 'Season' in games.columns and 'Season' not in features:
        # Create a copy of X_train with Season column added
        X_train = games[features + ['Season']].fillna(0)
        
        # If submission data is provided
        if submission_df is not None:
            # Include tracking columns if they exist
            if tracking_cols:
                # If submission data doesn't have Season, add a default season (e.g., 2025)
                if 'Season' not in submission_df.columns:
                    X_sub = submission_df[features + tracking_cols].fillna(0)
                    X_sub['Season'] = 2025  # Use the current competition year
                else:
                    X_sub = submission_df[features + ['Season'] + tracking_cols].fillna(0)
            else:
                # No tracking columns
                if 'Season' not in submission_df.columns:
                    X_sub = submission_df[features].fillna(0)
                    X_sub['Season'] = 2025  # Use the current competition year
                else:
                    X_sub = submission_df[features + ['Season']].fillna(0)
        else:
            X_sub = None
    else:
        # No Season column
        if tracking_cols and submission_df is not None:
            X_train = games[features].fillna(0)
            X_sub = submission_df[features + tracking_cols].fillna(0) if submission_df is not None else None
        else:
            X_train = games[features].fillna(0)
            X_sub = submission_df[features].fillna(0) if submission_df is not None else None
    
    # Print shape information for debugging
    print(f"Training features shape: {X_train.shape}")
    if X_sub is not None:
        print(f"Submission features shape: {X_sub.shape}")
    
    # Verify both datasets have the same number of columns (except tracking columns)
    if X_sub is not None:
        train_cols = set(X_train.columns)
        sub_cols = set(X_sub.columns) - set(tracking_cols)
        missing_in_train = sub_cols - train_cols
        missing_in_sub = train_cols - sub_cols
        
        if missing_in_train:
            print(f"WARNING: {len(missing_in_train)} columns in submission data missing in training data: {missing_in_train}")
        
        if missing_in_sub:
            print(f"WARNING: {len(missing_in_sub)} columns in training data missing in submission data: {missing_in_sub}")
    
    y_train = games['WinA']
    
    return X_train, y_train, X_sub, features + (['Season'] if 'Season' in X_train.columns else [])

def evaluate_model(X_train, y_train, X_eval, y_eval, features):
    """
    Evaluate model performance on the evaluation dataset
    
    Args:
        X_train: Training features
        y_train: Training target
        X_eval: Evaluation features
        y_eval: Evaluation target
        features: List of feature names
        
    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
    import models  # Import here to avoid circular imports
    
    print("Evaluating model performance on 2024 tournament data...")
    
    # Train model using the same approach as for predictions
    pred_eval = models.stacking_ensemble_cv(X_train, y_train, X_eval, features, verbose=1)
    
    # Calculate metrics
    metrics = {
        'log_loss': log_loss(y_eval, pred_eval),
        'accuracy': accuracy_score(y_eval, pred_eval > 0.5),
        'roc_auc': roc_auc_score(y_eval, pred_eval)
    }
    
    # Calculate accuracy at different prediction thresholds
    thresholds = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    for thresh in thresholds:
        high_conf_mask = (pred_eval >= thresh) | (pred_eval <= (1-thresh))
        if high_conf_mask.sum() > 0:
            high_conf_acc = accuracy_score(
                y_eval[high_conf_mask], 
                pred_eval[high_conf_mask] > 0.5
            )
            metrics[f'accuracy_{thresh}'] = high_conf_acc
            metrics[f'coverage_{thresh}'] = high_conf_mask.mean()
    
    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"Log Loss: {metrics['log_loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    print("\nAccuracy at different confidence thresholds:")
    for thresh in thresholds:
        if f'accuracy_{thresh}' in metrics:
            coverage = metrics[f'coverage_{thresh}'] * 100
            acc = metrics[f'accuracy_{thresh}'] * 100
            print(f"  Threshold {thresh}: {acc:.1f}% accuracy ({coverage:.1f}% of predictions)")
    
    # Create a DataFrame with predicted probabilities and actual outcomes
    eval_df = pd.DataFrame({
        'Prediction': pred_eval,
        'Actual': y_eval
    })
    
    # Analyze model performance by prediction range
    print("\nPerformance by prediction range:")
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    
    eval_df['bin'] = pd.cut(eval_df['Prediction'], bins=bins, labels=bin_labels)
    bin_analysis = eval_df.groupby('bin').agg(
        count=('Actual', 'count'),
        actual_win_rate=('Actual', 'mean')
    ).reset_index()
    
    for _, row in bin_analysis.iterrows():
        bin_range = row['bin']
        count = row['count']
        win_rate = row['actual_win_rate'] * 100
        print(f"  {bin_range}: {count} games, {win_rate:.1f}% actual win rate")
    
    return metrics

def save_evaluation_results(metrics, output_file):
    """
    Save evaluation results to CSV file
    
    Args:
        metrics: Dictionary with evaluation metrics
        output_file: Path to output file
    """
    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame([metrics])
    
    # Save to CSV
    metrics_df.to_csv(output_file, index=False)
    
    print(f"Evaluation metrics saved to {output_file}")