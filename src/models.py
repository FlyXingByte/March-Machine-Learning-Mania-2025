import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit

def safe_column_selection(df, column_name):
    """
    Safely select a column from a DataFrame, handling the case where
    the selection returns a DataFrame instead of a Series.
    """
    col_data = df[column_name]
    if isinstance(col_data, pd.DataFrame):
        print(f"  Safe column selection: Column '{column_name}' returned a DataFrame. Using first column.")
        col_data = col_data.iloc[:, 0]
    return col_data

def get_base_models():
    """
    Define a list of base models for the ensemble.
    Note: Ridge regression is replaced by a calibrated Ridge classifier.
    
    Returns:
        List of initialized model objects.
    """
    model_xgb = XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                              n_estimators=300, random_state=42)
    model_lr = LogisticRegression(C=1, max_iter=1000, random_state=42, solver='liblinear')
    # Replace Ridge regression with a calibrated Ridge classifier for classification
    model_ridge = CalibratedClassifierCV(RidgeClassifier(random_state=42), cv=3)
    model_et = ExtraTreesClassifier(n_estimators=100, random_state=42)
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_cat = CatBoostClassifier(iterations=300, verbose=0, random_seed=42)
    
    # Add LightGBM model
    model_lgb = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
        importance_type='gain'
    )
    
    # Add PyTorch MLP model (input_size will be set during fit)
    model_mlp = PyTorchMLPWrapper(
        input_size=None,  # Will be set during fit
        hidden_size=128,
        dropout_rate=0.3,
        lr=0.001,
        batch_size=64,
        epochs=20,  # Reduced epochs for faster training
        patience=5,  # Early stopping patience
        random_state=42
    )
    
    # Add new models for ensemble diversity
    
    # Gradient Boosting Classifier
    model_gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
    
    # Histogram-based Gradient Boosting (faster version of GBM)
    model_hgb = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=10,
        random_state=42
    )
    
    # AdaBoost Classifier
    model_ada = AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
    
    # Support Vector Machine with probability calibration
    model_svm = CalibratedClassifierCV(
        SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
        cv=3
    )
    
    # K-Nearest Neighbors
    model_knn = KNeighborsClassifier(
        n_neighbors=15,
        weights='distance'
    )
    
    # Gaussian Naive Bayes
    model_gnb = GaussianNB()
    
    # Scikit-learn MLP (Neural Network)
    model_sklearn_mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=200,
        early_stopping=True,
        random_state=42
    )
    
    # SGD Classifier with calibration
    model_sgd = CalibratedClassifierCV(
        SGDClassifier(loss='log_loss', penalty='elasticnet', 
                     alpha=0.0001, l1_ratio=0.15,
                     max_iter=1000, random_state=42),
        cv=3
    )
    
    # Decision Tree Classifier
    model_dt = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # Add the new deep neural network model
    model_deep_nn = PyTorchDeepNNWrapper(
        input_size=None,  # Will be set during fit
        hidden_sizes=[256, 128, 64],
        dropout_rates=[0.3, 0.3, 0.3],
        lr=0.001,
        batch_size=64,
        epochs=25,  # Reduced epochs for faster training
        patience=6,  # Early stopping patience
        random_state=42
    )
    
    # Return all models
    return [
        model_xgb, model_lr, model_ridge, model_et, model_rf, model_cat, model_lgb, model_mlp,
        model_gb, model_hgb, model_ada, model_svm, model_knn, model_gnb, model_sklearn_mlp, model_sgd, model_dt,
        model_deep_nn
    ]

def print_feature_importance(models, features):
    """
    Print feature importance for models that expose this attribute.
    
    Args:
        models: List of trained model objects.
        features: List of feature names.
    """
    for model in models:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            print(f"\n{model.__class__.__name__} Feature Importance:")
            for i in range(min(10, len(features))):
                print(f"  {features[indices[i]]}: {importances[indices[i]]:.4f}")

def flatten_dataframe_columns(df):
    """
    Flattens multi-level column names that can come from imputation or feature engineering
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
    
    # Also ensure each column in the DataFrame is a Series (not a DataFrame)
    return pd.DataFrame({col: (val if not isinstance(val, pd.DataFrame) else val.iloc[:, 0])
                         for col, val in df.items()}, index=df.index)

def align_feature_columns(train_df, test_df, model_features):
    """
    Ensure that both training and test dataframes have the same feature columns
    in the same order, adding missing columns with zeros as needed.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        model_features: List of feature names to use for modeling
        
    Returns:
        Tuple of aligned dataframes (train_df, test_df) and updated model_features
    """
    print(f"Aligning features between training and test data...")
    
    # Get a complete list of all unique features from both datasets
    all_features = sorted(list(set(train_df.columns) | set(test_df.columns)))
    
    # Start with model features
    valid_features = []
    for feat in model_features:
        # Track features that exist in both datasets or can be added
        if feat in train_df.columns and feat in test_df.columns:
            valid_features.append(feat)
        elif feat in train_df.columns:
            # Feature only in training data, add to test with zeros
            print(f"  Adding missing feature to test data: {feat}")
            test_df[feat] = 0
            valid_features.append(feat)
        elif feat in test_df.columns:
            # Feature only in test data, add to training with zeros
            print(f"  Adding missing feature to training data: {feat}")
            train_df[feat] = 0
            valid_features.append(feat)
    
    # Ensure both dataframes have exactly the same columns in the same order
    train_columns = set(train_df.columns)
    test_columns = set(test_df.columns)
    
    # Find features that exist in one dataframe but not the other
    train_only = train_columns - test_columns
    test_only = test_columns - train_columns
    
    # Add missing features to each dataframe
    for feat in train_only:
        if feat not in valid_features and feat in model_features:
            print(f"  Adding missing feature to test data: {feat}")
            test_df[feat] = 0
            valid_features.append(feat)
    
    for feat in test_only:
        if feat not in valid_features and feat in model_features:
            print(f"  Adding missing feature to training data: {feat}")
            train_df[feat] = 0
            valid_features.append(feat)
    
    # Now select only the valid features, ensuring both dataframes have exactly the same columns in the same order
    print(f"  After alignment: {len(valid_features)} valid features for modeling")
    
    # Verify shape consistency before returning
    train_subset = train_df[valid_features].copy()
    test_subset = test_df[valid_features].copy()
    
    print(f"  Train shape: {train_subset.shape}, Test shape: {test_subset.shape}")
    
    # Double-check consistency before returning
    if set(train_subset.columns) != set(test_subset.columns):
        print("WARNING: Column mismatch detected! Fixing...")
        # Get a definitive shared list of columns
        common_cols = sorted(list(set(train_subset.columns) & set(test_subset.columns)))
        # Only use the common columns
        train_subset = train_subset[common_cols]
        test_subset = test_subset[common_cols]
        valid_features = common_cols
        print(f"  After fixing: Train shape: {train_subset.shape}, Test shape: {test_subset.shape}")
    
    return train_subset, test_subset, valid_features

def stacking_ensemble_cv(X_train, y_train, X_test, features, verbose=1, use_extended_models=True):
    """
    Stacking ensemble with cross-validation.
    Trains base models using time-series CV, then trains a meta-model on out-of-fold predictions.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        features: List of features to use for training
        verbose: Whether to print detailed output (0=minimal, 1=normal, 2=detailed)
        use_extended_models: Whether to use extended model set
        
    Returns:
        Numpy array of predictions for X_test
    """
    if verbose:
        print("\nStacking ensemble with time-series cross-validation")
        print(f"Input data shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}")
        print(f"Using {len(features)} features")
    
    # Copy data to avoid modifying the original
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    # 重要修复：重置索引以避免索引超出范围错误
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    
    # Set up cross-validation with time series split
    if 'Season' in X_train.columns:
        # Safely get Season column
        season_col = safe_column_selection(X_train, 'Season')
        years = sorted(season_col.unique())
        if verbose:
            print(f"Available seasons: {years}")
            print(f"Performing time series cross-validation:")
        
        # 修改交叉验证分割创建方式
        splits = []
        for i in range(len(years)-1):
            train_years = years[:i+1]
            val_year = years[i+1]
            if verbose:
                print(f"  Split {i+1}: Train on {train_years}, validate on {val_year}")
            
            # 使用布尔掩码而不是直接索引
            train_mask = season_col.isin(train_years)
            val_mask = season_col == val_year
            
            # 只有当验证集不为空时才添加这个分割
            if val_mask.any():
                splits.append((np.where(train_mask)[0], np.where(val_mask)[0]))
            else:
                print(f"  Warning: No validation data for year {val_year}, skipping this split")
    else:
        # If no Season column, use default TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        splits = list(tscv.split(X_train))
        if verbose:
            print("Using default TimeSeriesSplit without Season column")
    
    # Prepare for meta-model training
    meta_features_all = []
    meta_labels_all = []
    test_meta_features_all = []
    cv_scores = []
    model_performances = []  # To track model performance across folds
    
    # Remove original_index and original_ID from features
    model_features = [f for f in features if f not in ['original_index', 'original_ID']]
    
    # Align features between training and test sets
    X_train, X_test, model_features = align_feature_columns(X_train, X_test, model_features)
    
    # Process by fold
    for fold, (train_idx, val_idx) in enumerate(splits):
        if verbose:
            print(f"\nProcessing fold {fold+1}/{len(splits)}")
        
        # 添加安全检查，确保索引在范围内
        if np.max(train_idx) >= len(X_train) or np.max(val_idx) >= len(X_train):
            print(f"  Warning: Indices out of bounds in fold {fold+1}. Max train index: {np.max(train_idx)}, Max val index: {np.max(val_idx)}, X_train length: {len(X_train)}")
            print("  Skipping this fold")
            continue
        
        # Split data for current fold
        train_fold = X_train.iloc[train_idx].copy()
        val_fold = X_train.iloc[val_idx].copy()
        train_y = y_train.iloc[train_idx]
        val_y = y_train.iloc[val_idx]
        X_test_fold = X_test.copy()
        
        # Flatten multi-level columns if present
        train_fold = flatten_dataframe_columns(train_fold)
        val_fold = flatten_dataframe_columns(val_fold)
        X_test_fold = flatten_dataframe_columns(X_test_fold)
        
        # One-hot encode 'GameType' if exists
        if 'GameType' in train_fold.columns:
            if verbose:
                print("One-hot encoding 'GameType' feature for training, validation and test sets.")
            
            # Safely select GameType column
            train_game_type = safe_column_selection(train_fold, 'GameType')
            val_game_type = safe_column_selection(val_fold, 'GameType')
            test_game_type = safe_column_selection(X_test_fold, 'GameType')
            
            train_dummies = pd.get_dummies(train_game_type, prefix='GameType')
            val_dummies = pd.get_dummies(val_game_type, prefix='GameType')
            test_dummies = pd.get_dummies(test_game_type, prefix='GameType')
            
            all_dummy_cols = sorted(set(train_dummies.columns).union(val_dummies.columns).union(test_dummies.columns))
            train_dummies = train_dummies.reindex(columns=all_dummy_cols, fill_value=0)
            val_dummies = val_dummies.reindex(columns=all_dummy_cols, fill_value=0)
            test_dummies = test_dummies.reindex(columns=all_dummy_cols, fill_value=0)
            
            train_fold = pd.concat([train_fold.drop(columns=['GameType']), train_dummies], axis=1)
            val_fold = pd.concat([val_fold.drop(columns=['GameType']), val_dummies], axis=1)
            X_test_fold = pd.concat([X_test_fold.drop(columns=['GameType']), test_dummies], axis=1)
            
            if 'GameType' in model_features:
                model_features.remove('GameType')
            model_features.extend(all_dummy_cols)
        
        # Convert object type columns to numeric if possible
        features_to_remove = []
        for col in model_features:
            try:
                # Safely select columns
                train_col = safe_column_selection(train_fold, col)
                val_col = safe_column_selection(val_fold, col)
                test_col = safe_column_selection(X_test_fold, col)
                
                # Now check the dtype
                if (np.issubdtype(train_col.dtype, np.object_) or 
                    np.issubdtype(val_col.dtype, np.object_) or 
                    np.issubdtype(test_col.dtype, np.object_)):
                    
                    if verbose:
                        print(f"  Converting column {col} from {train_col.dtype} to numeric")
                        print(f"    Sample values: {train_col.iloc[:3].tolist()}")
                    
                    try:
                        train_fold[col] = pd.to_numeric(train_col, errors='coerce')
                        val_fold[col] = pd.to_numeric(val_col, errors='coerce')
                        X_test_fold[col] = pd.to_numeric(test_col, errors='coerce')
                    except Exception as e:
                        print(f"  Warning: Cannot convert feature {col} to numeric - {e}")
                        features_to_remove.append(col)
            except Exception as e:
                print(f"  Warning: Error processing column {col} - {e}")
                features_to_remove.append(col)
        for col in features_to_remove:
            if col in model_features:
                model_features.remove(col)
                print(f"  Removed non-numeric feature: {col}")
        
        if len(model_features) == 0:
            print("Error: No numeric features available for training!")
            return np.zeros(X_test.shape[0])
        
        # Imputation: median with missing indicator - only using regular season data
        # Filter training data to only include regular season games for imputation
        train_fold_regular = train_fold[train_fold.get('GameType', '') != 'Tournament'] if 'GameType' in train_fold.columns else train_fold
        
        imputer = SimpleImputer(strategy='median', add_indicator=True)
        # Fit imputer only on regular season data
        imputer.fit(train_fold_regular[model_features])
        # Transform using the fitted imputer
        train_imputed = imputer.transform(train_fold[model_features])
        val_imputed = imputer.transform(val_fold[model_features])
        test_imputed = imputer.transform(X_test_fold[model_features])
        if imputer.add_indicator:
            indicator_features = [f"missing_{model_features[i]}" for i in imputer.indicator_.features_]
            new_columns = model_features + indicator_features
        else:
            new_columns = model_features
        
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_imputed)
        val_scaled = scaler.transform(val_imputed)
        test_scaled = scaler.transform(test_imputed)
        
        # Ensure all dataframes have the exact same columns
        try:
            train_norm = pd.DataFrame(train_scaled, columns=new_columns, index=train_fold.index)
            val_norm = pd.DataFrame(val_scaled, columns=new_columns, index=val_fold.index)
            test_norm = pd.DataFrame(test_scaled, columns=new_columns, index=X_test_fold.index)
        except ValueError as e:
            print(f"ERROR during DataFrame creation: {e}")
            print(f"Shapes: train_scaled={train_scaled.shape}, val_scaled={val_scaled.shape}, test_scaled={test_scaled.shape}")
            print(f"new_columns length: {len(new_columns)}")
            
            # Fix: Ensure shapes match by truncating columns if necessary
            column_count = min(train_scaled.shape[1], val_scaled.shape[1], test_scaled.shape[1], len(new_columns))
            print(f"Using {column_count} columns (truncated) to ensure consistency")
            
            # Use only the first 'column_count' columns
            truncated_columns = new_columns[:column_count]
            train_norm = pd.DataFrame(train_scaled[:, :column_count], columns=truncated_columns, index=train_fold.index)
            val_norm = pd.DataFrame(val_scaled[:, :column_count], columns=truncated_columns, index=val_fold.index)
            test_norm = pd.DataFrame(test_scaled[:, :column_count], columns=truncated_columns, index=X_test_fold.index)
            
            print(f"After fixing: train_norm={train_norm.shape}, val_norm={val_norm.shape}, test_norm={test_norm.shape}")
            
        # Flatten any multi-level columns
        train_norm = flatten_dataframe_columns(train_norm)
        val_norm = flatten_dataframe_columns(val_norm)
        test_norm = flatten_dataframe_columns(test_norm)
        
        # Train base models and collect meta features (out-of-fold predictions)
        if use_extended_models:
            print("Using extended model set with model variants for greater diversity")
            base_models = get_extended_models()
        else:
            base_models = get_base_models()
            
        val_preds = []
        test_preds = []
        fold_model_performances = []  # Track performance for this fold
        
        for i, model in enumerate(base_models):
            model_name = model.__class__.__name__
            if verbose:
                print(f"  Training base model {i+1}/{len(base_models)}: {model_name}")
            try:
                model.fit(train_norm, train_y)
            except ValueError as e:
                if "3-fold" in str(e):
                    print(f"Warning: Not enough samples for calibration in {model_name}, using uncalibrated base estimator for this fold.")
                    base_est = model.estimator  
                    base_est.fit(train_norm, train_y)
                    if hasattr(base_est, "predict_proba"):
                        val_pred = base_est.predict_proba(val_norm)[:, 1]
                        test_pred = base_est.predict_proba(test_norm)[:, 1]
                    elif hasattr(base_est, "decision_function"):
                        val_dec = base_est.decision_function(val_norm)
                        test_dec = base_est.decision_function(test_norm)
                        val_pred = 1/(1+np.exp(-val_dec))
                        test_pred = 1/(1+np.exp(-test_dec))
                    else:
                        val_pred = np.full(val_norm.shape[0], 0.5)
                        test_pred = np.full(test_norm.shape[0], 0.5)
                    val_preds.append(val_pred)
                    test_preds.append(test_pred)
                    
                    # Calculate model performance (Brier score)
                    model_brier = brier_score_loss(val_y, val_pred)
                    fold_model_performances.append((i, model_name, model_brier))
                    continue
                else:
                    raise
            try:
                if hasattr(model, 'predict_proba'):
                    val_pred = model.predict_proba(val_norm)[:, 1]
                    test_pred = model.predict_proba(test_norm)[:, 1]
                else:
                    val_pred = model.predict(val_norm)
                    test_pred = model.predict(test_norm)
                    val_pred = np.clip((val_pred - val_pred.min()) / (val_pred.max() - val_pred.min() + 1e-8), 0, 1)
                    test_pred = np.clip((test_pred - test_pred.min()) / (test_pred.max() - test_pred.min() + 1e-8), 0, 1)
            except Exception as e:
                val_pred = model.predict(val_norm)
                test_pred = model.predict(test_norm)
                val_pred = np.clip((val_pred - val_pred.min()) / (val_pred.max() - val_pred.min() + 1e-8), 0, 1)
                test_pred = np.clip((test_pred - test_pred.min()) / (test_pred.max() - test_pred.min() + 1e-8), 0, 1)
            
            val_preds.append(val_pred)
            test_preds.append(test_pred)
            
            # Calculate model performance (Brier score)
            model_brier = brier_score_loss(val_y, val_pred)
            fold_model_performances.append((i, model_name, model_brier))
            
            if verbose:
                print(f"    {model_name} Brier score: {model_brier:.4f}")
        
        # Store model performances for this fold
        model_performances.append(fold_model_performances)
        
        # Construct meta features for the current fold
        meta_X_val_fold = np.column_stack(val_preds)
        meta_features_all.append(meta_X_val_fold)
        meta_labels_all.append(val_y.to_numpy())
        test_meta_features_all.append(np.column_stack(test_preds))
        
        # Optionally, report Brier score on current fold using a temporary meta-model
        temp_meta = LogisticRegression(C=1, max_iter=1000, random_state=42, solver='liblinear')
        temp_meta.fit(meta_X_val_fold, val_y)
        fold_brier = brier_score_loss(val_y, temp_meta.predict_proba(meta_X_val_fold)[:, 1])
        cv_scores.append(fold_brier)
        if verbose:
            print(f"  Current fold temporary meta-model Brier score: {fold_brier:.4f}")
    
    # Aggregate out-of-fold meta features and labels if any fold was processed
    if len(meta_features_all) == 0:
        print("Warning: No valid folds were processed in time-series CV. Using default prediction 0.5.")
        return np.full(X_test.shape[0], 0.5)
    
    meta_X_train = np.vstack(meta_features_all)
    meta_y_train = np.concatenate(meta_labels_all)
    # Average test meta features across folds
    meta_test = np.mean(np.array(test_meta_features_all), axis=0)
    
    # Calculate average model performance across folds
    avg_model_performance = {}
    for fold_performances in model_performances:
        for idx, name, score in fold_performances:
            if name not in avg_model_performance:
                avg_model_performance[name] = []
            avg_model_performance[name].append(score)
    
    # Calculate average Brier score for each model
    for name, scores in avg_model_performance.items():
        avg_model_performance[name] = np.mean(scores)
    
    if verbose:
        print("\nAverage model performance across folds:")
        for name, avg_score in sorted(avg_model_performance.items(), key=lambda x: x[1]):
            print(f"  {name}: {avg_score:.4f}")
    
    # Use weighted meta-model based on model performance
    # Convert Brier scores to weights (lower Brier score = higher weight)
    model_weights = {}
    for name, score in avg_model_performance.items():
        # Invert and normalize the Brier score (lower is better)
        model_weights[name] = 1.0 / (score + 1e-8)
    
    # Normalize weights to sum to 1
    total_weight = sum(model_weights.values())
    for name in model_weights:
        model_weights[name] /= total_weight
    
    if verbose:
        print("\nModel weights for ensemble:")
        for name, weight in sorted(model_weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {weight:.4f}")
    
    # Train final meta-model on full out-of-fold predictions
    meta_model = LogisticRegression(C=1, max_iter=1000, random_state=42, solver='liblinear')
    if verbose:
        print("Training final meta-model (Logistic Regression) on out-of-fold predictions")
    meta_model.fit(meta_X_train, meta_y_train)
    final_meta_brier = brier_score_loss(meta_y_train, meta_model.predict_proba(meta_X_train)[:, 1])
    if verbose:
        print(f"Out-of-fold meta-model Brier score: {final_meta_brier:.4f}")
        print("Final meta-model coefficients:")
        models_to_display = get_base_models() if not use_extended_models else get_extended_models()
        for i, coef in enumerate(meta_model.coef_[0]):
            if i < len(models_to_display):
                model_name = models_to_display[i].__class__.__name__
                print(f"    {model_name}: {coef:.4f}")
            else:
                print(f"    Model_{i}: {coef:.4f}")
    
    # Predict test set using final meta-model
    test_pred_ensemble = meta_model.predict_proba(meta_test)[:, 1]
    
    # Also create a weighted average prediction based on model performance
    weighted_test_pred = np.zeros(meta_test.shape[0])
    models_for_weights = get_base_models() if not use_extended_models else get_extended_models()
    for i, model in enumerate(models_for_weights):
        if i < meta_test.shape[1]:  # Make sure we don't exceed the number of columns
            model_name = model.__class__.__name__
            if model_name in model_weights:
                weighted_test_pred += meta_test[:, i] * model_weights[model_name]
    
    # Blend the meta-model prediction with the weighted average (70% meta-model, 30% weighted average)
    final_blend_pred = 0.7 * test_pred_ensemble + 0.3 * weighted_test_pred
    test_pred_ensemble = final_blend_pred
    
    if len(cv_scores) > 0:
        print(f"\n[Stacking] Average temporary fold Brier Score: {np.mean(cv_scores):.4f}")
    
    final_test_pred = spread_predictions(test_pred_ensemble)
    return final_test_pred

def spread_predictions(preds, spread_factor=1.5):
    """
    Spread predictions using a sigmoid transformation for diversity.
    
    Args:
        preds: Array of predictions.
        spread_factor: Factor to control spreading intensity.
        
    Returns:
        Array of spread predictions.
    """
    logits = np.log(preds / (1 - preds + 1e-8))
    spread_logits = logits * spread_factor
    return 1 / (1 + np.exp(-spread_logits))

# PyTorch MLP model class
class PyTorchMLP(nn.Module):
    def __init__(self, input_size, hidden_size=32, dropout_rate=0.2):
        super(PyTorchMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x):
        return self.model(x).squeeze()

# PyTorch wrapper for scikit-learn compatibility
class PyTorchMLPWrapper:
    def __init__(self, input_size, hidden_size=32, dropout_rate=0.2, 
                 lr=0.001, batch_size=64, epochs=20, patience=5, random_state=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
    def fit(self, X, y):
        # Set random seeds for reproducibility
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Initialize model
        self.input_size = X.shape[1]
        self.model = PyTorchMLP(self.input_size, self.hidden_size, self.dropout_rate)
        
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X)
        y_tensor = torch.FloatTensor(y.values if hasattr(y, 'values') else y)
        
        # Split data for early stopping
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_tensor, y_tensor, test_size=0.2, random_state=self.random_state
        )
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            total_train_loss = 0
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    total_val_loss += loss.item()
            
            # Print progress every 5 epochs
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Train Loss: {total_train_loss/len(train_loader):.4f}, Val Loss: {total_val_loss/len(val_loader):.4f}')
            
            # Early stopping check
            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                # Save best model state
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    # Restore best model
                    self.model.load_state_dict(best_model_state)
                    break
        
        return self
    
    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X)
        
        # Prediction
        self.model.eval()
        with torch.no_grad():
            X_tensor = X_tensor.to(self.device)
            y_pred = self.model(X_tensor).cpu().numpy()
        
        # Return in the format expected by sklearn (with two columns: [1-p, p])
        return np.column_stack((1 - y_pred, y_pred))
    
    def predict(self, X):
        return self.predict_proba(X)[:, 1]

class PyTorchDeepNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[64, 32, 16], dropout_rates=[0.2, 0.2, 0.2]):
        """
        A deeper neural network with multiple hidden layers, batch normalization, and residual connections.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            dropout_rates: List of dropout rates for each hidden layer
        """
        super(PyTorchDeepNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rates = dropout_rates
        
        # Input layer
        layers = [nn.Linear(input_size, hidden_sizes[0])]
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rates[0]))
        
        # Hidden layers with residual connections where possible
        for i in range(1, len(hidden_sizes)):
            # Main path
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rates[i]))
            
            # Add residual connection if dimensions match
            if hidden_sizes[i-1] == hidden_sizes[i]:
                # Identity residual connection
                self.has_residual = True
            else:
                self.has_residual = False
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        layers.append(nn.Sigmoid())
        
        self.main_path = nn.ModuleList(layers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x):
        # Process through layers with residual connections
        layer_input = x
        residual = None
        layer_idx = 0
        
        for i, layer in enumerate(self.main_path[:-2]):  # Exclude final Linear and Sigmoid
            if isinstance(layer, nn.Linear):
                if i > 0 and self.has_residual and residual is not None:
                    # Apply residual connection
                    layer_input = layer_input + residual
                
                # Process through linear layer
                layer_output = layer(layer_input)
                
                # Store for potential residual connection
                if layer_idx < len(self.hidden_sizes) - 1:
                    residual = layer_output
                
                layer_input = layer_output
                layer_idx += 1
            else:
                # Process through non-linear layers
                layer_input = layer(layer_input)
        
        # Final output layers
        output = self.main_path[-2](layer_input)  # Linear
        output = self.main_path[-1](output)       # Sigmoid
        
        return output.squeeze()

class PyTorchDeepNNWrapper:
    def __init__(self, input_size=None, hidden_sizes=[64, 32, 16], dropout_rates=[0.2, 0.2, 0.2],
                 lr=0.001, batch_size=64, epochs=30, patience=7, random_state=42):
        """
        Wrapper for PyTorchDeepNN to make it compatible with scikit-learn.
        
        Args:
            input_size: Number of input features (will be set during fit if None)
            hidden_sizes: List of hidden layer sizes
            dropout_rates: List of dropout rates for each hidden layer
            lr: Learning rate
            batch_size: Batch size for training
            epochs: Maximum number of epochs
            patience: Early stopping patience
            random_state: Random seed for reproducibility
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rates = dropout_rates
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
    def fit(self, X, y):
        # Set random seeds for reproducibility
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Initialize model
        self.input_size = X.shape[1]
        self.model = PyTorchDeepNN(self.input_size, self.hidden_sizes, self.dropout_rates)
        
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X)
        y_tensor = torch.FloatTensor(y.values if hasattr(y, 'values') else y)
        
        # Split data for early stopping
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_tensor, y_tensor, test_size=0.2, random_state=self.random_state
        )
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            total_train_loss = 0
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    total_val_loss += loss.item()
            
            # Update learning rate scheduler
            scheduler.step(total_val_loss)
            
            # Print progress every 5 epochs
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Train Loss: {total_train_loss/len(train_loader):.4f}, Val Loss: {total_val_loss/len(val_loader):.4f}')
            
            # Early stopping check
            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                # Save best model state
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    # Restore best model
                    self.model.load_state_dict(best_model_state)
                    break
        
        return self
    
    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X)
        
        # Prediction
        self.model.eval()
        with torch.no_grad():
            X_tensor = X_tensor.to(self.device)
            y_pred = self.model(X_tensor).cpu().numpy()
        
        # Return in the format expected by sklearn (with two columns: [1-p, p])
        return np.column_stack((1 - y_pred, y_pred))
    
    def predict(self, X):
        return self.predict_proba(X)[:, 1]

def create_custom_ensemble():
    """
    Create a custom ensemble model that combines multiple models with different hyperparameters.
    This provides additional diversity to the ensemble beyond the base models.
    
    Returns:
        List of initialized model objects with varied hyperparameters.
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
    
    custom_models = []
    
    # XGBoost variants
    xgb_variants = [
        XGBClassifier(objective='binary:logistic', eval_metric='logloss', n_estimators=200, 
                     learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, 
                     random_state=42),
        XGBClassifier(objective='binary:logistic', eval_metric='logloss', n_estimators=300, 
                     learning_rate=0.03, max_depth=5, subsample=0.7, colsample_bytree=0.7, 
                     random_state=43),
        XGBClassifier(objective='binary:logistic', eval_metric='logloss', n_estimators=400, 
                     learning_rate=0.01, max_depth=6, subsample=0.9, colsample_bytree=0.9, 
                     random_state=44)
    ]
    custom_models.extend(xgb_variants)
    
    # LightGBM variants
    lgb_variants = [
        lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, max_depth=5,
                          subsample=0.8, colsample_bytree=0.8, random_state=42),
        lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=63, max_depth=7,
                          subsample=0.7, colsample_bytree=0.7, random_state=43),
        lgb.LGBMClassifier(n_estimators=400, learning_rate=0.01, num_leaves=127, max_depth=9,
                          subsample=0.9, colsample_bytree=0.9, random_state=44)
    ]
    custom_models.extend(lgb_variants)
    
    # CatBoost variants
    cat_variants = [
        CatBoostClassifier(iterations=200, learning_rate=0.05, depth=4, verbose=0, random_seed=42),
        CatBoostClassifier(iterations=300, learning_rate=0.03, depth=6, verbose=0, random_seed=43),
        CatBoostClassifier(iterations=400, learning_rate=0.01, depth=8, verbose=0, random_seed=44)
    ]
    custom_models.extend(cat_variants)
    
    # Random Forest variants
    rf_variants = [
        RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42),
        RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=10, random_state=43),
        RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2, random_state=44)
    ]
    custom_models.extend(rf_variants)
    
    # Extra Trees variants
    et_variants = [
        ExtraTreesClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42),
        ExtraTreesClassifier(n_estimators=200, max_depth=15, min_samples_split=10, random_state=43),
        ExtraTreesClassifier(n_estimators=300, max_depth=None, min_samples_split=2, random_state=44)
    ]
    custom_models.extend(et_variants)
    
    # Gradient Boosting variants
    gb_variants = [
        GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=43),
        GradientBoostingClassifier(n_estimators=300, learning_rate=0.01, max_depth=5, random_state=44)
    ]
    custom_models.extend(gb_variants)
    
    # PyTorch MLP variants
    mlp_variants = [
        PyTorchMLPWrapper(input_size=None, hidden_size=64, dropout_rate=0.2, lr=0.001, 
                         batch_size=32, epochs=15, patience=5, random_state=42),
        PyTorchMLPWrapper(input_size=None, hidden_size=128, dropout_rate=0.3, lr=0.0005, 
                         batch_size=64, epochs=20, patience=5, random_state=43),
        PyTorchMLPWrapper(input_size=None, hidden_size=256, dropout_rate=0.4, lr=0.0001, 
                         batch_size=128, epochs=25, patience=5, random_state=44)
    ]
    custom_models.extend(mlp_variants)
    
    # Deep NN variants
    deep_nn_variants = [
        PyTorchDeepNNWrapper(input_size=None, hidden_sizes=[128, 64], dropout_rates=[0.2, 0.2],
                            lr=0.001, batch_size=64, epochs=20, patience=5, random_state=42),
        PyTorchDeepNNWrapper(input_size=None, hidden_sizes=[256, 128, 64], dropout_rates=[0.3, 0.3, 0.3],
                            lr=0.0005, batch_size=64, epochs=25, patience=6, random_state=43),
        PyTorchDeepNNWrapper(input_size=None, hidden_sizes=[512, 256, 128, 64], dropout_rates=[0.4, 0.4, 0.4, 0.4],
                            lr=0.0001, batch_size=64, epochs=30, patience=7, random_state=44)
    ]
    custom_models.extend(deep_nn_variants)
    
    return custom_models

def get_extended_models():
    """
    Get an extended set of models for the ensemble, including base models and custom ensemble models.
    
    Returns:
        List of initialized model objects.
    """
    # Get base models
    base_models = get_base_models()
    
    # Get custom ensemble models
    custom_models = create_custom_ensemble()
    
    # Combine all models
    all_models = base_models + custom_models
    
    return all_models