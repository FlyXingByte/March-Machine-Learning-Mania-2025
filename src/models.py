import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def get_base_models():
    """
    Define a list of base models for the ensemble
    
    Returns:
        List of initialized model objects
    """
    model_xgb = XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                              n_estimators=300, use_label_encoder=False, random_state=42)
    model_lr = LogisticRegression(C=1, max_iter=1000, random_state=42, solver='liblinear')
    model_ridge = Ridge(alpha=1.0, random_state=42)
    model_et = ExtraTreesClassifier(n_estimators=100, random_state=42)
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_cat = CatBoostClassifier(iterations=300, verbose=0, random_seed=42)
    return [model_xgb, model_lr, model_ridge, model_et, model_rf, model_cat]

def print_feature_importance(models, features):
    """
    Print feature importance for models that expose this attribute
    
    Args:
        models: List of trained model objects
        features: List of feature names
    """
    for model in models:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            print(f"\n{model.__class__.__name__} Feature Importance:")
            for i in range(min(10, len(features))):
                print(f"  {features[indices[i]]}: {importances[indices[i]]:.4f}")

def stacking_ensemble_cv(X_train, y_train, X_test, features, verbose=1):
    """
    Perform time-series cross-validation with stacking ensemble.
    For each season (excluding the earliest), use all previous seasons for training and current season for validation.
    
    Args:
        X_train: Training features DataFrame 
        y_train: Target variable Series
        X_test: Test features DataFrame
        features: List of feature names
        verbose: Whether to print detailed training information
        
    Returns:
        Array of predictions for the test set
    """
    # Check if Season column is available for time-series CV
    if 'Season' in X_train.columns:
        seasons = np.sort(X_train['Season'].unique())
        print(f"Found {len(seasons)} seasons for time-series cross-validation: {seasons}")
        do_time_series_cv = len(seasons) > 1
    else:
        print("Warning: 'Season' column not found. Using regular cross-validation instead of time-series CV.")
        do_time_series_cv = False
        # Create a dummy season column for compatibility
        X_train['Season'] = 1
        seasons = np.array([1])
    
    cv_scores = []
    test_preds_list = []
    
    # Exclude 'Season' from features during model training.
    model_features = [c for c in features if c != 'Season']
    
    # 调试输出：检查特征数据类型
    if verbose:
        print("\nFeature data types before processing:")
        for col in model_features[:10]:  # 只显示前10个特征
            print(f"  {col}: {X_train[col].dtype}")
            # 如果是对象类型，显示样本值
            if X_train[col].dtype == object:
                print(f"    Sample values: {X_train[col].iloc[:5].tolist()}")
    
    if do_time_series_cv:
        # Perform time-series cross-validation
        for season in seasons[1:]:
            if verbose:
                print(f"\n[Stacking] Validating on season {season}")
            train_fold = X_train[X_train['Season'] < season].copy()
            val_fold = X_train[X_train['Season'] == season].copy()
            X_test_fold = X_test.copy()
            
            train_y = y_train.loc[train_fold.index]
            val_y = y_train.loc[val_fold.index]
            
            # Fill missing values
            train_fold.fillna(0, inplace=True)
            val_fold.fillna(0, inplace=True)
            X_test_fold.fillna(0, inplace=True)
            
            # 记录需要移除的非数值型特征
            features_to_remove = []
            
            # 确保所有特征都是数值类型
            for col in model_features:
                if train_fold[col].dtype == object or val_fold[col].dtype == object or X_test_fold[col].dtype == object:
                    if verbose:
                        print(f"  Converting column {col} from {train_fold[col].dtype} to numeric")
                        # 显示一些样本值
                        print(f"    Sample values: {train_fold[col].iloc[:3].tolist()}")
                    try:
                        # 尝试将字符串转换为数值
                        train_fold[col] = pd.to_numeric(train_fold[col], errors='coerce')
                        val_fold[col] = pd.to_numeric(val_fold[col], errors='coerce')
                        X_test_fold[col] = pd.to_numeric(X_test_fold[col], errors='coerce')
                    except Exception as e:
                        # 如果无法转换，则删除该特征
                        print(f"  Warning: Cannot convert feature {col} to numeric - {e}")
                        features_to_remove.append(col)
            
            # 移除无法转换为数值的特征
            for col in features_to_remove:
                if col in model_features:
                    model_features.remove(col)
                    print(f"  Removed non-numeric feature: {col}")
            
            if len(model_features) == 0:
                print("Error: No numeric features available for training!")
                return np.zeros(X_test.shape[0])
                
            # 再次检查数据类型
            if verbose:
                print("\nFeature data types after conversion:")
                for col in model_features[:5]:  # 只显示前5个特征
                    print(f"  {col}: {train_fold[col].dtype}")
            
            # Fill any remaining NaN values after type conversion
            train_fold[model_features] = train_fold[model_features].fillna(0)
            val_fold[model_features] = val_fold[model_features].fillna(0)
            X_test_fold[model_features] = X_test_fold[model_features].fillna(0)
            
            # Normalize features using min-max scaling based on training fold
            try:
                min_vals = train_fold[model_features].min()
                max_vals = train_fold[model_features].max() - min_vals + 1e-8
                train_norm = (train_fold[model_features] - min_vals) / max_vals
                val_norm = (val_fold[model_features] - min_vals) / max_vals
                test_norm = (X_test_fold[model_features] - min_vals) / max_vals
                
                # Ensure no NaNs remain after normalization
                train_norm = train_norm.fillna(0)
                val_norm = val_norm.fillna(0)
                test_norm = test_norm.fillna(0)
                
                # Double-check for NaNs before model training
                if train_norm.isna().any().any() or val_norm.isna().any().any() or test_norm.isna().any().any():
                    if verbose:
                        print("Warning: NaN values detected after normalization. Filling with zeros.")
                    train_norm = train_norm.fillna(0)
                    val_norm = val_norm.fillna(0)
                    test_norm = test_norm.fillna(0)
            except Exception as e:
                print(f"Error during normalization: {e}")
                print("Checking problematic features...")
                for col in model_features:
                    try:
                        min_val = train_fold[col].min()
                        max_val = train_fold[col].max()
                        diff = max_val - min_val
                        print(f"  {col}: min={min_val}, max={max_val}, diff={diff}, dtype={train_fold[col].dtype}")
                    except Exception as e2:
                        print(f"  Error checking {col}: {e2}")
                raise
            
            if train_y.nunique() < 2:
                default_pred = float(train_y.iloc[0])
                val_pred_ensemble = np.full(val_norm.shape[0], default_pred)
                test_pred_ensemble = np.full(test_norm.shape[0], default_pred)
            else:
                base_models = get_base_models()
                val_preds = []
                test_preds = []
                trained_models = []
                
                for i, model in enumerate(base_models):
                    model_name = model.__class__.__name__
                    if verbose:
                        print(f"  Training base model {i+1}/{len(base_models)}: {model_name}")
                    
                    # Handle models that don't support NaN values
                    if model_name in ['LogisticRegression', 'Ridge']:
                        # Create imputer for models that don't handle NaNs
                        imputer = SimpleImputer(strategy='mean')
                        train_data = imputer.fit_transform(train_norm)
                        val_data = imputer.transform(val_norm)
                        test_data = imputer.transform(test_norm)
                    else:
                        train_data = train_norm
                        val_data = val_norm
                        test_data = test_norm
                    
                    model.fit(train_data, train_y)
                    trained_models.append(model)
                    
                    try:
                        if hasattr(model, 'predict_proba'):
                            val_pred = model.predict_proba(val_data)[:, 1]
                            test_pred = model.predict_proba(test_data)[:, 1]
                        else:
                            val_pred = model.predict(val_data)
                            test_pred = model.predict(test_data)
                            val_pred = np.clip((val_pred - val_pred.min()) / (val_pred.max() - val_pred.min() + 1e-8), 0, 1)
                            test_pred = np.clip((test_pred - test_pred.min()) / (test_pred.max() - test_pred.min() + 1e-8), 0, 1)
                    except:
                        val_pred = model.predict(val_data)
                        test_pred = model.predict(test_data)
                        val_pred = np.clip((val_pred - val_pred.min()) / (val_pred.max() - val_pred.min() + 1e-8), 0, 1)
                        test_pred = np.clip((test_pred - test_pred.min()) / (test_pred.max() - test_pred.min() + 1e-8), 0, 1)
                    
                    val_preds.append(val_pred)
                    test_preds.append(test_pred)
                
                if verbose:
                    print_feature_importance(trained_models, model_features)
                
                # Construct meta-feature matrix from base model predictions
                meta_X_val = np.column_stack(val_preds)
                meta_X_test = np.column_stack(test_preds)
                
                # Ensure meta-features don't have NaNs
                meta_X_val = np.nan_to_num(meta_X_val, nan=0.5)
                meta_X_test = np.nan_to_num(meta_X_test, nan=0.5)
                
                meta_model = LogisticRegression(C=1, max_iter=1000, random_state=42, solver='liblinear')
                if verbose:
                    print("  Training meta-model (Logistic Regression)")
                meta_model.fit(meta_X_val, val_y)
                val_pred_ensemble = meta_model.predict_proba(meta_X_val)[:, 1]
                test_pred_ensemble = meta_model.predict_proba(meta_X_test)[:, 1]
                
                if verbose:
                    print("  Meta-model coefficients:")
                    for i, coef in enumerate(meta_model.coef_[0]):
                        model_name = base_models[i].__class__.__name__
                        print(f"    {model_name}: {coef:.4f}")
                
            # Post-processing: spread predictions using a sigmoid transformation for diversity
            val_pred_ensemble = spread_predictions(val_pred_ensemble)
            test_pred_ensemble = spread_predictions(test_pred_ensemble)
            
            val_pred_ensemble = np.clip(val_pred_ensemble, 0.001, 0.999)
            score = brier_score_loss(val_y, val_pred_ensemble)
            cv_scores.append(score)
            if verbose:
                print(f"  Current fold Brier score: {score:.4f}")
                print(f"  Prediction distribution: Min={val_pred_ensemble.min():.4f}, Max={val_pred_ensemble.max():.4f}, Mean={val_pred_ensemble.mean():.4f}, Std={val_pred_ensemble.std():.4f}")
            
            test_preds_list.append(test_pred_ensemble)
    else:
        # Regular cross-validation (not time-series based)
        if verbose:
            print("\n[Stacking] Using regular cross-validation (no time-series)")
        
        # Simple train/test split using random 80/20 split
        from sklearn.model_selection import train_test_split
        train_fold, val_fold, train_y, val_y = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        X_test_fold = X_test.copy()
        
        # Drop Season column if it exists (the dummy column we created)
        if 'Season' in train_fold.columns and 'Season' not in model_features:
            train_fold = train_fold.drop('Season', axis=1)
            val_fold = val_fold.drop('Season', axis=1)
            if 'Season' in X_test_fold.columns:
                X_test_fold = X_test_fold.drop('Season', axis=1)
        
        # Fill missing values
        train_fold.fillna(0, inplace=True)
        val_fold.fillna(0, inplace=True)
        X_test_fold.fillna(0, inplace=True)
        
        # Process exactly as in the time-series case (with same data processing steps)
        # ... (all the same data processing as above) ...
        
        # 记录需要移除的非数值型特征
        features_to_remove = []
        
        # 确保所有特征都是数值类型
        for col in model_features:
            if train_fold[col].dtype == object or val_fold[col].dtype == object or X_test_fold[col].dtype == object:
                if verbose:
                    print(f"  Converting column {col} from {train_fold[col].dtype} to numeric")
                    # 显示一些样本值
                    print(f"    Sample values: {train_fold[col].iloc[:3].tolist()}")
                try:
                    # 尝试将字符串转换为数值
                    train_fold[col] = pd.to_numeric(train_fold[col], errors='coerce')
                    val_fold[col] = pd.to_numeric(val_fold[col], errors='coerce')
                    X_test_fold[col] = pd.to_numeric(X_test_fold[col], errors='coerce')
                except Exception as e:
                    # 如果无法转换，则删除该特征
                    print(f"  Warning: Cannot convert feature {col} to numeric - {e}")
                    features_to_remove.append(col)
        
        # 移除无法转换为数值的特征
        for col in features_to_remove:
            if col in model_features:
                model_features.remove(col)
                print(f"  Removed non-numeric feature: {col}")
        
        if len(model_features) == 0:
            print("Error: No numeric features available for training!")
            return np.zeros(X_test.shape[0])
        
        # Train models and generate predictions similar to the time-series case
        # Normalize features
        try:
            min_vals = train_fold[model_features].min()
            max_vals = train_fold[model_features].max() - min_vals + 1e-8
            train_norm = (train_fold[model_features] - min_vals) / max_vals
            val_norm = (val_fold[model_features] - min_vals) / max_vals
            test_norm = (X_test_fold[model_features] - min_vals) / max_vals
            
            # Ensure no NaNs remain after normalization
            train_norm = train_norm.fillna(0)
            val_norm = val_norm.fillna(0)
            test_norm = test_norm.fillna(0)
        except Exception as e:
            print(f"Error during normalization: {e}")
            return np.zeros(X_test.shape[0])
        
        # Train base models and meta-model
        if train_y.nunique() < 2:
            default_pred = float(train_y.iloc[0])
            test_pred_ensemble = np.full(test_norm.shape[0], default_pred)
            test_preds_list.append(test_pred_ensemble)
        else:
            base_models = get_base_models()
            val_preds = []
            test_preds = []
            trained_models = []
            
            for i, model in enumerate(base_models):
                model_name = model.__class__.__name__
                if verbose:
                    print(f"  Training base model {i+1}/{len(base_models)}: {model_name}")
                
                model.fit(train_norm, train_y)
                trained_models.append(model)
                
                try:
                    if hasattr(model, 'predict_proba'):
                        val_pred = model.predict_proba(val_norm)[:, 1]
                        test_pred = model.predict_proba(test_norm)[:, 1]
                    else:
                        val_pred = model.predict(val_norm)
                        test_pred = model.predict(test_norm)
                        val_pred = np.clip((val_pred - val_pred.min()) / (val_pred.max() - val_pred.min() + 1e-8), 0, 1)
                        test_pred = np.clip((test_pred - test_pred.min()) / (test_pred.max() - test_pred.min() + 1e-8), 0, 1)
                except:
                    val_pred = model.predict(val_norm)
                    test_pred = model.predict(test_norm)
                    val_pred = np.clip((val_pred - val_pred.min()) / (val_pred.max() - val_pred.min() + 1e-8), 0, 1)
                    test_pred = np.clip((test_pred - test_pred.min()) / (test_pred.max() - test_pred.min() + 1e-8), 0, 1)
                
                val_preds.append(val_pred)
                test_preds.append(test_pred)
            
            if verbose:
                print_feature_importance(trained_models, model_features)
            
            # Train meta-model
            meta_X_val = np.column_stack(val_preds)
            meta_X_test = np.column_stack(test_preds)
            
            # Ensure meta-features don't have NaNs
            meta_X_val = np.nan_to_num(meta_X_val, nan=0.5)
            meta_X_test = np.nan_to_num(meta_X_test, nan=0.5)
            
            meta_model = LogisticRegression(C=1, max_iter=1000, random_state=42, solver='liblinear')
            meta_model.fit(meta_X_val, val_y)
            val_pred_ensemble = meta_model.predict_proba(meta_X_val)[:, 1]
            test_pred_ensemble = meta_model.predict_proba(meta_X_test)[:, 1]
            
            # Post-processing
            val_pred_ensemble = spread_predictions(val_pred_ensemble)
            test_pred_ensemble = spread_predictions(test_pred_ensemble)
            
            val_pred_ensemble = np.clip(val_pred_ensemble, 0.001, 0.999)
            score = brier_score_loss(val_y, val_pred_ensemble)
            cv_scores.append(score)
            
            if verbose:
                print(f"  Validation Brier score: {score:.4f}")
                print(f"  Prediction distribution: Min={val_pred_ensemble.min():.4f}, Max={val_pred_ensemble.max():.4f}, Mean={val_pred_ensemble.mean():.4f}, Std={val_pred_ensemble.std():.4f}")
            
            test_preds_list.append(test_pred_ensemble)
    
    if len(cv_scores) > 0:
        print(f"\n[Stacking] Cross-validation Brier Score (mean): {np.mean(cv_scores):.4f}")
    
    final_test_pred = np.mean(test_preds_list, axis=0)
    final_test_pred = spread_predictions(final_test_pred)
    
    return final_test_pred

def spread_predictions(preds, spread_factor=1.5):
    """
    Spread predictions using a sigmoid transformation for diversity
    
    Args:
        preds: Array of predictions
        spread_factor: Factor to control spreading intensity
        
    Returns:
        Array of spread predictions
    """
    logits = np.log(preds / (1 - preds + 1e-8))
    spread_logits = logits * spread_factor
    return 1 / (1 + np.exp(-spread_logits)) 