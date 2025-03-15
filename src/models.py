import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler

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
    seasons = np.sort(X_train['Season'].unique())
    cv_scores = []
    test_preds_list = []
    
    # Exclude 'Season' from features during model training.
    model_features = [c for c in features if c != 'Season']
    
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
        
        # Normalize features using min-max scaling based on training fold
        min_vals = train_fold[model_features].min()
        max_vals = train_fold[model_features].max() - min_vals + 1e-8
        train_norm = (train_fold[model_features] - min_vals) / max_vals
        val_norm = (val_fold[model_features] - min_vals) / max_vals
        test_norm = (X_test_fold[model_features] - min_vals) / max_vals
        
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
            
            # Construct meta-feature matrix from base model predictions
            meta_X_val = np.column_stack(val_preds)
            meta_X_test = np.column_stack(test_preds)
            
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
    
    print(f"\n[Stacking] Local CV Brier Score (mean): {np.mean(cv_scores):.4f}")
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