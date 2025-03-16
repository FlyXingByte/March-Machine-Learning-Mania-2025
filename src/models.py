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

def get_base_models():
    """
    Define a list of base models for the ensemble.
    Note: Ridge regression is replaced by a calibrated Ridge classifier.
    
    Returns:
        List of initialized model objects.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import RidgeClassifier

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
    
    return [model_xgb, model_lr, model_ridge, model_et, model_rf, model_cat, model_lgb, model_mlp]

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
    Ensure each column in the DataFrame is a Series.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        DataFrame with flattened columns.
    """
    return pd.DataFrame({col: (val if not isinstance(val, pd.DataFrame) else val.iloc[:, 0])
                         for col, val in df.items()}, index=df.index)

def stacking_ensemble_cv(X_train, y_train, X_test, features, verbose=1):
    """
    Perform time-series cross-validation with stacking ensemble using out-of-fold meta model training.
    Modified to use only tournament games for validation in each fold, while training on all available data.
    
    Args:
        X_train: Training features DataFrame.
        y_train: Target variable Series.
        X_test: Test features DataFrame.
        features: List of feature names.
        verbose: Whether to print detailed training information.
        
    Returns:
        Array of predictions for the test set.
    """
    # Check if 'Season' column is available for time-series CV
    if 'Season' in X_train.columns:
        seasons = np.sort(X_train['Season'].unique())
        print(f"Found {len(seasons)} seasons for time-series cross-validation: {seasons}")
        do_time_series_cv = len(seasons) > 1
    else:
        print("Warning: 'Season' column not found. Using regular cross-validation instead of time-series CV.")
        do_time_series_cv = False
        X_train['Season'] = 1
        seasons = np.array([1])
    
    # Lists to collect out-of-fold meta features and corresponding labels,
    # as well as meta features for the test set from each fold.
    meta_features_all = []
    meta_labels_all = []
    test_meta_features_all = []
    cv_scores = []  # For reporting out-of-fold Brier scores

    # Exclude 'Season' from features during model training.
    model_features = [c for c in features if c != 'Season']

    # Debug: print feature data types for a few features
    if verbose:
        print("\nFeature data types before processing:")
        for col in model_features[:10]:
            print(f"  {col}: {X_train[col].dtype}")
            if X_train[col].dtype == object:
                print(f"    Sample values: {X_train[col].iloc[:5].tolist()}")
    
    if do_time_series_cv:
        # Time-series cross-validation: iterate over seasons (skip the earliest season)
        for season in seasons[1:]:
            if verbose:
                print(f"\n[Stacking] Validating on season {season}")
            # Use all data from previous seasons (both Regular and Tournament) for training
            train_fold = X_train[X_train['Season'] < season].copy()
            # Use only tournament games from the current season for validation
            val_fold = X_train[(X_train['Season'] == season) & (X_train['GameType'] == 'Tournament')].copy()
            # If no tournament data is available for validation in this season, skip the fold
            if val_fold.empty:
                print(f"Warning: Season {season} has no tournament data for validation; skipping this fold.")
                continue
            X_test_fold = X_test.copy()
            train_y = y_train.loc[train_fold.index]
            val_y = y_train.loc[val_fold.index]
            
            # One-hot encode 'GameType' if exists
            if 'GameType' in train_fold.columns:
                if verbose:
                    print("One-hot encoding 'GameType' feature for training, validation and test sets.")
                train_dummies = pd.get_dummies(train_fold['GameType'], prefix='GameType')
                val_dummies = pd.get_dummies(val_fold['GameType'], prefix='GameType')
                test_dummies = pd.get_dummies(X_test_fold['GameType'], prefix='GameType')
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
                if (train_fold[col].dtype == object or 
                    val_fold[col].dtype == object or 
                    X_test_fold[col].dtype == object):
                    if verbose:
                        print(f"  Converting column {col} from {train_fold[col].dtype} to numeric")
                        print(f"    Sample values: {train_fold[col].iloc[:3].tolist()}")
                    try:
                        train_fold[col] = pd.to_numeric(train_fold[col], errors='coerce')
                        val_fold[col] = pd.to_numeric(val_fold[col], errors='coerce')
                        X_test_fold[col] = pd.to_numeric(X_test_fold[col], errors='coerce')
                    except Exception as e:
                        print(f"  Warning: Cannot convert feature {col} to numeric - {e}")
                        features_to_remove.append(col)
            for col in features_to_remove:
                if col in model_features:
                    model_features.remove(col)
                    print(f"  Removed non-numeric feature: {col}")
            
            if len(model_features) == 0:
                print("Error: No numeric features available for training!")
                return np.zeros(X_test.shape[0])
            
            if verbose:
                print("\nFeature data types after conversion:")
                for col in model_features[:5]:
                    print(f"  {col}: {train_fold[col].dtype}")
            
            # Imputation: median with missing indicator
            imputer = SimpleImputer(strategy='median', add_indicator=True)
            train_imputed = imputer.fit_transform(train_fold[model_features])
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
            train_norm = pd.DataFrame(train_scaled, columns=new_columns, index=train_fold.index)
            val_norm = pd.DataFrame(val_scaled, columns=new_columns, index=val_fold.index)
            test_norm = pd.DataFrame(test_scaled, columns=new_columns, index=X_test_fold.index)
            train_norm = flatten_dataframe_columns(train_norm)
            val_norm = flatten_dataframe_columns(val_norm)
            test_norm = flatten_dataframe_columns(test_norm)
            
            # Train base models and collect meta features (out-of-fold predictions)
            base_models = get_base_models()
            val_preds = []
            test_preds = []
            for i, model in enumerate(base_models):
                model_name = model.__class__.__name__
                if verbose:
                    print(f"  Training base model {i+1}/{len(base_models)}: {model_name}")
                try:
                    model.fit(train_norm, train_y)
                except ValueError as e:
                    if "3-fold" in str(e):
                        print("Warning: Not enough samples for calibration in CalibratedClassifierCV, using uncalibrated base estimator for this fold.")
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
        
        # Train final meta-model on full out-of-fold predictions
        meta_model = LogisticRegression(C=1, max_iter=1000, random_state=42, solver='liblinear')
        if verbose:
            print("Training final meta-model (Logistic Regression) on out-of-fold predictions")
        meta_model.fit(meta_X_train, meta_y_train)
        final_meta_brier = brier_score_loss(meta_y_train, meta_model.predict_proba(meta_X_train)[:, 1])
        if verbose:
            print(f"Out-of-fold meta-model Brier score: {final_meta_brier:.4f}")
            print("Final meta-model coefficients:")
            for i, coef in enumerate(meta_model.coef_[0]):
                model_name = get_base_models()[i].__class__.__name__
                print(f"    {model_name}: {coef:.4f}")
        # Predict test set using final meta-model
        test_pred_ensemble = meta_model.predict_proba(meta_test)[:, 1]
    
    else:
        # Regular cross-validation branch (non-time-series)
        if verbose:
            print("\n[Stacking] Using regular cross-validation (no time-series)")
        from sklearn.model_selection import train_test_split
        # Use only tournament games for validation if available
        if 'GameType' in X_train.columns:
            tournament_data = X_train[X_train['GameType'] == 'Tournament']
            if tournament_data.empty:
                print("Warning: No tournament data available for validation; falling back to default split.")
                train_fold, val_fold, train_y, val_y = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42
                )
            else:
                val_fold = tournament_data.copy()
                train_fold = X_train.drop(val_fold.index)
                train_y = y_train.loc[train_fold.index]
                val_y = y_train.loc[val_fold.index]
        else:
            train_fold, val_fold, train_y, val_y = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
        X_test_fold = X_test.copy()
        if 'Season' in train_fold.columns and 'Season' not in model_features:
            train_fold = train_fold.drop('Season', axis=1)
            val_fold = val_fold.drop('Season', axis=1)
            if 'Season' in X_test_fold.columns:
                X_test_fold = X_test_fold.drop('Season', axis=1)
        
        # One-hot encode "GameType" if exists
        if 'GameType' in train_fold.columns:
            if verbose:
                print("One-hot encoding 'GameType' feature for training, validation and test sets.")
            train_dummies = pd.get_dummies(train_fold['GameType'], prefix='GameType')
            val_dummies = pd.get_dummies(val_fold['GameType'], prefix='GameType')
            test_dummies = pd.get_dummies(X_test_fold['GameType'], prefix='GameType')
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
        
        features_to_remove = []
        for col in model_features:
            if (train_fold[col].dtype == object or 
                val_fold[col].dtype == object or 
                X_test_fold[col].dtype == object):
                if verbose:
                    print(f"  Converting column {col} from {train_fold[col].dtype} to numeric")
                    print(f"    Sample values: {train_fold[col].iloc[:3].tolist()}")
                try:
                    train_fold[col] = pd.to_numeric(train_fold[col], errors='coerce')
                    val_fold[col] = pd.to_numeric(val_fold[col], errors='coerce')
                    X_test_fold[col] = pd.to_numeric(X_test_fold[col], errors='coerce')
                except Exception as e:
                    print(f"  Warning: Cannot convert feature {col} to numeric - {e}")
                    features_to_remove.append(col)
        for col in features_to_remove:
            if col in model_features:
                model_features.remove(col)
                print(f"  Removed non-numeric feature: {col}")
            
        if len(model_features) == 0:
            print("Error: No numeric features available for training!")
            return np.zeros(X_test.shape[0])
        
        imputer = SimpleImputer(strategy='median', add_indicator=True)
        train_imputed = imputer.fit_transform(train_fold[model_features])
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
        train_norm = pd.DataFrame(train_scaled, columns=new_columns, index=train_fold.index)
        val_norm = pd.DataFrame(val_scaled, columns=new_columns, index=val_fold.index)
        test_norm = pd.DataFrame(test_scaled, columns=new_columns, index=X_test_fold.index)
        train_norm = flatten_dataframe_columns(train_norm)
        val_norm = flatten_dataframe_columns(val_norm)
        test_norm = flatten_dataframe_columns(test_norm)
        
        # Collect out-of-fold meta features using inner CV split
        base_models = get_base_models()
        val_preds = []
        test_preds = []
        for i, model in enumerate(base_models):
            if verbose:
                print(f"  Training base model {i+1}/{len(base_models)}: {model.__class__.__name__}")
            try:
                model.fit(train_norm, train_y)
            except ValueError as e:
                if "3-fold" in str(e):
                    print("Warning: Not enough samples for calibration in CalibratedClassifierCV, using uncalibrated base estimator for this split.")
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
        
        meta_X_train = np.column_stack(val_preds)
        meta_test = np.column_stack(test_preds)
        meta_model = LogisticRegression(C=1, max_iter=1000, random_state=42, solver='liblinear')
        if verbose:
            print("Training final meta-model (Logistic Regression) on out-of-fold predictions")
        meta_model.fit(meta_X_train, val_y)
        final_meta_brier = brier_score_loss(val_y, meta_model.predict_proba(meta_X_train)[:, 1])
        if verbose:
            print(f"Out-of-fold meta-model Brier score: {final_meta_brier:.4f}")
            print("Final meta-model coefficients:")
            for i, coef in enumerate(meta_model.coef_[0]):
                model_name = get_base_models()[i].__class__.__name__
                print(f"    {model_name}: {coef:.4f}")
        test_pred_ensemble = meta_model.predict_proba(meta_test)[:, 1]
    
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
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.3):
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
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.3, 
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
