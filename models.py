from sklearn.impute import SimpleImputer

def stacking_ensemble_cv(X_train, y_train, X_test, features, verbose=False):
    for i, model in enumerate(base_models, 1):
        if verbose:
            print(f"  Training base model {i}/{len(base_models)}: {model.__class__.__name__}")
        
        if model.__class__.__name__ == 'LogisticRegression' or not hasattr(model, 'missing_'):
            imputer = SimpleImputer(strategy='mean')
            train_norm_imputed = imputer.fit_transform(train_norm)
            model.fit(train_norm_imputed, train_y)
            
            test_norm_imputed = imputer.transform(test_norm)
            test_pred[:, i-1] = model.predict_proba(test_norm_imputed)[:, 1]
        else:
            model.fit(train_norm, train_y)
            test_pred[:, i-1] = model.predict_proba(test_norm)[:, 1]
    
    # ... existing code ... 