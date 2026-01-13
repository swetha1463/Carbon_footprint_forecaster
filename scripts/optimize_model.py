"""Hyperparameter optimization script."""
import optuna
from optuna.samplers import TPESampler
import numpy as np
from sklearn.model_selection import cross_val_score
from models.hybrid_model import HybridCarbonForecaster

def objective(trial, X, y):
    """Optuna objective function."""
    # Suggest hyperparameters
    params = {
        'num_features': X.shape[1],
        'tcn_channels': [
            trial.suggest_int('tcn_ch1', 32, 128),
            trial.suggest_int('tcn_ch2', 64, 256),
            trial.suggest_int('tcn_ch3', 128, 512),
        ],
        'attention_hidden': trial.suggest_int('att_hidden', 64, 256),
        'attention_heads': trial.suggest_categorical('att_heads', [2, 4, 8]),
        'xgb_params': {
            'max_depth': trial.suggest_int('xgb_depth', 3, 10),
            'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.3),
            'n_estimators': trial.suggest_int('xgb_n_est', 100, 1000),
        }
    }
    
    # Create and evaluate model
    model = HybridCarbonForecaster(**params)
    model.fit(X, y)
    predictions = model.predict(X)
    rmse = np.sqrt(np.mean((y - predictions) ** 2))
    
    return rmse

def run_optimization(X, y, n_trials=50):
    """Run hyperparameter optimization."""
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42)
    )
    
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    
    print(f"\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"\nBest RMSE: {study.best_value:.4f}")
    
    return study.best_params

# Run if needed
# best_params = run_optimization(X_train, y_train, n_trials=50)