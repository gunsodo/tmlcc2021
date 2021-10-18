from sklearn.svm import SVR
from sklearn.ensemble import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.gaussian_process.kernels import RBF, DotProduct
from math import log10
import numpy as np

from models.nn import NeuralNetwork

MODELS_GS = {
    'svr': SVR,
    'rf': RandomForestRegressor,
    'gpr': GaussianProcessRegressor,
    'mlp': MLPRegressor,
    'gbr': GradientBoostingRegressor,
    'ada': AdaBoostRegressor,
    'bag': BaggingRegressor,
    'exr': ExtraTreesRegressor,
    'xgb': XGBRegressor,
    'lgb': LGBMRegressor,
    'cat': CatBoostRegressor,
}

MODELS = {
    'svr': SVR(verbose=2),
    'rf': RandomForestRegressor(verbose=2),
    'gpr': GaussianProcessRegressor(copy_X_train=False),
    'mlp': MLPRegressor(verbose=2),
    'gbr': GradientBoostingRegressor(verbose=2),
    'ada': AdaBoostRegressor(),
    'bag': BaggingRegressor(verbose=2),
    'exr': ExtraTreesRegressor(verbose=2),
    'xgb': XGBRegressor(verbosity=2),
    'lgb': LGBMRegressor(verbose=2),
    'cat': CatBoostRegressor(verbose=2),
    'nn': NeuralNetwork(),
}

TUNED_PARAMS = {
    'svr': [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
            'C': [1e-2, 0.1, 1, 10, 100, 1000]},
            {'kernel': ['linear'], 'C': [1e-2, 0.1, 1, 10, 100, 1000]}],
    'rf': {
        'n_estimators': [100, 200, 500],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 2, 4],
    },
    'gpr': [{
            "alpha":  [1e-2, 1e-3],
            "kernel": [RBF(l) for l in np.logspace(-1, 1, 2)]
        }, {
            "alpha":  [1e-2, 1e-3],
            "kernel": [DotProduct(sigma_0) for sigma_0 in np.logspace(-1, 1, 2)]
        }],
    'nn': {
        "hidden_layer_sizes": [(1,), (50,)], 
        "activation": ["identity", "logistic", "tanh", "relu"], 
        "solver": ["lbfgs", "sgd", "adam"], 
        "alpha": [0.00005, 0.0005]
    },
    'gbr': {
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [1, 0.5],
        'n_estimators': [500, 1000],
        'max_depth': [6, None]
    },
    'exr': {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 6, None]
    },
    'xgb': {
        "n_estimators": [100],
        "max_depth": [4, 12],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [1, 0.5],
        "alpha": [0.01, 0.1, 1, 10],
    }
}

N_ESTIMATORS = 1000
GROW_POLICY = 'Depthwise'

def objective(trial, X, y, model_name, n_splits=5, n_repeats=1, n_jobs=2, early_stopping_rounds=10):
    ## Add your model here
    if model_name == 'cat':
        param = {
            "objective": "MAE",
            "n_estimators": N_ESTIMATORS, 
            "depth": trial.suggest_int("max_depth", 8, 16),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.01),
            "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 0.5, 5),
            "min_child_samples": trial.suggest_loguniform("min_child_samples", 1, 32),
            "grow_policy": GROW_POLICY,
            "use_best_model": True,
            "eval_metric": "MAE",
            "od_type": 'iter',
            "od_wait": 20,
        }
    elif model_name == 'lgb':
        param = {
            "objective": "MAE",
            "metric": "MAE",
            "n_estimators": N_ESTIMATORS, 
            "verbosity": -1,
            "boosting_type": "gbdt",                
            "seed": 42,
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
            'num_leaves': trial.suggest_int('num_leaves', 2, 512),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 0, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 32),
        }
    else:
        raise KeyError(f"{model_name} does not Optuna support yet.")

    model = MODELS_GS[model_name](**param)

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
    X_values = X.values
    y_values = y.values
    y_pred = np.zeros_like(y_values)
    for train_index, test_index in rkf.split(X_values):
        X_A, X_B = X_values[train_index, :], X_values[test_index, :]
        y_A, y_B = y_values[train_index], y_values[test_index]

        ## Add your model here
        if model_name == 'cat':
            model.fit(X_A, y_A, eval_set=[(X_B, y_B)],
                verbose=0,
                early_stopping_rounds=early_stopping_rounds,
            )
        elif model_name == 'lgb':
            model.fit(X_A, y_A, eval_set=[(X_B, y_B)], early_stopping_rounds=early_stopping_rounds, eval_metric="mae")

        y_pred[test_index] += model.predict(X_B)
    y_pred /= n_repeats
    return log10(mean_absolute_error(y, y_pred))

def train(X, y, model_name, grid_search, save_dir, param=None):
    if grid_search:
        if model_name not in MODELS_GS:
            raise KeyError(f"{model_name} does not have GridSearchCV support yet.")
        
        clf = GridSearchCV(MODELS_GS[model_name](), TUNED_PARAMS[model_name], verbose=10, cv=3, n_jobs=2)
        print("Training the model...")
        clf.fit(X, y)

        print(f"Best parameters set found on development set: {clf.best_params_}")
        print("Grid scores on development set:\n")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

        return clf

    else:
        if param:
            if model_name == 'cat':
                param["n_estimators"] = N_ESTIMATORS
                param["grow_policy"] = GROW_POLICY
            elif model_name == 'lgb':
                param["n_estimators"] = N_ESTIMATORS
            else:
                pass

            model = MODELS_GS[model_name](**param)
            print("Training model with the best parameter...")
        else:
            model = MODELS[model_name]
            print("Training the model...")

        if model_name == "gpr":
            X = X.iloc[:5000, :]
            y = y[:5000]

        model.fit(X, y)
        return model

def predict(X, y, model):
    y_pred = model.predict(X)
    lmae = log10(mean_absolute_error(y, y_pred))
    print(f"LMAE loss: {lmae}")
    return lmae