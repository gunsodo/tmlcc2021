from sklearn.svm import SVR
from sklearn.ensemble import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.gaussian_process.kernels import RBF, DotProduct
from math import log10
import numpy as np

MODELS = {
    'svr': SVR(),
    'rf': RandomForestRegressor(),
    'gpr': GaussianProcessRegressor(copy_X_train=False),
    'nn': MLPRegressor(),
    'gbr': GradientBoostingRegressor(),
    'ada': AdaBoostRegressor(),
    'bag': BaggingRegressor(),
    'exr': ExtraTreesRegressor(),
}

TUNED_PARAMS = {
    'svr': [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
            'C': [1e-2, 0.1, 1, 10, 100, 1000]},
            {'kernel': ['linear'], 'C': [1e-2, 0.1, 1, 10, 100, 1000]}],
    'rf': {
        'n_estimators': [100, 200, 500, 1000],
        'max_features': ['auto', 'sqrt'],
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
            'subsample': [1, 0.5, 0.1],
            'n_estimators': [100, 200, 500],
            'max_depth': [3, 6]
        },
}

def train(X, y, model_name, grid_search, save_dir):
    if grid_search:
        clf = GridSearchCV(MODELS[model_name], TUNED_PARAMS[model_name], verbose=10, cv=3, n_jobs=2)
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
        model = MODELS[model_name]
        print("Training the model...")
        model.fit(X, y)
        return model

def predict(X, y, model, grid_search):
    y_pred = model.predict(X)
    lmae = log10(mean_absolute_error(y, y_pred))
    print(f"LMAE loss: {lmae}")
    return lmae