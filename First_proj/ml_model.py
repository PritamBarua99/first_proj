import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

def train_linear_regression(X_train, y_train):
    """
    Train a simple Linear Regression model.
    """
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest with hyperparameter tuning via RandomizedSearchCV.
    """
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    # Define parameter distributions for tuning
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt']
    }
    # Randomized search (e.g. 10 combinations, 3-fold CV)
    rand_search = RandomizedSearchCV(
        rf, param_distributions=param_dist,
        n_iter=10, cv=3, scoring='r2', n_jobs=-1, random_state=42
    )
    rand_search.fit(X_train, y_train)
    best_rf = rand_search.best_estimator_
    return best_rf, rand_search.best_params_
