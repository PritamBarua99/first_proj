import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

def evaluate_model(model, X_test, y_test):
    """
    Predict on test data and print R^2 and MSE.
    Returns predicted values.
    """
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{model.__class__.__name__} R^2: {r2:.3f}, MSE: {mse:.3f}")
    return y_pred

def plot_feature_importance(model, feature_names):
    """
    Plot the feature importances for tree-based model.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    plt.figure(figsize=(8, 6))
    plt.title("Feature Importances (Random Forest)")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), sorted_features, rotation=90)
    plt.tight_layout()
    plt.show()

def plot_residuals(y_test, y_pred):
    """
    Plot residuals (y_test - y_pred) vs predicted values.
    """
    residuals = y_test - y_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.hlines(0, y_pred.min(), y_pred.max(), colors='red', linestyles='dashed')
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals (True - Predicted)")
    plt.title("Residuals vs Predicted")
    plt.tight_layout()
    plt.show()
