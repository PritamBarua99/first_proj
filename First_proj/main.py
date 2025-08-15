from init_data import load_data
from data_preprocess import split_data
from ml_model import train_linear_regression, train_random_forest
from data_eval import evaluate_model, plot_feature_importance, plot_residuals
import pickle

# Load and split the data
X, y = load_data()
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

# Train models
lr_model = train_linear_regression(X_train, y_train)
rf_model, best_params = train_random_forest(X_train, y_train)

# Evaluate models
print("Test Set Performance:")
lr_pred = evaluate_model(lr_model, X_test, y_test)
rf_pred = evaluate_model(rf_model, X_test, y_test)

# Show tuned parameters for Random Forest
print("Best Random Forest Parameters:", best_params)

# Visualize Random Forest results
plot_feature_importance(rf_model, X.columns)   # feature importance plot
plot_residuals(y_test, rf_pred)               # residuals vs predicted plot

# Save the trained Random Forest model to disk
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("Random Forest model saved as 'random_forest_model.pkl'.")
