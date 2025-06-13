import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error

# Load the data from the Excel file
file_path = 'data/New_DED.xlsx'
data = pd.read_excel(file_path)

# Create directories if they don't exist for LOOCV results
os.makedirs('logs/OLS_LOOCV', exist_ok=True)

# Open a text file to save the LOOCV output
output_file = open('logs/OLS_LOOCV/loocv_output.txt', 'w', encoding='utf-8')

# Redirect print statements to both console and text file
class DualOutput:
    def __init__(self, file1, file2):
        self.file1 = file1
        self.file2 = file2

    def write(self, data):
        self.file1.write(data)
        self.file2.write(data)

    def flush(self):
        pass

# Set sys.stdout to output to both console and file
sys.stdout = DualOutput(sys.stdout, output_file)

# Define independent variables (X) and dependent variables (y)
X = data[['Travel Speed', 'Wire Feed Speed', 'Stepover Distance']]
poly_degree = 2  # You can change the polynomial degree here
poly = PolynomialFeatures(degree=poly_degree)
X_poly = poly.fit_transform(X)

dependent_variables = [
    'Ratio of Valley Area to Bead Area',
    'Ratio of Bead Heights',
    'Ratio of Upper Bead Height and Lowest Valley Point Height',
    'Ratio of Deposition Width to Lowest Valley Point Height'
]

# Perform LOOCV for each dependent variable and each model
for y_var in dependent_variables:
    y = data[y_var]
    y_mean = np.mean(y)

    # Initialize empty lists to store metrics for each model
    errors_ols = []
    errors_ridge = []
    errors_lasso = []
    errors_elastic = []

    total_variance_y = sum((y - y_mean) ** 2)  # Total variance of the dependent variable

    print(f"\n{'=' * 80}")
    print(f"LOOCV for {y_var}")
    print(f"{'=' * 80}")

    n_samples, n_features = X_poly.shape

    for i in range(n_samples):  # Loop through each sample
        # Leave one sample out
        X_train = np.delete(X_poly, i, axis=0)
        y_train = np.delete(y.values, i, axis=0)

        X_test = X_poly[i].reshape(1, -1)  # The sample left out as the test set
        y_test = y.values[i]  # The true value for this sample

        # Train models on the remaining data
        ols_model = LinearRegression().fit(X_train, y_train)
        ridge_model = Ridge(alpha=1.0).fit(X_train, y_train)
        lasso_model = Lasso(alpha=0.1, max_iter=50000).fit(X_train, y_train)
        elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=50000).fit(X_train, y_train)

        # Predict the left-out sample
        y_pred_ols = ols_model.predict(X_test)
        y_pred_ridge = ridge_model.predict(X_test)
        y_pred_lasso = lasso_model.predict(X_test)
        y_pred_elastic = elastic_model.predict(X_test)

        # Calculate mean squared error for the left-out sample
        mse_ols = mean_squared_error([y_test], y_pred_ols)
        mse_ridge = mean_squared_error([y_test], y_pred_ridge)
        mse_lasso = mean_squared_error([y_test], y_pred_lasso)
        mse_elastic = mean_squared_error([y_test], y_pred_elastic)

        # Append errors to the list
        errors_ols.append(mse_ols)
        errors_ridge.append(mse_ridge)
        errors_lasso.append(mse_lasso)
        errors_elastic.append(mse_elastic)

    # Calculate LOOCV R² based on total variance explained
    r2_loocv_ols = 1 - (sum(errors_ols) / total_variance_y)
    r2_loocv_ridge = 1 - (sum(errors_ridge) / total_variance_y)
    r2_loocv_lasso = 1 - (sum(errors_lasso) / total_variance_y)
    r2_loocv_elastic = 1 - (sum(errors_elastic) / total_variance_y)

    # Output all the key metrics for each model
    print(f"\nOLS Regression Results for {y_var}:")
    print(f"LOOCV R²: {r2_loocv_ols:.4f}")
    print(f"Mean LOOCV MSE: {np.mean(errors_ols):.6f}")

    print(f"\nRidge Regression Results for {y_var}:")
    print(f"LOOCV R²: {r2_loocv_ridge:.4f}")
    print(f"Mean LOOCV MSE: {np.mean(errors_ridge):.6f}")

    print(f"\nLasso Regression Results for {y_var}:")
    print(f"LOOCV R²: {r2_loocv_lasso:.4f}")
    print(f"Mean LOOCV MSE: {np.mean(errors_lasso):.6f}")

    print(f"\nElasticNet Regression Results for {y_var}:")
    print(f"LOOCV R²: {r2_loocv_elastic:.4f}")
    print(f"Mean LOOCV MSE: {np.mean(errors_elastic):.6f}")

# Close the output file
output_file.close()

# Reset sys.stdout to its original state
sys.stdout = sys.__stdout__
