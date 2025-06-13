import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold

# Load the data from the Excel file
file_path = 'data/New_DED.xlsx'
data = pd.read_excel(file_path)

# Create directories if they don't exist for K-Fold CV results
os.makedirs('logs/OLS_KFold', exist_ok=True)

# Open a text file to save the K-Fold CV output
output_file = open('logs/OLS_KFold/kfold_output.txt', 'w', encoding='utf-8')

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
poly = PolynomialFeatures(degree=1)
X_poly = poly.fit_transform(X)

dependent_variables = [
    'Ratio of Valley Area to Bead Area', 
    'Ratio of Bead Heights', 
    'Ratio of Upper Bead Height and Lowest Valley Point Height', 
    'Ratio of Deposition Width to Lowest Valley Point Height'
]

# Set up K-Fold Cross-Validation with k=5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform K-Fold CV for each dependent variable using OLS, Ridge, Lasso, and ElasticNet
for y_var in dependent_variables:
    y = data[y_var]
    y_mean = np.mean(y)
    
    errors_ols, errors_ridge, errors_lasso, errors_elastic = [], [], [], []
    total_variance_y = sum((y - y_mean) ** 2)  # Total variance of the dependent variable
    
    print(f"\n{'='*80}")
    print(f"K-Fold CV for {y_var}")
    print(f"{'='*80}")
    
    for train_index, test_index in kf.split(X_poly):
        # Split the data into training and test sets
        X_train, X_test = X_poly[train_index], X_poly[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train models on the training set
        ols_model = LinearRegression().fit(X_train, y_train)
        ridge_model = Ridge(alpha=1.0).fit(X_train, y_train)
        lasso_model = Lasso(alpha=0.1, max_iter=50000).fit(X_train, y_train)
        elastic_net_model = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=50000).fit(X_train, y_train)
        
        # Predict the test set
        y_pred_ols = ols_model.predict(X_test)
        y_pred_ridge = ridge_model.predict(X_test)
        y_pred_lasso = lasso_model.predict(X_test)
        y_pred_elastic = elastic_net_model.predict(X_test)
        
        # Calculate mean squared error for the test set
        mse_ols = mean_squared_error(y_test, y_pred_ols)
        mse_ridge = mean_squared_error(y_test, y_pred_ridge)
        mse_lasso = mean_squared_error(y_test, y_pred_lasso)
        mse_elastic = mean_squared_error(y_test, y_pred_elastic)
        
        # Append errors to the list
        errors_ols.append(mse_ols)
        errors_ridge.append(mse_ridge)
        errors_lasso.append(mse_lasso)
        errors_elastic.append(mse_elastic)
    
    # Calculate K-Fold CV R² based on total variance explained
    r2_kfold_ols = 1 - (sum(errors_ols) / total_variance_y)
    r2_kfold_ridge = 1 - (sum(errors_ridge) / total_variance_y)
    r2_kfold_lasso = 1 - (sum(errors_lasso) / total_variance_y)
    r2_kfold_elastic = 1 - (sum(errors_elastic) / total_variance_y)
    
    # Output results for OLS
    print(f"OLS K-Fold R² for {y_var}: {r2_kfold_ols:.4f}")
    print(f"OLS Mean K-Fold MSE for {y_var}: {np.mean(errors_ols):.6f}\n")
    
    # Output results for Ridge
    print(f"Ridge K-Fold R² for {y_var}: {r2_kfold_ridge:.4f}")
    print(f"Ridge Mean K-Fold MSE for {y_var}: {np.mean(errors_ridge):.6f}\n")
    
    # Output results for Lasso
    print(f"Lasso K-Fold R² for {y_var}: {r2_kfold_lasso:.4f}")
    print(f"Lasso Mean K-Fold MSE for {y_var}: {np.mean(errors_lasso):.6f}\n")
    
    # Output results for ElasticNet
    print(f"ElasticNet K-Fold R² for {y_var}: {r2_kfold_elastic:.4f}")
    print(f"ElasticNet Mean K-Fold MSE for {y_var}: {np.mean(errors_elastic):.6f}\n")

# Close the output file
output_file.close()

# Reset sys.stdout to its original state
sys.stdout = sys.__stdout__
