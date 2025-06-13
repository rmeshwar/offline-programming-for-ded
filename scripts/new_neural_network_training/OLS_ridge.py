import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import joblib

# Load the data from the Excel file
file_path = 'data/New_DED.xlsx'
data = pd.read_excel(file_path)

# Create directories if they don't exist for synthetic data and model output
os.makedirs('data/synthetic', exist_ok=True)
os.makedirs('models/OLS_Ridge', exist_ok=True)

# Expanded ranges for independent variables
expanded_ranges = {
    'Travel Speed': (10, 60),  # Original: (20, 45)
    'Wire Feed Speed': (3, 12),  # Original: (4, 7)
    'Stepover Distance': (2, 12)  # Original: (3.3, 10.2)
}

# Define independent variables (X) and dependent variables (y)
dependent_variables = [
    'Ratio of Valley Area to Bead Area', 
    'Ratio of Bead Heights', 
    'Ratio of Upper Bead Height and Lowest Valley Point Height', 
    'Ratio of Deposition Width to Lowest Valley Point Height'
]

# Generate random values within the expanded range
def generate_expanded_data(num_samples, column_ranges):
    return pd.DataFrame({
        column: np.random.uniform(low, high, num_samples)
        for column, (low, high) in column_ranges.items()
    })

# Function to generate synthetic data based on Ridge regression
def generate_synthetic_data(num_samples=500):
    synthetic_data = []
    
    # Generate expanded independent variables
    X_synthetic = generate_expanded_data(num_samples, expanded_ranges)
    poly = PolynomialFeatures(degree=1)
    X_poly = poly.fit_transform(X_synthetic)
    
    for y_var in dependent_variables:
        # Match the number of samples for the dependent variable
        y = np.tile(data[y_var], num_samples // len(data) + 1)[:num_samples]
        
        # Train Ridge model
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_poly, y)
        
        # Predict synthetic data
        y_pred_ridge = ridge_model.predict(X_poly)
        
        # Save the Ridge model
        model_filename = f"models/OLS_Ridge/ridge_model_{y_var.replace(' ', '_')}.pkl"
        joblib.dump(ridge_model, model_filename)
        print(f"Model for {y_var} saved as {model_filename}")
        
        synthetic_data.append(y_pred_ridge)
    
    # Combine X with synthetic targets
    synthetic_df = pd.DataFrame(
        np.column_stack([X_synthetic] + synthetic_data),
        columns=['Travel Speed', 'Wire Feed Speed', 'Stepover Distance'] + dependent_variables
    )
    
    synthetic_file = f'data/synthetic/synthetic_data_ridge_{num_samples}_samples.xlsx'
    synthetic_df.to_excel(synthetic_file, index=False)
    print(f"Synthetic data with {num_samples} samples saved to {synthetic_file}")

# Execute synthetic data generation
generate_synthetic_data(num_samples=500)
