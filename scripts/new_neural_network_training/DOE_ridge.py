import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import Ridge
import joblib

# Load the data from the Excel file
file_path = 'data/New_DED.xlsx'
data = pd.read_excel(file_path)

# Create directories if they don't exist for synthetic data and model output
os.makedirs('data/synthetic', exist_ok=True)
os.makedirs('models/DOE_Ridge', exist_ok=True)

# Define independent variables (X) and dependent variables (y)
X = data[['Travel Speed', 'Wire Feed Speed', 'Stepover Distance']]
y_variables = [
    'Ratio of Valley Area to Bead Area', 
    'Ratio of Bead Heights', 
    'Ratio of Upper Bead Height and Lowest Valley Point Height', 
    'Ratio of Deposition Width to Lowest Valley Point Height'
]

# Define polynomial transformation and scaler
poly = PolynomialFeatures(degree=2, include_bias=False)  # For second-order interactions
scaler = MinMaxScaler(feature_range=(0, 1))  # Ensures all values are between 0 and 1

# Define the allowed ranges for independent variables
allowed_ranges = {
    'Travel Speed': (10, 60),
    'Wire Feed Speed': (3, 12),
    'Stepover Distance': (2, 12)
}

# Function to clamp values within the allowed range
def clamp_values(df, column_ranges):
    for column, (min_val, max_val) in column_ranges.items():
        df[column] = df[column].clip(lower=min_val, upper=max_val)
    return df

# Step 1: Create an initial DOE
def create_initial_doe(X, y, y_var_name):
    """
    Train Ridge regression on initial DOE with combined variables.
    """
    # Simplify variables using polynomial transformation
    X_poly = poly.fit_transform(X)
    
    # Normalize the features
    X_poly_scaled = scaler.fit_transform(X_poly)
    
    # Train Ridge regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_poly_scaled, y)
    
    # Save the model
    model_filename = f"models/DOE_Ridge/initial_ridge_model_{y_var_name.replace(' ', '_')}.pkl"
    joblib.dump(ridge, model_filename)
    print(f"Initial Ridge model for {y_var_name} saved as {model_filename}")
    
    # Predict initial synthetic data
    y_pred = ridge.predict(X_poly_scaled)
    return y_pred

# Update Step 1: Generate Initial DOE Synthetic Data
def generate_initial_doe_synthetic_data():
    """
    Generate synthetic data based on the initial DOE using simplified variables.
    """
    initial_synthetic_data = []
    
    for y_var in y_variables:
        y = data[y_var]
        
        # Create initial synthetic data
        y_synthetic = create_initial_doe(X, y, y_var)
        
        # Append predictions
        initial_synthetic_data.append(y_synthetic)
    
    # Combine X (transformed) with the initial synthetic targets
    X_poly = poly.fit_transform(X)
    X_poly_scaled = scaler.fit_transform(X_poly)  # Use MinMaxScaler
    feature_names = poly.get_feature_names_out(['Travel Speed', 'Wire Feed Speed', 'Stepover Distance'])
    initial_synthetic_df = pd.DataFrame(
        np.column_stack([X_poly_scaled] + initial_synthetic_data),
        columns=list(feature_names) + y_variables
    )
    
    # Reverse scaling to bring data back to original range
    X_scaled_back = scaler.inverse_transform(X_poly_scaled)
    initial_synthetic_df[feature_names] = X_scaled_back
    
    # Clamp the independent variables within the allowed range
    initial_synthetic_df = clamp_values(initial_synthetic_df, allowed_ranges)
    
    # Save the initial synthetic dataset
    initial_file = 'data/synthetic/initial_synthetic_data.xlsx'
    initial_synthetic_df.to_excel(initial_file, index=False)
    print(f"Initial synthetic data saved to {initial_file}")
    
    return initial_synthetic_df

# Update Step 2: Expand DOE with More Variables
def expand_doe(initial_synthetic_df, num_samples=500):
    """
    Expand the DOE to map a larger process window by separating simplified variables.
    """
    expanded_synthetic_data = []
    X_expanded = initial_synthetic_df[['Travel Speed', 'Wire Feed Speed', 'Stepover Distance']]
    
    for y_var in y_variables:
        # Load the initial Ridge model
        model_path = f"models/DOE_Ridge/initial_ridge_model_{y_var.replace(' ', '_')}.pkl"
        ridge_model = joblib.load(model_path)
        
        # Predict expanded synthetic data
        X_poly_expanded = poly.transform(X_expanded)
        X_scaled_expanded = scaler.transform(X_poly_expanded)
        y_expanded = ridge_model.predict(X_scaled_expanded)
        
        # Append predictions
        expanded_synthetic_data.append(y_expanded)
    
    # Repeat X values and expand predictions to reach desired sample size
    X_repeated = np.tile(X_expanded.values, (num_samples // len(X_expanded) + 1, 1))[:num_samples]
    y_repeated = [np.tile(y, num_samples // len(y) + 1)[:num_samples] for y in expanded_synthetic_data]
    
    # Combine X and expanded y into a single DataFrame
    expanded_df = pd.DataFrame(
        np.column_stack([X_repeated] + y_repeated),
        columns=['Travel Speed', 'Wire Feed Speed', 'Stepover Distance'] + y_variables
    )
    
    # Clamp the independent variables within the allowed range
    expanded_df = clamp_values(expanded_df, allowed_ranges)
    
    # Save the expanded synthetic dataset
    expanded_file = f'data/synthetic/expanded_synthetic_data_{num_samples}_samples.xlsx'
    expanded_df.to_excel(expanded_file, index=False)
    print(f"Expanded synthetic data with {num_samples} samples saved to {expanded_file}")

# Generate synthetic data using the expanded DOE
initial_synthetic_df = generate_initial_doe_synthetic_data()
expand_doe(initial_synthetic_df)