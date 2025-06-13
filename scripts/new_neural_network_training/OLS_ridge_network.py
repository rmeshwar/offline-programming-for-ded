import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import os
import joblib

# Create directories if they don't exist for logs and saved models
os.makedirs('logs/neural/OLS_ridge', exist_ok=True)
os.makedirs('saved_models/OLS_ridge', exist_ok=True)

# Load the synthetic data from the generated Ridge model dataset
file_path = 'data/synthetic/synthetic_data_ridge_500_samples.xlsx'  # Adjust based on generated samples
data = pd.read_excel(file_path)

# Open a text file to save the output
output_file = open('logs/neural/OLS_ridge/neural_network_output.txt', 'w', encoding='utf-8')

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

# Set Pandas to display all columns
pd.set_option('display.max_columns', None)

# Define the independent variables (features) and dependent variables (targets)
X = data[['Travel Speed', 'Wire Feed Speed', 'Stepover Distance']]
y = data[['Ratio of Valley Area to Bead Area', 
          'Ratio of Bead Heights', 
          'Ratio of Upper Bead Height and Lowest Valley Point Height',
          'Ratio of Deposition Width to Lowest Valley Point Height']]  # Adding the 4th dependent variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(4, activation='linear')  # Output layer for 4 regression outputs
])

# Compile the model with MSE loss and the Adam optimizer
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Display the model architecture
model.summary()

# Train the neural network
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
plt.show()

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=1)
print(f"Test Loss (MSE): {test_loss}")
print(f"Test MAE: {test_mae}")

# Predict values using the test set
predictions = model.predict(X_test_scaled)

# Create a DataFrame to compare actual and predicted values for all 4 outputs
comparison_df = pd.DataFrame({
    'Actual Ratio of Valley Area to Bead Area': y_test['Ratio of Valley Area to Bead Area'],
    'Predicted Ratio of Valley Area to Bead Area': predictions[:, 0],
    'Actual Ratio of Bead Heights': y_test['Ratio of Bead Heights'],
    'Predicted Ratio of Bead Heights': predictions[:, 1],
    'Actual Ratio of Upper Bead Height and Lowest Valley Point Height': y_test['Ratio of Upper Bead Height and Lowest Valley Point Height'],
    'Predicted Ratio of Upper Bead Height and Lowest Valley Point Height': predictions[:, 2],
    'Actual Ratio of Deposition Width to Lowest Valley Point Height': y_test['Ratio of Deposition Width to Lowest Valley Point Height'],
    'Predicted Ratio of Deposition Width to Lowest Valley Point Height': predictions[:, 3]
})

print(comparison_df.head())

# Save the trained model in Keras recommended format
keras_model_path = 'saved_models/OLS_ridge/neural_network_model.keras'
model.save(keras_model_path)
print(f"Model saved to {keras_model_path}")

# Alternatively, save the model in HDF5 format if needed
h5_model_path = 'saved_models/OLS_ridge/neural_network_model.h5'
model.save(h5_model_path)
print(f"Model saved to {h5_model_path}")

# Save the scaler
scaler_path = 'saved_models/OLS_ridge/scaler.pkl'  # Adjust for each model accordingly
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to {scaler_path}")

# Close the output file
output_file.close()
