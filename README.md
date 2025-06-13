# Artificial Intelligence and Statistical Mapping Applied to Additive Manufacturing Toolpath Optimization in Wire-arc DED 

This project is focused on creating an offline programming tool for Wire-Arc DED using synthetic data generation and neural networks.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Functionality Overview](#functionality)

## Installation

To set up the project, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/rmeshwar/offline-programming-for-ded.git
   cd offline-programming-for-ded
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**

   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run any python script, use the following command:

```bash
python 'script name here'.py
```

## Functionality Overview

Following is an explanation of each script in the repository, their functionality, and their usage.

**neural_network_training**
- **OLS_ridge.py**: Creates a set of synthetic data from the starting data using OLS regression with Ridge regularization.
- **OLS_ridge_network.py**: Creates and trains a neural network off of the OLS_ridge synthetic data.
- **OLS_ridge_NODROPOUT.py**: Trains a neural network off the OLS_ridge synthetic data without dropout layers.
- **DOE_ridge.py**: Creates a set of synthetic data using a statistical mapping technique, also fit with Ridge regularization.
- **DOE_ridge_network.py**: Creates and trains a neural network off of the DOE_ridge synthetic data.
- **DOE_ridge_NODROPOUT.py**: Trains a neural network off the DOE_ridge synthetic data without dropout layers.


**synthetic_data_validation**
- Scripts to perform data validation using KFOLD cross-validation and Leave-One-Out cross-validation techniques on both the DOE_ridge and OLS_ridge synethtic data.

**neural_network_validation**
- **DOE_vs_OLS_ridge.py**: A relatively older script meant to compare the effectiveness of the DOE vs. OLS ridge models.
- **model_stress_testing.py**: A simple script use to test the models in various edge cases or extreme inputs.
- **New Models Evaluation.py**: A short script to test the models after updating the data with 8 new samples and re-training them.
- **New_Data_Comparison_Test.py**: A test script which was used to compare the OLD models, before the 8 new samples were added, based on their predictive power in relation to those 8 new samples. It used those 8 samples as unseen, real-world data to provide a more accurate comparison of the models.

- **gui_application.py**: A GUI application to provide a graphical display of predicted stepover values and their estimated loss based on inputted wire feed speed and travel speed.
