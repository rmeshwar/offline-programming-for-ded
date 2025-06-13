import sys
import os
import numpy as np
import joblib
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QMessageBox
)
from PyQt5.QtGui import QFont
from tensorflow.keras.models import load_model
import pandas as pd
import traceback
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Load model paths dynamically based on script/executable directory
def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, 'saved_models')
    else:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')

base_model_path = get_base_path()
MODEL_PATHS = {
    "OLS Ridge OLD": {
        "model": os.path.join(base_model_path, 'OLS_ridge_OLD/neural_network_model.keras'),
        "scaler": os.path.join(base_model_path, 'OLS_ridge_OLD/scaler.pkl')
    },
    "OLS Ridge NODROPOUT OLD": {
        "model": os.path.join(base_model_path, 'OLS_ridge_NODROPOUT_OLD/neural_network_model.keras'),
        "scaler": os.path.join(base_model_path, 'OLS_ridge_NODROPOUT_OLD/scaler.pkl')
    },
    "DOE Ridge OLD": {
        "model": os.path.join(base_model_path, 'DOE_Ridge_OLD/neural_network_model.keras'),
        "scaler": os.path.join(base_model_path, 'DOE_Ridge_OLD/scaler.pkl')
    },
    "DOE Ridge NODROPOUT OLD": {
        "model": os.path.join(base_model_path, 'DOE_Ridge_NODROPOUT_OLD/neural_network_model.keras'),
        "scaler": os.path.join(base_model_path, 'DOE_Ridge_NODROPOUT_OLD/scaler.pkl')
    },
    "OLS Ridge": {
        "model": os.path.join(base_model_path, 'OLS_ridge/neural_network_model.keras'),
        "scaler": os.path.join(base_model_path, 'OLS_ridge/scaler.pkl')
    },
    "OLS Ridge NODROPOUT": {
        "model": os.path.join(base_model_path, 'OLS_ridge_NODROPOUT/neural_network_model.keras'),
        "scaler": os.path.join(base_model_path, 'OLS_ridge_NODROPOUT/scaler.pkl')
    },
    "DOE Ridge": {
        "model": os.path.join(base_model_path, 'DOE_Ridge/neural_network_model.keras'),
        "scaler": os.path.join(base_model_path, 'DOE_Ridge/scaler.pkl')
    },
    "DOE Ridge NODROPOUT": {
        "model": os.path.join(base_model_path, 'DOE_Ridge_NODROPOUT/neural_network_model.keras'),
        "scaler": os.path.join(base_model_path, 'DOE_Ridge_NODROPOUT/scaler.pkl')
    }
}

# Ideal values for the 2 dependent variables used in calculating loss value.
# Note that only Ratio of Valley area to Bead Area and Ratio of Bead Heights are used here;
# These were found to be the best combination of model outputs to use to provide the most predictive power.
IDEAL_VALUES = {
    "Ratio of Valley Area to Bead Area": 0,
    "Ratio of Bead Heights": 1
}


class PredictStepoverApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Optimal Stepover Distance Prediction')

        # Set the font size for the entire application
        QApplication.setFont(QFont("Arial", 12))

        layout = QVBoxLayout()

        # Model input selection
        self.model_label = QLabel('Select Model:')
        layout.addWidget(self.model_label)
        self.model_dropdown = QComboBox(self)
        self.model_dropdown.addItems(MODEL_PATHS.keys())
        layout.addWidget(self.model_dropdown)

        # Input boxes for travel speed and wire feed speed
        self.travel_speed_input = QLineEdit(self)
        self.travel_speed_input.setPlaceholderText("Enter Travel Speed (10 - 60)")
        layout.addWidget(self.travel_speed_input)
        self.wire_feed_speed_input = QLineEdit(self)
        self.wire_feed_speed_input.setPlaceholderText("Enter Wire Feed Speed (3 - 12)")
        layout.addWidget(self.wire_feed_speed_input)

        self.calculate_button = QPushButton('Calculate', self)
        self.calculate_button.clicked.connect(self.calculate_stepover)
        layout.addWidget(self.calculate_button)

        self.loading_label = QLabel('', self)  # Label to show loading text
        layout.addWidget(self.loading_label)

        self.result_label = QLabel('Optimal Stepover Distance will be displayed here', self)
        layout.addWidget(self.result_label)

        # Add matplotlib canvas for the graph
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Label for displaying hover information
        self.hover_label = QLabel(self)
        layout.addWidget(self.hover_label)

        self.setLayout(layout)

    def calculate_stepover(self):
        try:
            # Show loading text
            self.loading_label.setText('Calculating...')
            QApplication.processEvents()

            travel_speed = self.travel_speed_input.text()
            wire_feed_speed = self.wire_feed_speed_input.text()

            try:
                travel_speed = float(travel_speed)
                wire_feed_speed = float(wire_feed_speed)
            except ValueError:
                QMessageBox.warning(self, "Input Error", "Please enter valid numerical values for Travel Speed and Wire Feed Speed.")
                self.loading_label.setText('')
                return

            if not (10 <= travel_speed <= 60):
                QMessageBox.warning(self, "Input Error", "Travel Speed must be between 10 and 60.")
                self.loading_label.setText('')
                return

            if not (3 <= wire_feed_speed <= 12):
                QMessageBox.warning(self, "Input Error", "Wire Feed Speed must be between 3 and 12.")
                self.loading_label.setText('')
                return

            model_name = self.model_dropdown.currentText()
            model_path = MODEL_PATHS[model_name]["model"]
            scaler_path = MODEL_PATHS[model_name]["scaler"]

            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                QMessageBox.critical(self, "File Error", f"Model or scaler file not found for {model_name}.")
                self.loading_label.setText('')
                return

            model = load_model(model_path)
            scaler = joblib.load(scaler_path)

            stepover_distances = np.linspace(2, 12, 100)
            best_stepover = None
            best_output_difference = float('inf')

            losses = []

            for stepover in stepover_distances:
                input_data = np.array([[travel_speed, wire_feed_speed, stepover]])
                input_data_df = pd.DataFrame(input_data, columns=['Travel Speed', 'Wire Feed Speed', 'Stepover Distance'])
                input_data_scaled = scaler.transform(input_data_df)

                predictions = model.predict(input_data_scaled)
                pred_valley = predictions[0][0]
                pred_bead_heights = predictions[0][1]

                loss = abs(pred_valley - IDEAL_VALUES["Ratio of Valley Area to Bead Area"]) + \
                       abs(pred_bead_heights - IDEAL_VALUES["Ratio of Bead Heights"])

                losses.append(loss)

                if abs(loss) < abs(best_output_difference):
                    best_output_difference = abs(loss)
                    best_stepover = stepover

            self.result_label.setText(f'Optimal Stepover Distance: {best_stepover:.2f}')

            # Plot the graph
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(stepover_distances, losses, label='Loss vs. Stepover Distance')
            ax.axvline(best_stepover, color='red', linestyle='--', label=f'Optimal Stepover ({best_stepover:.2f})')
            ax.set_xlabel('Stepover Distance')
            ax.set_ylabel('Predicted Loss')
            ax.set_title('Loss vs. Stepover Distance')
            ax.legend()

            # Add hover interaction with logic to display closest point on graph
            def on_hover(event):
                if event.inaxes == ax:
                    x = event.xdata
                    if x is not None:
                        closest_index = np.argmin(np.abs(stepover_distances - x))
                        closest_stepover = stepover_distances[closest_index]
                        closest_loss = losses[closest_index]
                        self.hover_label.setText(f"Stepover: {closest_stepover:.2f}, Loss: {closest_loss:.4f}")

            self.canvas.mpl_connect("motion_notify_event", on_hover)
            self.canvas.draw()

            # Clear loading text
            self.loading_label.setText('')

        except Exception as e:
            error_message = f"An unexpected error occurred: {str(e)}"
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", error_message)
            self.loading_label.setText('')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PredictStepoverApp()
    window.show()
    sys.exit(app.exec_())
