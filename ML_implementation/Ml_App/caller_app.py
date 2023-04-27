import os
import sys
# Set current working directory to the folder containing this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Add current working directory to Python path
sys.path.append(os.getcwd())
from Ml_App.employee_ml import predict_late_tomorrow

# Ask the user to select an employee
selected_employee_id = input("Enter the ID of the employee you want to make a prediction for: ")

# Make a prediction for tomorrow's status
prediction = predict_late_tomorrow(selected_employee_id)

# Display the prediction
print(prediction)
