import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor

# Load the dataset into a pandas dataframe
df = pd.read_csv("D:/Rayal/Machine-Learning/new_dataset.csv")

# Get a list of unique employee names and IDs
employees = df[["Name", "user_id"]].drop_duplicates()

# Display the list of employees
print("List of employees:")
for i, row in employees.iterrows():
    print(f"{i + 1}. {row['Name']} (ID: {row['user_id']})")

# Ask the user to select an employee
selected_employee_id = input("Enter the ID of the employee you want to make a prediction for: ")

# Filter the dataset to only include rows for the selected employee
selected_employee_data = df.loc[df["user_id"] == int(selected_employee_id)]

# Filter out rows with "Leave", "Absent", and "Holiday" status
selected_employee_data = selected_employee_data.loc[(selected_employee_data["status"] != "Leave") &
                                                    (selected_employee_data["status"] != "Absent") &
                                                    (selected_employee_data["status"] != "Holiday")]

# Map dayOfQuestion to day_numeric
selected_employee_data["day_numeric"] = selected_employee_data["dayOfQuestion"].apply(lambda x: pd.to_datetime(x, format="%A").dayofweek + 1)

# Map status to a binary value (0 for Present, 1 for Late)
selected_employee_data["status_binary"] = selected_employee_data["status"].apply(lambda x: 0 if x == "Present" else 1)

# Train a random forest regressor on the selected employee's data
X = selected_employee_data[["day_numeric"]]
y = selected_employee_data["status_binary"]
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X, y)

# Get the current day of the week (as a string)
today = datetime.now().strftime("%A")

# Get the numeric value of tomorrow's day of the week
tomorrow_numeric = (datetime.now() + timedelta(days=1)).strftime("%w")

# Map tomorrow's day of the week to a string
days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
tomorrow = days[int(tomorrow_numeric)]

# Make a prediction for tomorrow's status
prediction = regressor.predict([[int(tomorrow_numeric) + 1]])

# Print the prediction
print(f"The chance that {employees.loc[employees['user_id'] == int(selected_employee_id), 'Name'].iloc[0]} will be late on {tomorrow} is {prediction[0] * 100:.2f}%.")
