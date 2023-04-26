import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

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

# Split the data into training and testing sets
X = selected_employee_data[["day_numeric"]]
y = selected_employee_data["status_binary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest regressor on the training data
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = regressor.predict(X_test)

# Calculate the mean absolute error
mae = abs(y_test - y_pred).mean()

# Print the mean absolute error
print(f"Mean absolute error: {mae}")

# Make a prediction for today
today_day_numeric = pd.Timestamp.today().dayofweek + 1
today_pred = regressor.predict([[today_day_numeric]])

# Print the predicted chance of being late today
print(f"Predicted chance of being late today: {today_pred[0]}")
