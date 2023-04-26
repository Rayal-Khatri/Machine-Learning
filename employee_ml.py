import pandas as pd

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

# Display the selected employee data
print(selected_employee_data.head())
