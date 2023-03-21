import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

df = pd.read_csv("D:/Rayal/new_dataset.csv", names=['id', 'dateOfQuestion', 'checkInTime', 'checkOutTime', 'Name', 'Duration', 'status', 'user_id', 'dayOfQuestion'], skiprows=[0])

# Display the employee names and IDs
print(df[['Name', 'user_id']].drop_duplicates().reset_index(drop=True))

# Prompt user to select an employee
emp_id = int(input("Select an employee by entering their ID: "))

# Create a new dataset for the selected employee
selected_df = df[df['user_id'] == emp_id]

# Save the new dataset to a CSV file
selected_df.to_csv("selected_employee.csv", index=False)

# Read the dataset
df = pd.read_csv("D:/Rayal/selected_employee.csv", names=['id', 'dateOfQuestion', 'checkInTime', 'checkOutTime', 'Name', 'Duration', 'status', 'user_id', 'dayOfQuestion'], skiprows=[0])

# Encode day of the week as one-hot vector
day_dummies = pd.get_dummies(df['dayOfQuestion'], prefix='dayOfQuestion')
df = pd.concat([df, day_dummies], axis=1)

# Drop original dayOfQuestion column
df.drop('dayOfQuestion', axis=1, inplace=True)

# Set the target variable
y = df['status']

# Set the features
X = df[['dayOfQuestion_Friday', 'dayOfQuestion_Monday', 'dayOfQuestion_Saturday', 'dayOfQuestion_Sunday', 'dayOfQuestion_Thursday', 'dayOfQuestion_Tuesday', 'dayOfQuestion_Wednesday', 'Duration']]

# Impute missing values using the most frequent strategy
imputer = SimpleImputer(strategy='most_frequent')
X = imputer.fit_transform(X)

# Train the decision tree classifier on the entire dataset
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Get the employee's data for the most recent day
recent_data = df.iloc[-1][['dayOfQuestion_Friday', 'dayOfQuestion_Monday', 'dayOfQuestion_Saturday', 'dayOfQuestion_Sunday', 'dayOfQuestion_Thursday', 'dayOfQuestion_Tuesday', 'dayOfQuestion_Wednesday', 'Duration']]
recent_data = recent_data.values.reshape(1, -1)

# Make prediction for the employee for the most recent day
prediction = clf.predict(recent_data)

# Print the prediction
if prediction == 1:
    print("Employee is predicted to be late tomorrow.")
else:
    print("Employee is predicted to be on time tomorrow.")
    
import os
if os.path.exists("D:/Rayal/selected_employee.csv"):
    os.remove("D:/Rayal/selected_employee.csv")
