import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load data from CSV file
data = pd.read_csv("D:/Rayal/Machine-Learning/new_dataset.csv")

# Convert dayOfQuestion column to corresponding numeric value
data['dayOfQuestion'] = pd.to_datetime(data['dayOfQuestion']).dt.dayofweek

# Filter out rows with absent or holiday status
data = data[data['status'].isin(['Late', 'Present'])]

# Define features and target
X = data[['status', 'dayOfQuestion']]
y = data['status']

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Get the day of the week for tomorrow
import datetime
import calendar
tomorrow = datetime.datetime.today() + datetime.timedelta(days=1)
tomorrow_day_name = calendar.day_name[tomorrow.weekday()]
tomorrow_day_num = list(calendar.day_name).index(tomorrow_day_name)

# Predict the probability of being late tomorrow
prob_late = model.predict_proba([[tomorrow_day_num, 'Present']])[:, 1][0] * 100

# Output the result
print(f"There is a {prob_late:.2f}% chance that the employee will be late tomorrow.")
