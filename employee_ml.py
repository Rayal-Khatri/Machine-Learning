import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('D:/Rayal/Machine-Learning/new_dataset.csv')

print(df[['Name', 'user_id']].drop_duplicates().reset_index(drop=True))

emp_id = int(input("Select an employee by entering their ID: "))

df = df[df['user_id'] == emp_id].reset_index(drop=True)

df = df[(df['status'] != 'Holiday') & (df['status'] != 'Absent')]

day_map = {'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6}

df['day_num'] = df['dayOfQuestion'].apply(lambda x: day_map[x])

df['late_flag'] = (df['status'] == 'Late')

if sum(df['late_flag']) > 0:
    X = df.loc[df['late_flag'], ['day_num']]
    y = df.loc[df['late_flag'], 'late_flag']
    rfc = RandomForestClassifier(n_estimators=10, random_state=42)
    rfc.fit(X, y)

    tomorrow = datetime.now() + timedelta(days=1)
    tomorrow_day_of_week = tomorrow.strftime('%A')
    tomorrow_day_num = day_map[tomorrow_day_of_week]
    prob_late = rfc.predict_proba([[tomorrow_day_num]])[0][1] * 100


    print(f"There is a {prob_late:.2f}% chance that the employee will be late tomorrow.")
else:
    print("No late records found for this employee.")
