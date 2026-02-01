import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

df['Day'] = df.index.day
df['Month'] = df.index.month
df['Year'] = df.index.year
df['DayofWeek'] = df.index.dayofweek
df['DayofYear'] = df.index.dayofyear

train, test = train_test_split(df, test_size=0.2, shuffle=False)

X_train = train[['Day', 'Month', 'Year', 'DayofWeek', 'DayofYear']]
y_train = train['Close']
X_test = test[['Day', 'Month', 'Year', 'DayofWeek', 'DayofYear']]
y_test = test['Close']

model = XGBRegressor()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(train.index, train['Close'], label='Train data')
plt.plot(test.index, test['Close'], label='Test data')
plt.plot(test.index, predictions, label='Predicted data', linestyle='--', marker='o')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.title('Stock Price Prediction using XGBoost')
plt.show()
