import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
print(file_path)

df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
print(df.head())
print(df.describe())

train = df[df['ds'] < '2017-01-01']
test = df[df['ds'] >= '2017-01-01']


model = Prophet()
model.fit(train)

future = model.make_future_dataframe(periods=len(test))
forecast = model.predict(future)

plt.figure(figsize=(10, 6))
plt.plot(train['ds'], train['y'], label='Train data')
plt.plot(test['ds'], test['y'], label='Data after prediction')
plt.plot(forecast['ds'], forecast['yhat'], label='Pattern')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='k', alpha=.10)

plt.plot(forecast[forecast['ds'] >= '2017-01-01']['ds'], forecast[forecast['ds'] >= '2017-01-01']['yhat'], color='orange', linestyle='--', marker='o', label='Predicted data from Train date')

plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.title('Stock Price Prediction using Prophet')
plt.show()
