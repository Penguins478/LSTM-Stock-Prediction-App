import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import yfinance as yfin


yfin.pdr_override()

st.title('Stock Trend Prediction')

file = open("data.txt", "a")

user_input = st.text_input('Enter Stock Ticker', 'AAPL')


start = '2010-01-01'
end = '2019-12-31'

y_symbols = [user_input]
df = pdr.get_data_yahoo(y_symbols, start, end)


st.subheader('Data from 2010 - 2019')
st.write(df.describe())

st.subheader('Closing Price vs. Time Chart')
fig = plt.figure(figsize = (12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs. Time Chart with 100MA (Moving Average)')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs. Time Chart with 100MA and 200 MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.7) : int(len(df))])


scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)


# load the ML model (which is already pre-trained from the notebook)

model = load_model('keras_model.h5')

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor



total = 0
len = min(len(y_predicted), len(y_test))
for i in range(0, len):
    total += abs(float(y_predicted[i]) / float(y_test[i]))
error = 100 - ((total / len - 1) * 100)

file.write(str(y_symbols) + "  " + str(error) + "\n")


# Final Graph

st.subheader('Predictions vs. Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time (Days)')
plt.ylabel('Price ($)')
plt.legend()
st.pyplot(fig2)

