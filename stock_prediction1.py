import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from datetime import date
import pandas_datareader as data
from sklearn.metrics import accuracy_score
import yfinance
st.title('Pink Slips')
start = '2000-01-01'
end = date.today().strftime("%Y-%m-%d")
company = st.text_input('Enter Stock Ticker')
df = data.DataReader(company, 'yahoo', start, end)
#describing data
st.subheader('Data  from 2000- till now')
st.write(df.describe())

#visualization
st.subheader('CLosing Price vs time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'])
st.pyplot(fig)


st.subheader('CLosing Price vs time chart with 100MA' )
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df['Close'])
st.pyplot(fig)


st.subheader('CLosing Price vs time chart with 100MA & 200MA' )
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df['Close'])
st.pyplot(fig)

#open rate
#visualization
st.subheader('Opening Price vs time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df['Opening'])
st.pyplot(fig)


st.subheader('Opening Price vs time chart with 100MA' )
ma100 = df.Open.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df['Open'])
st.pyplot(fig)


st.subheader('Opening Price vs time chart with 100MA & 200MA' )
ma100 = df.Open.rolling(100).mean()
ma200 = df.Open.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df['Open'])
st.pyplot(fig)

#volume 
#visualization
st.subheader('Volume vs time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df['Volume'])
st.pyplot(fig)


st.subheader('Volume vs time chart with 100MA' )
ma100 = df.Volume.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df['Volume'])
st.pyplot(fig)


st.subheader('Volume vs time chart with 100MA & 200MA' )
ma100 = df.Volume.rolling(100).mean()
ma200 = df.Volume.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df['Volume'])
st.pyplot(fig)



#splitting data into training and testing
train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
test = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))
train_array = scaler.fit_transform(train)

#load model
model = load_model('LSTM_model.h5')

#testing part
past_100_days = train.tail(100)
final_df = past_100_days.append(test, ignore_index = True)
input_data = scaler.fit_transform(final_df)
x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])
x_test, y_test = np.array(x_test),np.array(y_test)
y_predicted = model.predict(x_test)
scale_factor = 1/scaler.scale_[0]
y_predicted =  y_predicted*scale_factor
y_test = y_test*scale_factor
#final graph
st.subheader('Predictions vs Original' )
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'ORiginal  price')
plt.plot(y_predicted, 'r', label = 'predicted price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig2)