import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from datetime import date
import yfinance as  yf



st.title('Pink Slips')
start = '2010-01-01'
end = date.today().strftime("%Y-%m-%d")
company = st.text_input("Enter Stock Ticker")
df = yf.download(company, start, end)
#describing data
st.subheader('Data  from 2010-current')
st.write(df)

#visualization
st.subheader('CLosing Price vs time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, 'c', label = 'closing price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig)


st.subheader('CLosing Price vs time chart with 100MA' )
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close,'c', label = 'closing  price')
plt.plot(ma100,'r',label = '100 moving averages for closing price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig)


st.subheader('CLosing Price vs time chart with 100MA & 200MA' )
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close,'c', label = 'closing  price')
plt.plot(ma100,'r',label = '100 moving averages for closing price')
plt.plot(ma200,'g',label = '200 moving averages for closing price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig)

#visualization for opening price
st.subheader('Opening Price vs time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Open,'m', label = 'opening price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig)


st.subheader('Opening Price vs time chart with 100MA' )
ma100 = df.Open.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Open,'m',label = 'opening price')
plt.plot(ma100,'y',label = '100 moving averages for opening price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig)


st.subheader('Opening Price vs time chart with 100MA & 200MA' )
ma100 = df.Open.rolling(100).mean()
ma200 = df.Open.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Open,'m',label = 'opening price')
plt.plot(ma100,'y', label = '100 moving averages for opening price')
plt.plot(ma200,'k', label = '200 moving averages for opening price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig)

#visualization for highestprice
st.subheader('highest Price vs time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.High, 'c', label = 'Highest price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig)


st.subheader('highest Price vs time chart with 100MA' )
ma100 = df.High.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.High,'c', label = 'Highest  price')
plt.plot(ma100,'r',label = '100 moving averages for High price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig)


st.subheader('Highest Price vs time chart with 100MA & 200MA' )
ma100 = df.High.rolling(100).mean()
ma200 = df.High.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.High,'c', label = 'Highest price')
plt.plot(ma100,'r',label = '100 moving averages for highest price')
plt.plot(ma200,'g',label = '200 moving averages for highest price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig)


#splitting data into training and testing
train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
test = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))
train_array = scaler.fit_transform(train)

model = load_model('Stock prediction.h5')


#splitting opening data into training and testing
train_open = pd.DataFrame(df['Open'][0:int(len(df)*0.70)])
test_open = pd.DataFrame(df['Open'][int(len(df)*0.70):int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))
train_open_array = scaler.fit_transform(train_open)

model = load_model('Stock prediction.h5')


#splitting highest data into training and testing
train_high = pd.DataFrame(df['High'][0:int(len(df)*0.70)])
test_high = pd.DataFrame(df['High'][int(len(df)*0.70):int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))
train_array_high = scaler.fit_transform(train_high)


#load model
model = load_model('Stock prediction.h5')

# #load model
# from keras.layers import Dense, Dropout, LSTM
# from keras.models import Sequential

# model = Sequential()
# model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, 
#                input_shape = (x_train.shape[1],1)))
# model.add(Dropout(0.2))


# model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
# model.add(Dropout(0.3))


# model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
# model.add(Dropout(0.4))


# model.add(LSTM(units = 120, activation = 'relu'))
# model.add(Dropout(0.5))

# model.add(Dense(units = 1))
# model.compile(optimizer  = 'adam', loss = 'mean_squared_error')
# model.fit(x_train, y_train, epochs = 50)

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
st.subheader('Closing Predictions vs Closing Original' )
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original CLosing price')
plt.plot(y_predicted, 'r', label = 'Predicted CLosing price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig2)






#opening testing part
past_100_days_open = train_open.tail(100)
final_df_open = past_100_days_open.append(test_open, ignore_index = True)
input_data_open = scaler.fit_transform(final_df_open)
x_test_open = []
y_test_open = []
for i in range(100, input_data_open .shape[0]):
  x_test_open.append(input_data_open[i-100:i])
  y_test_open.append(input_data_open[i,0])
x_test_open, y_test_open = np.array(x_test_open),np.array(y_test_open)
y_predicted_open = model.predict(x_test_open)
scale_factor = 1/scaler.scale_[0]
y_predicted_open =  y_predicted_open*scale_factor
y_test_open = y_test_open*scale_factor
#final graph
st.subheader('Opening Predictions vs Opening Original' )
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test_open, 'm', label = 'Original Opening price')
plt.plot(y_predicted_open, 'y', label = 'Predicted Opening price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig2)

#highest testing part
past_100_days_high = train_high.tail(100)
final_df_high = past_100_days_high.append(test_high, ignore_index = True)
input_data_high = scaler.fit_transform(final_df_high)
x_test_high = []
y_test_high = []
for i in range(100, input_data_high .shape[0]):
  x_test_high.append(input_data_high[i-100:i])
  y_test_high.append(input_data_high[i,0])
x_test_high, y_test_high = np.array(x_test_high),np.array(y_test_high)
y_predicted_high = model.predict(x_test_high)
scale_factor = 1/scaler.scale_[0]
y_predicted_high =  y_predicted_high*scale_factor
y_test_high = y_test_high*scale_factor
#final graph
st.subheader('Highest Predictions  vs Highest Original ' )
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test_high, 'g', label = 'Highest Original  price')
plt.plot(y_predicted_high, 'k', label = 'Highest Predicted  price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig2)