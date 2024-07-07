import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


st.title("Stock Price Predictor App")

default_stock = "BTC-USD"
stock = st.text_input("Enter the stock here", default_stock)

# Fetch stock data
stock_data = yf.Ticker(stock)
stock_info = stock_data.info
stock_name = stock_info.get("shortName", stock)  # Use the short name or fall back to the symbol if not available

end = datetime.now()
start = datetime(end.year-10, end.month, end.day)

bit_coin_data = yf.download(stock, start, end)

#Using our trained model

model = load_model("Latest_bit_coin_model2.keras")
st.subheader(f"{stock_name} Data")
st.write(bit_coin_data)

splitting_len = int(len(bit_coin_data) * 0.9)
x_test = pd.DataFrame(bit_coin_data.Close[splitting_len:])

st.subheader(f'Original Close Price for {stock_name}')
figsize = (15, 6)
fig = plt.figure(figsize=figsize)
plt.plot(bit_coin_data.Close, 'b')
plt.title(f'Original Close Price for {stock_name}', fontweight='bold')
plt.xlabel('Date', fontweight='bold')
plt.ylabel('Close Price', fontweight='bold')
st.pyplot(fig)

st.subheader(f"Test Close Price for {stock_name}")
st.write(x_test)

st.subheader(f'Test Close Price for {stock_name}')
fig = plt.figure(figsize=figsize)
plt.plot(x_test, 'b')
plt.title(f'Test Close Price for {stock_name}', fontweight='bold')
plt.xlabel('Date', fontweight='bold')
plt.ylabel('Close Price', fontweight='bold')
st.pyplot(fig)

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test[['Close']].values)

x_data = []
y_data = []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

plotting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_pre.reshape(-1)
    },
    index=bit_coin_data.index[splitting_len+100:]
)
st.subheader(f"Original values vs Predicted values for {stock_name}")
st.write(plotting_data)

st.subheader(f'Original Close Price vs Predicted Close Price for {stock_name}')
fig = plt.figure(figsize=(15, 6))
plt.plot()
plt.plot(pd.concat([bit_coin_data.Close[:splitting_len+100], plotting_data], axis=0))
plt.legend(["Training data", "Original Test data", "Predicted Test data"], prop={'weight': 'bold'})
plt.title(f'Original Close Price vs Predicted Close Price for {stock_name}', fontweight='bold')
plt.xlabel('Date', fontweight='bold')
plt.ylabel('Close Price', fontweight='bold')
st.pyplot(fig)

st.subheader(f"Future Price values for {stock_name}")

last_100 = bit_coin_data[['Close']].tail(100)
last_100 = scaler.transform(last_100['Close'].values.reshape(-1, 1)).reshape(1, -1, 1)
prev_100 = np.copy(last_100)

def predict_future(no_of_days, prev_100):
    future_predictions = []
    for i in range(no_of_days):
        next_day = model.predict(prev_100)
        future_predictions.append(scaler.inverse_transform(next_day))
        prev_100 = np.append(prev_100[:, 1:, :], next_day.reshape(1, 1, 1), axis=1)
    return future_predictions

no_of_days = int(st.text_input("Enter the No of days to be predicted from current date:", "10"))
future_results = predict_future(no_of_days, prev_100)
future_results = np.array(future_results).reshape(-1, 1)

fig = plt.figure(figsize=(15, 6))
plt.plot(pd.DataFrame(future_results), marker='o')

# Adjust y-axis labels to avoid overlap
step_size = int((max(future_results) - min(future_results)) / 10)
if step_size == 0:
    step_size = 1

for i in range(len(future_results)):
    plt.text(i, future_results[i], int(future_results[i][0]), fontweight='bold')

plt.xlabel('Days', fontweight='bold')
plt.ylabel('Close Price', fontweight='bold')
plt.xticks(range(no_of_days), fontweight='bold')
plt.yticks(np.arange(int(min(future_results)), int(max(future_results)) + step_size, step_size), fontweight='bold')
plt.title(f'Future Prices for {stock_name}', fontweight='bold')
st.pyplot(fig)




# # Calculating Accuracy and Confusion Matrix
# # For simplicity, let's assume we are using a binary classifier for some classification task.
# # We'll generate binary labels by thresholding the predicted and true prices.
# threshold = np.mean(inv_y_test)  # This is just an example; adjust according to your logic.

# true_labels = inv_y_test > threshold
# predicted_labels = inv_pre > threshold

# accuracy = accuracy_score(true_labels, predicted_labels)
# conf_matrix = confusion_matrix(true_labels, predicted_labels)

# st.subheader(f"Accuracy and Confusion Matrix for {stock_name}")
# st.write(f"Accuracy: {accuracy * 100:.2f}%")
# st.write("Confusion Matrix:")
# st.write(conf_matrix)











