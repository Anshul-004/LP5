import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# 1. Load Dataset
# Assuming your file is named 'Google_Stock_Price_Train.csv'
# It should have a 'Open' or 'Close' column.
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values  # Taking the 'Open' price column

# 2. Feature Scaling
# RNNs are very sensitive to the scale of the input data.
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# 3. Creating a Data Structure with 60 Timesteps
# We use the previous 60 days of stock prices to predict the next day's price.
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# 4. Reshaping for LSTM
# LSTM expects input in 3D: (Batch Size, Timesteps, Features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# 5. Build the RNN (LSTM)
model = Sequential([
    # First LSTM layer with Dropout to prevent overfitting
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    
    # Second LSTM layer
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    
    # Third LSTM layer
    LSTM(units=50),
    Dropout(0.2),
    
    # Output layer
    Dense(units=1)
])

# 6. Compile and Train
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32)

# 7. Making Predictions (Visualization)
# Note: In a real scenario, you would use a separate 'Test' CSV here.
# For demonstration, we will predict based on the training data.
predicted_stock_price = model.predict(X_train)
predicted_stock_price = sc.inverse_transform(predicted_stock_price) # Back to original scale

# 8. Visualizing Results
plt.figure(figsize=(12,6))
plt.plot(training_set[60:], color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction (RNN/LSTM)')
plt.xlabel('Time (Days)')
plt.ylabel('Stock Price')
plt.legend()
plt.show()