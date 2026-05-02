import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Load the Boston Housing Dataset
# Note: fetch_openml is used as load_boston is deprecated
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]

# 2. Split the data into Training and Testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Feature Scaling (Crucial for Deep Learning)
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

# 4. Build the Deep Neural Network (DNN)
model = Sequential([
    # Input Layer + First Hidden Layer
    Dense(64, activation='relu', input_shape=(xtrain.shape[1],)),
    # Second Hidden Layer
    Dense(32, activation='relu'),
    # Third Hidden Layer
    Dense(16, activation='relu'),
    # Output Layer (1 node for regression, no activation)
    Dense(1)
])

# 5. Compile the Model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# 6. Train the Model
print("Training the Deep Neural Network...")
history = model.fit(
    xtrain, ytrain, 
    epochs=100, 
    batch_size=32, 
    validation_split=0.2, 
    verbose=0
)

# 7. Model Evaluation
y_pred = model.predict(xtest).flatten()

mse = mean_squared_error(ytest, y_pred)
mae = mean_absolute_error(ytest, y_pred)

print("\n--- Model Performance ---")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")

# 8. Visualization
plt.figure(figsize=(10, 6))
plt.scatter(ytest, y_pred, c='green', alpha=0.6)
plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], 'r--', lw=2) # Ideal line
plt.xlabel("Actual Price ($1000s)")
plt.ylabel("Predicted Price ($1000s)")
plt.title("Actual vs Predicted Housing Prices (DNN)")
plt.grid(True)
plt.show()

# 9. Plot Training History (Optional but helpful)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss Progression')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()