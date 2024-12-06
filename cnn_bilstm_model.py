import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Bidirectional, LSTM, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('normalized_TSLA.csv')

# Parameters
window = 5
lstm_units = 16
epochs = 50
batch_size = 32

# Prepare sequences
sequences, labels = [], []
for i in range(len(data) - window):
    sequences.append(data.iloc[i:i + window].values)
    labels.append(data.iloc[i + window]['Close'])

X = np.array(sequences)
y = np.array(labels)

# Train-test split
train_size = int(0.9 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define CNN + BiLSTM model
inputs = Input(shape=X_train.shape[1:])
x = Conv1D(filters=lstm_units, kernel_size=1, activation='relu')(inputs)
x = MaxPooling1D(pool_size=window)(x)
x = Dropout(0.1)(x)
lstm_out = Bidirectional(LSTM(lstm_units, activation='tanh'))(x)
outputs = Dense(1, activation='linear')(lstm_out)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1)

# Evaluate the model
y_pred = model.predict(X_test).flatten()
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"CNN + BiLSTM: RMSE = {rmse}, MAE = {mae}")

# Plot predictions
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True Values', color='blue')
plt.plot(y_pred, label='Predictions', color='orange')
plt.title('CNN + BiLSTM: True vs Predicted Stock Prices')
plt.xlabel('Time Steps')
plt.ylabel('Normalized Prices')
plt.legend()
plt.show()
