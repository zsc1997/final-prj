import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Model
from keras.layers import Input, Bidirectional, LSTM, Dense, Multiply, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('TSLA.csv')

# Select relevant columns (Open, High, Low, Close)
columns_to_normalize = ['Open', 'High', 'Low', 'Close']
data = data[columns_to_normalize]

# Drop rows with missing values
data = data.dropna()

# Normalize the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)
data = pd.DataFrame(normalized_data, columns=columns_to_normalize)

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

# Define BiLSTM + Attention model
inputs = Input(shape=X_train.shape[1:])
lstm_out = Bidirectional(LSTM(lstm_units, activation='tanh', return_sequences=True))(inputs)
attention = Dense(1, activation='softmax')(lstm_out)
attention_mul = Multiply()([lstm_out, attention])
x = Flatten()(attention_mul)  # Flatten the attention output
x = Dense(lstm_units, activation='relu')(x)  # Dense layer after attention
outputs = Dense(1, activation='linear')(x)  # Final output
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1)

# Evaluate the model
y_pred = model.predict(X_test).flatten()

# Debug: Check shapes
print(f"Shape of y_test: {y_test.shape}, Shape of y_pred: {y_pred.shape}")

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"BiLSTM + Attention: RMSE = {rmse}, MAE = {mae}")

# Plot predictions
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True Values', color='blue')
plt.plot(y_pred, label='Predictions', color='orange')
plt.title('BiLSTM + Attention: True vs Predicted Stock Prices')
plt.xlabel('Time Steps')
plt.ylabel('Normalized Prices')
plt.legend()
plt.show()
