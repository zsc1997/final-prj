import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Bidirectional, LSTM, Dense, Multiply, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess the dataset
data = pd.read_csv('TSLA.csv')

# Select columns for modeling
selected_columns = ['Open', 'High', 'Low', 'Close']
data = data[selected_columns].dropna()

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
data = pd.DataFrame(scaled_data, columns=selected_columns)

# Parameters
lookback_window = 5  # Time steps for input sequences
filters = 32         # Number of CNN filters
lstm_units = 20      # LSTM units for BiLSTM
dropout_rate = 0.15  # Dropout rate
epochs = 50          # Number of training epochs
batch_size = 32      # Batch size

# Prepare sequences for training
sequences, labels = [], []
for i in range(len(data) - lookback_window):
    sequences.append(data.iloc[i:i + lookback_window].values)
    labels.append(data.iloc[i + lookback_window]['Close'])

X = np.array(sequences)
y = np.array(labels)

# Split into training and testing sets
train_size = int(0.9 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the CNN + BiLSTM + Attention model
input_layer = Input(shape=X_train.shape[1:], name='Input_Layer')

# CNN Block
conv = Conv1D(filters=filters, kernel_size=2, activation='relu', name='Conv1D_Layer')(input_layer)
pool = MaxPooling1D(pool_size=2, name='MaxPooling_Layer')(conv)
dropout = Dropout(dropout_rate, name='Dropout_Layer')(pool)

# BiLSTM Block
bilstm = Bidirectional(LSTM(lstm_units, activation='tanh', return_sequences=True), name='BiLSTM_Layer')(dropout)

# Attention Mechanism
attention_weights = Dense(1, activation='softmax', name='Attention_Weights')(bilstm)
attention_output = Multiply(name='Attention_Multiplication')([bilstm, attention_weights])

# Flatten and Dense Layers
flat = Flatten(name='Flatten_Layer')(attention_output)
dense = Dense(lstm_units, activation='relu', name='Dense_Layer')(flat)
output = Dense(1, activation='linear', name='Output_Layer')(dense)

# Compile the model
model = Model(inputs=input_layer, outputs=output, name='CNN_BiLSTM_Attention_Model')
model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1)

# Evaluate the model
y_pred = model.predict(X_test).flatten()

# Debugging: Check shapes of predictions and test data
print(f"Shape of y_test: {y_test.shape}, Shape of y_pred: {y_pred.shape}")

# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"CNN + BiLSTM + Attention: RMSE = {rmse}, MAE = {mae}")

# Plot predictions vs actual values
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True Values', color='blue')
plt.plot(y_pred, label='Predictions', color='orange')
plt.title('CNN + BiLSTM + Attention: True vs Predicted Stock Prices')
plt.xlabel('Time Steps')
plt.ylabel('Normalized Prices')
plt.legend()
plt.show()
