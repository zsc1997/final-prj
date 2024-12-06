import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data = pd.read_csv('TSLA.csv')

# Select relevant columns (Open, High, Low, Close)
columns_to_normalize = ['Open', 'High', 'Low', 'Close']
data = data[columns_to_normalize]

# Drop rows with missing values
data = data.dropna()

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Normalize the data
normalized_data = scaler.fit_transform(data)

# Convert normalized data back to a DataFrame
normalized_df = pd.DataFrame(normalized_data, columns=columns_to_normalize)

# Save the normalized data to a new CSV file
normalized_df.to_csv('normalized_TSLA.csv', index=False)

print("Normalization complete! Saved as 'normalized_TSLA.csv'.")
