import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Directory containing the CSV files
csv_directory = 'data/stock/processed'

# List of features to keep
features = [
    'Volatility', 'RSI', 'P/E Ratio', '% Change Adj Close', '% Change Open', 
    '% Change Volume', '% Change Volatility', '% Change RSI', 'Log Open', 
    'Log Adjusted Close', 'Log Volume'
]

# Initialize a dictionary to store feature data for scaling
feature_data = {feature: [] for feature in features}

# Iterate over all files in the directory
for filename in os.listdir(csv_directory):
    if filename.endswith('_numerical_features_processed.csv'):
        file_path = os.path.join(csv_directory, filename)
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Keep only 'Date' and the specified features
        columns_to_keep = ['Date'] + features
        df = df[columns_to_keep]
        
        # Save the modified DataFrame back to the CSV file
        df.to_csv(file_path, index=False)
        print(f"Rewritten {filename} with selected features.")
        
        # Collect data for scaling
        for feature in features:
            feature_data[feature].extend(df[feature].dropna().values)

# Fit scalers for each feature
scalers = {}
for feature in features:
    scaler = StandardScaler()
    scaler.fit([[x] for x in feature_data[feature]])
    scalers[feature] = scaler

# Save the scalers to a file
with open('feature_scalers.pkl', 'wb') as f:
    pickle.dump(scalers, f)

print("Scalers have been computed and saved to 'feature_scalers.pkl'.")