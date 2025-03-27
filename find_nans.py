import os
import pandas as pd

# Directory containing the CSV files
csv_directory = 'data/stock/processed'
headline_path = 'data/headline/filtered_headlines_2015_error_removed.csv'

# List of columns to keep
selected_columns = [
    'Volatility', 'RSI', 'P/E Ratio', '% Change Adj Close', '% Change Open',
    '% Change Volume', '% Change Volatility', '% Change RSI', 'Log Open',
    'Log Adjusted Close', 'Log Volume'
]

# Initialize counters and a set to store tickers with NaN values after forward-filling
total_files = 0
files_with_nans = 0
tickers_with_nans = set()

# Iterate over all files in the directory
for filename in os.listdir(csv_directory):
    if filename.endswith('_numerical_features_processed.csv'):
        total_files += 1  # Count total files processed
        file_path = os.path.join(csv_directory, filename)
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Drop columns not in the selected_columns list
        columns_to_drop = [col for col in df.columns if col not in selected_columns and col != 'Date']
        df = df.drop(columns=columns_to_drop)
        
        # Extract the ticker name from the filename (assuming the format is "TICKER_numerical_features_processed.csv")
        ticker = filename.split('_')[0]
        
        # Forward-fill the selected columns
        df[selected_columns] = df[selected_columns].ffill()
        
        # Check for NaN values in the selected columns starting from the 15th row
        # Also check if any column is entirely NaN
        if df[selected_columns].iloc[14:].isna().any().any() or df[selected_columns].isna().all().any():
            files_with_nans += 1  # Count files with NaNs
            tickers_with_nans.add(ticker)
        
        # Overwrite the original CSV file with the modified data
        df.to_csv(file_path, index=False)
        print(f"Processed and saved {filename}.")

# Calculate the percentage of files with NaNs
if total_files > 0:
    percentage_with_nans = (files_with_nans / total_files) * 100
else:
    percentage_with_nans = 0

# Convert the set to a list
tickers_with_nans = list(tickers_with_nans)

# Print the results
print(f"Percentage of files with NaN values after forward-filling (starting from the 15th row): {percentage_with_nans:.2f}%")
print("Tickers with NaN values after forward-filling:", tickers_with_nans)

# Process the headline CSV file
if os.path.exists(headline_path):
    # Read the headline CSV file
    headlines_df = pd.read_csv(headline_path)
    
    # Print the length of the CSV file before removal
    initial_length = len(headlines_df)
    print(f"Initial number of rows in headlines CSV: {initial_length}")
    
    # Remove rows where Stock_symbol is in the tickers_with_nans list
    headlines_df = headlines_df[~headlines_df['Stock_symbol'].isin(tickers_with_nans)]
    
    # Print the length of the CSV file after removal
    final_length = len(headlines_df)
    print(f"Number of rows in headlines CSV after removal: {final_length}")
    
    # Calculate and print the percentage decrease
    if initial_length > 0:
        percent_decrease = ((initial_length - final_length) / initial_length) * 100
        print(f"Percentage decrease in headlines CSV: {percent_decrease:.2f}%")
    else:
        print("Headlines CSV is empty.")
    
    # Save the modified headlines CSV file
    headlines_df.to_csv(headline_path, index=False)
    print(f"Modified headlines CSV saved to {headline_path}.")
else:
    print(f"Headline file not found at {headline_path}.")