import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib

# Use relative paths for portability
pitcher_data_path = "./Savant Pitcher 2021-2025.csv"
batter_data_path = "./Savant Batter 2021-2025.csv"

try:
    # Load the data
    pitcher_data = pd.read_csv(pitcher_data_path)
    batter_data = pd.read_csv(batter_data_path)
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Print column names and row counts for debugging
print("Pitcher Data Columns:", pitcher_data.columns)
print("Pitcher Data Shape:", pitcher_data.shape)
print("Batter Data Columns:", batter_data.columns)
print("Batter Data Shape:", batter_data.shape)

if pitcher_data.empty or batter_data.empty:
    print("One or both CSV files are empty. Please check the data files.")
    exit()

# Remove duplicate player_id rows
pitcher_data = pitcher_data.drop_duplicates(subset='player_id')
batter_data = batter_data.drop_duplicates(subset='player_id')

# Check for missing or duplicate player_id values
if pitcher_data['player_id'].isnull().any() or batter_data['player_id'].isnull().any():
    print("Missing player_id values found in one of the datasets.")
    exit()

# Merge datasets on 'player_id' column
merged_data = pd.merge(pitcher_data, batter_data, on="player_id", suffixes=('_pitcher', '_batter'))

# Print merged data shape and columns for debugging
print("Merged Data Shape (before handling missing values):", merged_data.shape)
print("Merged Data Columns:", merged_data.columns)

# Check for missing values in the merged dataset
missing_values = merged_data.isnull().sum()
print("Missing values per column before handling:", missing_values[missing_values > 0])

# Drop columns with more than 50% missing values
threshold = 0.5 * len(merged_data)
columns_to_drop = missing_values[missing_values > threshold].index
merged_data.drop(columns=columns_to_drop, inplace=True)
print(f"Dropped columns with more than 50% missing values: {list(columns_to_drop)}")

# Fill remaining missing values with column mean
merged_data.fillna(merged_data.mean(), inplace=True)

# Check if merged data is empty
if merged_data.empty:
    print("Merged data is empty after handling missing values. Please check the input data.")
    exit()

# Select numeric columns for scaling
numeric_columns = merged_data.select_dtypes(include=['float64', 'int64']).columns
features = merged_data[numeric_columns]

# Ensure features DataFrame is not empty
if features.empty:
    print("No numeric features found for scaling. Please check the dataset.")
    exit()

# Scale features
scaler = StandardScaler()
try:
    features_scaled = scaler.fit_transform(features)
    print("Features scaled successfully.")
except ValueError as e:
    print(f"Error during scaling: {e}")
    exit()

# Define target variable
if 'WAR' not in merged_data.columns:
    print("Target column 'WAR' not found in the dataset.")
    exit()
target = merged_data['WAR']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Save the model
joblib.dump(model, "baseball_model.pkl")

print("Model training complete. Saved as 'baseball_model.pkl'.")