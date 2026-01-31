import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

pitcher_data_path = "./Savant Pitcher 2021-2025.csv"
batter_data_path = "./Savant Batter 2021-2025.csv"

try:
    pitcher_data = pd.read_csv(pitcher_data_path)
    batter_data = pd.read_csv(batter_data_path)
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

print("Pitcher Data Shape:", pitcher_data.shape)
print("Batter Data Shape:", batter_data.shape)

if pitcher_data.empty or batter_data.empty:
    print("One or both CSV files are empty. Please check the data files.")
    exit()

# Remove duplicates keeping first occurrence
pitcher_data = pitcher_data.drop_duplicates(subset=['player_id', 'year'], keep='first')
batter_data = batter_data.drop_duplicates(subset=['player_id', 'year'], keep='first')

# Check for missing player_id or year values
if pitcher_data['player_id'].isnull().any() or pitcher_data['year'].isnull().any():
    print("Removing rows with missing player_id or year in pitcher data...")
    pitcher_data = pitcher_data.dropna(subset=['player_id', 'year'])

if batter_data['player_id'].isnull().any() or batter_data['year'].isnull().any():
    print("Removing rows with missing player_id or year in batter data...")
    batter_data = batter_data.dropna(subset=['player_id', 'year'])

# Merge datasets on player_id and year
merged_data = pd.merge(
    batter_data, 
    pitcher_data, 
    on=['player_id', 'year'], 
    suffixes=('_batter', '_pitcher'),
    how='inner'
)

print(f"\nMerged Data Shape: {merged_data.shape}")
print(f"Rows in merged dataset: {len(merged_data)}")

if merged_data.empty:
    print("No common player-year combinations found. Check your data.")
    exit()

target_column = 'batting_avg_batter'

if target_column not in merged_data.columns:
    print(f"Error: Target column '{target_column}' not found.")
    print(f"Available columns: {[c for c in merged_data.columns if 'batting' in c.lower()]}")
    exit()

# Drop rows where target is missing
merged_data = merged_data.dropna(subset=[target_column])
print(f"Rows after removing missing target: {len(merged_data)}")

# Check for missing values
missing_values = merged_data.isnull().sum()
missing_counts = missing_values[missing_values > 0]
threshold = 0.5 * len(merged_data)
columns_to_drop = missing_values[missing_values > threshold].index.tolist()
if columns_to_drop:
    merged_data = merged_data.drop(columns=columns_to_drop)

numeric_columns = merged_data.select_dtypes(include=['float64', 'int64']).columns
merged_data[numeric_columns] = merged_data[numeric_columns].fillna(
    merged_data[numeric_columns].mean()
)

# Feature selection - exclude pitcher columns and metadata
exclude_cols = {
    'player_id', 'year', target_column, 
    'last_name, first_name_batter', 'last_name, first_name_pitcher',
    'player_age_batter', 'player_age_pitcher'
}

feature_columns = [col for col in merged_data.columns 
                   if col not in exclude_cols 
                   and merged_data[col].dtype in ['float64', 'int64']
                   and not col.startswith('p_')]  # Exclude pitcher stats

print(f"\nUsing {len(feature_columns)} features for training")

X = merged_data[feature_columns].copy()
y = merged_data[target_column].copy()

# Remove any remaining NaN values
valid_indices = X.notna().all(axis=1) & y.notna()
X = X[valid_indices]
y = y[valid_indices]

print(f"Final training set size: X={X.shape}, y={y.shape}")

if X.empty or y.empty:
    print("No valid data for training.")
    exit()

# Scale features
scaler = StandardScaler()
try:
    X_scaled = scaler.fit_transform(X)
    print("Features scaled successfully.")
except ValueError as e:
    print(f"Error during scaling: {e}")
    exit()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# Train multiple models
print("\n" + "="*70)
print("TRAINING MODELS")
print("="*70)

# Model 1: Random Forest (for feature importance)
print("\n1. Training Random Forest Regressor...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Model 2: Gradient Boosting (better generalization)
print("2. Training Gradient Boosting Regressor...")
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=7,
    min_samples_split=10,
    random_state=42
)
gb_model.fit(X_train, y_train)

# Evaluate models
y_pred_rf = rf_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)

print("\n" + "="*70)
print("MODEL PERFORMANCE")
print("="*70)

rf_r2 = r2_score(y_test, y_pred_rf)
gb_r2 = r2_score(y_test, y_pred_gb)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
gb_mae = mean_absolute_error(y_test, y_pred_gb)

print(f"\nRandom Forest:")
print(f"  R² Score: {rf_r2:.4f}")
print(f"  MAE: {rf_mae:.4f}")

print(f"\nGradient Boosting:")
print(f"  R² Score: {gb_r2:.4f}")
print(f"  MAE: {gb_mae:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

# Save models
os.makedirs('models', exist_ok=True)

joblib.dump(rf_model, "models/baseball_rf_model.pkl")
joblib.dump(gb_model, "models/baseball_gb_model.pkl")
joblib.dump(scaler, "models/baseball_scaler.pkl")
joblib.dump(feature_columns, "models/feature_columns.pkl")

print("\n✓ Models saved successfully to models/ directory")