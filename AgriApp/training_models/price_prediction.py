import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import os

# Path to the dataset file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE_PATH = os.path.join('AgriApp', 'data', 'crop_data.csv')

# Load and preprocess the dataset
data = pd.read_csv(DATA_FILE_PATH)
data.columns = data.columns.str.strip()

# Define numeric and categorical features
numeric_features = ['CostCultivation', 'CostCultivation2', 'Production', 'Yield', 'Temperature', 'RainFall Annual']
categorical_features = ['State', 'Crop']

# Convert columns to numeric where necessary and fill NaN values
for col in numeric_features:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Fill NaN in numeric features with their mean
data[numeric_features] = data[numeric_features].fillna(data[numeric_features].mean())

# Define target variable
target = 'Price'
data[target] = pd.to_numeric(data[target], errors='coerce')  # Convert target to numeric
data[target].fillna(data[target].mean(), inplace=True)  # Fill NaN in target with its mean

# Define features
features = numeric_features + categorical_features

# Split the dataset
X = data[features]
y = data[target]

# Ensure there are no NaN values in X or y before proceeding
assert X.notna().all().all(), "X contains NaN values!"
assert y.notna().all(), "y contains NaN values!"

# Define your preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Create and fit the model within a pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', DecisionTreeRegressor())
])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Save the pipeline
model_dir = os.path.join(BASE_DIR, './AgriApp/models')
os.makedirs(model_dir, exist_ok=True) 
model_path = os.path.join(model_dir, 'price_prediction_model.pkl')
with open(model_path, 'wb') as file:
    pickle.dump(pipeline, file)

print("Model saved successfully.")
print(f"Saving model to: {model_path}")
