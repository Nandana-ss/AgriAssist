import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

# Path to the dataset file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE_PATH = os.path.join(BASE_DIR, 'AgriApp', 'data', 'crop_data.csv')

# Load and preprocess the dataset
data = pd.read_csv("./AgriApp/data/crop_data.csv")
data.columns = data.columns.str.strip()

# Convert columns to numeric where necessary
numeric_columns = ['CostCultivation', 'CostCultivation2', 'Production', 'Temperature', 'RainFall Annual', 'Price']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Encode categorical variables
label_encoder_state = LabelEncoder()
label_encoder_crop = LabelEncoder()
data['State'] = label_encoder_state.fit_transform(data['State'].str.strip())
data['Crop'] = label_encoder_crop.fit_transform(data['Crop'].str.strip())

# Define features and target
features = ['State', 'Crop', 'CostCultivation', 'CostCultivation2', 'Production', 'Temperature', 'RainFall Annual']
target = 'Price'

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(data[features])
y = SimpleImputer(strategy='mean').fit_transform(data[target].values.reshape(-1, 1)).ravel()

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model and preprocessing objects
model_dir = os.path.join(BASE_DIR, 'models')
os.makedirs(model_dir, exist_ok=True) 
model_path = os.path.join(model_dir, 'price_prediction_model.pkl')
with open(model_path, 'wb') as file:
    pickle.dump({
        'model': model,
        'imputer': imputer,
        'label_encoder_state': label_encoder_state,
        'label_encoder_crop': label_encoder_crop
    }, file)

print("Model saved successfully.")
print(f"Saving model to: {model_path}")