import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Load your CSV data
csv_data = pd.read_csv('./AgriApp/data/plant_growth_data.csv')

# Define mappings for categorical features
soil_type_mapping = {'sandy': 0, 'loam': 1, 'clay': 2}  # Example mappings
water_frequency_mapping = {'bi-weekly': 0, 'weekly': 1, 'daily': 2}
fertilizer_type_mapping = {'organic': 0, 'chemical': 1, 'none': 2}

# Apply mappings to the DataFrame
csv_data['Soil_Type'] = csv_data['Soil_Type'].map(soil_type_mapping)
csv_data['Water_Frequency'] = csv_data['Water_Frequency'].map(water_frequency_mapping)
csv_data['Fertilizer_Type'] = csv_data['Fertilizer_Type'].map(fertilizer_type_mapping)

# Define numerical columns
numerical_columns = ['Sunlight_Hours', 'Temperature', 'Humidity']

# Preprocessing for numerical data
numerical_preprocessor = StandardScaler()

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_preprocessor, numerical_columns)
    ])

# Create a pipeline with preprocessing and model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())
])

# Split the data into train and test sets
X = csv_data.drop('Growth_Milestone', axis=1)
y = csv_data['Growth_Milestone']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model_pipeline.fit(X_train, y_train)

# Save the model using joblib
with open('./AgriApp/models/csv_growth_model.pkl', 'wb') as f:
    joblib.dump(model_pipeline, f)

print("Model training complete and saved as csv_growth_model.pkl")
