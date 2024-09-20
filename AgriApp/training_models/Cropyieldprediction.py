import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import pickle
plt.style.use("ggplot")

# Define the base directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the full file path
csv_file_path = os.path.join('./AgriApp/data/yield_df.csv')

# Load dataset
df = pd.read_csv(csv_file_path)
df.drop('Unnamed: 0', axis=1, inplace=True)

# Select relevant columns
col = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item', 'hg/ha_yield']
df = df[col]

# Split dataset
X = df.drop('hg/ha_yield', axis=1)
y = df['hg/ha_yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

# Define preprocesser with handle_unknown='ignore' for unseen categories
ohe = OneHotEncoder(drop='first', handle_unknown='ignore')
scale = StandardScaler()

preprocesser = ColumnTransformer(
    transformers=[
        ('StandardScale', scale, [0, 1, 2, 3]),  # Scaling numerical features
        ('OneHotEncodeing', ohe, [4, 5])        # Encoding 'Area' and 'Item' columns
    ],
    remainder='passthrough'
)

# Fit and transform training data
X_train_dummy = preprocesser.fit_transform(X_train)

# Transform test data (no re-fitting)
X_test_dummy = preprocesser.transform(X_test)

# Get feature names after transformation
preprocesser.get_feature_names_out()

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Lasso': Lasso(),
    'Ridge': Ridge(),
    'Decision Tree': DecisionTreeRegressor(),
    'KNN': KNeighborsRegressor(),
}

# Fit models and calculate metrics
for name, md in models.items():
    md.fit(X_train_dummy, y_train)
    y_pred = md.predict(X_test_dummy)
    print(f"{name}: MAE: {mean_absolute_error(y_test, y_pred)}, R² Score: {r2_score(y_test, y_pred)}")

# Train Decision Tree Regressor as the final model
dtr = DecisionTreeRegressor()
dtr.fit(X_train_dummy, y_train)

# Predictive system function
def prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item):
    features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
    
    # Safely transform features with the updated preprocessor
    transform_features = preprocesser.transform(features)
    
    # Predict the yield
    predicted_yield = dtr.predict(transform_features).reshape(-1, 1)
    return predicted_yield[0][0]

# Example prediction
result = prediction(1990, 1485.0, 121.0, 16.37, "Albania", "Maize")
print(f"Predicted yield: {result}")

# Define the directory where you want to save the models
save_dir = os.path.join(BASE_DIR, 'AgriApp', 'models')

# Create the directories if they don't exist
os.makedirs(save_dir, exist_ok=True)

# Save the decision tree regressor model using pickle
with open(os.path.join(save_dir, 'decision_tree_regressor.pkl'), 'wb') as model_file:
    pickle.dump(dtr, model_file)

# Save the preprocessor using pickle
with open(os.path.join(save_dir, 'preprocessor.pkl'), 'wb') as preprocessor_file:
    pickle.dump(preprocesser, preprocessor_file)

# Print scikit-learn version for reference
import sklearn
print(f"Scikit-learn version: {sklearn.__version__}")
