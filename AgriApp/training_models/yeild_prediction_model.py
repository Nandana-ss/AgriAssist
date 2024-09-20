import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import pickle
plt.style.use("ggplot")

# Load and clean dataset
df = pd.read_csv("./AgriApp/data/yield_df.csv")
df.drop('Unnamed: 0', axis=1, inplace=True)
df.drop_duplicates(inplace=True)

# Calculate yield per country
country = df['Area'].unique()
yield_per_country = [df[df['Area'] == state]['hg/ha_yield'].sum() for state in country]

# Calculate yield per crop
crops = df['Item'].unique()
yield_per_crop = [df[df['Item'] == crop]['hg/ha_yield'].sum() for crop in crops]

# Define features and target
X = df.drop('hg/ha_yield', axis=1)
y = df['hg/ha_yield']

# Split data into train and test sets, ensuring X_train and X_test remain DataFrames
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

# Define preprocessor using column names directly
ohe = OneHotEncoder(drop='first')
scale = StandardScaler()

preprocesser = ColumnTransformer(
    transformers=[
        ('StandardScale', scale, ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']),  # Numeric columns
        ('OneHotEncodeing', ohe, ['Area', 'Item'])  # Categorical columns
    ],
    remainder='passthrough'
)

# Preprocess training and test data (X_train and X_test are still DataFrames here)
X_train_dummy = preprocesser.fit_transform(X_train)
X_test_dummy = preprocesser.transform(X_test)

# Get feature names (optional)
preprocesser.get_feature_names_out()

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Lasso': Lasso(),
    'Ridge': Ridge(),
    'Decision Tree': DecisionTreeRegressor(),
    'KNN': KNeighborsRegressor(),
}

# Train and evaluate models
for name, md in models.items():
    md.fit(X_train_dummy, y_train)
    y_pred = md.predict(X_test_dummy)
    print(f"{name}: MAE : {mean_absolute_error(y_test, y_pred)} | R2 Score : {r2_score(y_test, y_pred)}")

# Train Decision Tree Regressor for the final model
dtr = DecisionTreeRegressor()
dtr.fit(X_train_dummy, y_train)

# Define prediction function
def prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item):
    # Keep the features as a DataFrame to match the structure used during training
    features = pd.DataFrame([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]],
                            columns=['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item'])
    transform_features = preprocesser.transform(features)
    predicted_yield = dtr.predict(transform_features).reshape(-1, 1)
    return predicted_yield[0][0]

# Test the prediction function with an example input
result = prediction(1990, 1485.0, 121.0, 16.37, "Albania", "Maize")
print(f"Predicted yield: {result} hg/ha")

# Save model and preprocessor
with open('./AgriApp/models/decision_tree_regressor.pkl', 'wb') as f:
    pickle.dump(dtr, f)

with open('./AgriApp/models/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocesser, f)