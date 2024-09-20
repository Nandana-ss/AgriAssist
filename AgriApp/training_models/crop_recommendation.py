# trained model for crop recommendation 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os


crop = pd.read_csv("./AgriApp/data/Crop_recommendation.csv")

# Remove duplicate values
crop.drop_duplicates(inplace=True)

# Handle null values in dataset
attr = ["N", "P", "K", "temperature", "humidity", "rainfall", "label"]
if crop.isna().any().sum() != 0:
    for i in range(len(attr)):
        crop[attr[i]].fillna(0.0, inplace=True)

    # Remove unwanted parts from strings in a column 
crop.columns = crop.columns.str.replace(' ', '') 

    # We have given 7 features to the algorithm
features = crop[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

    # Dependent variable is crop
target = crop['label']

    # Our model will contain training and testing data
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=2)

    # Here n_estimators is the number of trees in the forest
    # random_state is for controlling the randomness of the bootstrapping
RF = RandomForestClassifier(n_estimators=20, random_state=0)

    # We'll use rf.fit to build a forest of trees from the training set (X, y)
RF.fit(x_train, y_train)
    # At this stage our algorithm is trained and ready to use

    # Calculate accuracy on the test set
y_pred = RF.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

    # Save the trained model to disk
model_dir = os.path.join('./AgriApp/models')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'cropRecommendation_model.pkl')
with open(model_path, 'wb') as model_file:
    pickle.dump(RF, model_file)



# Model accuracy: 99.09%