import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Create dataset
data = {
    'Mileage': [20000, 30000, 40000, 50000, 60000, 25000, 45000, 35000, 15000, 55000],
    'Year': [2018, 2017, 2016, 2015, 2014, 2019, 2016, 2017, 2020, 2015],
    'Fuel_Type': ['Petrol', 'Diesel', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Petrol', 'Diesel'],
    'Transmission': ['Manual', 'Automatic', 'Manual', 'Manual', 'Automatic', 'Manual', 'Automatic', 'Manual', 'Automatic', 'Manual'],
    'Owners': [1, 2, 1, 1, 3, 1, 2, 1, 1, 2],
    'Price': [500000, 450000, 400000, 370000, 330000, 520000, 390000, 430000, 550000, 360000]
}

df = pd.DataFrame(data)

# Step 2: Create a binary target: 1 if Price > 450000 else 0
df['High_Price'] = df['Price'].apply(lambda x: 1 if x > 450000 else 0)

# Step 3: Features and target
X = df.drop(['Price', 'High_Price'], axis=1)
y = df['High_Price']

# Step 4: Preprocessing
categorical_features = ['Fuel_Type', 'Transmission']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), categorical_features)
], remainder='passthrough')

# Step 5: Pipeline with Logistic Regression
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Step 6: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train
model.fit(X_train, y_train)

# Step 8: Predict and evaluate
y_pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 9: Predict a new sample
new_car = pd.DataFrame({
    'Mileage': [25000],
    'Year': [2019],
    'Fuel_Type': ['Petrol'],
    'Transmission': ['Manual'],
    'Owners': [1]
})

prob = model.predict_proba(new_car)[0][1]
prediction = model.predict(new_car)[0]

print(f"Probability of price > ₹4.5 Lakhs: {prob:.2f}")
print(f"Class Prediction: {'High Price' if prediction == 1 else 'Low Price'}")
