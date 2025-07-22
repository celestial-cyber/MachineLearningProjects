import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create custom dataset
data = {
    'Mileage': [20000, 30000, 40000, 50000, 60000, 25000, 45000, 35000, 15000, 55000],
    'Year': [2018, 2017, 2016, 2015, 2014, 2019, 2016, 2017, 2020, 2015],
    'Fuel_Type': ['Petrol', 'Diesel', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Petrol', 'Diesel'],
    'Transmission': ['Manual', 'Automatic', 'Manual', 'Manual', 'Automatic', 'Manual', 'Automatic', 'Manual', 'Automatic', 'Manual'],
    'Owners': [1, 2, 1, 1, 3, 1, 2, 1, 1, 2],
    'Price': [500000, 450000, 400000, 370000, 330000, 520000, 390000, 430000, 550000, 360000]
}

df = pd.DataFrame(data)

# Step 2: Features and Target
X = df.drop('Price', axis=1)
y = df['Price']

# Step 3: Preprocessing for categorical features
categorical_features = ['Fuel_Type', 'Transmission']
numeric_features = ['Mileage', 'Year', 'Owners']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), categorical_features)
], remainder='passthrough')  # Pass numeric features as-is

# Step 4: Create Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 6: Train the model
model.fit(X_train, y_train)

# Step 7: Predict and evaluate
y_pred = model.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Step 8: Predict a new car's price
new_car = pd.DataFrame({
    'Mileage': [30000],
    'Year': [2018],
    'Fuel_Type': ['Petrol'],
    'Transmission': ['Manual'],
    'Owners': [1]
})

predicted_price = model.predict(new_car)
print("Predicted Price for New Car: ₹", round(predicted_price[0]))
