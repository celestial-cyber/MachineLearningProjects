import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Create a custom dataset
data = {
    'Age': [25, 45, 35, 50, 29, 60, 40, 30, 55, 33],
    'Blood_Pressure': [120, 140, 130, 150, 118, 160, 135, 125, 155, 132],
    'Cholesterol': [190, 230, 210, 250, 180, 270, 220, 200, 260, 215],
    'Symptom_Fever': [0, 1, 1, 1, 0, 1, 1, 0, 1, 1],
    'Symptom_Cough': [1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    'Has_Disease': [0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Step 2: Define features and target
X = df.drop('Has_Disease', axis=1)
y = df['Has_Disease']

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Predict on test data
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Predict for a new patient
new_patient = pd.DataFrame({
    'Age': [42],
    'Blood_Pressure': [145],
    'Cholesterol': [240],
    'Symptom_Fever': [1],
    'Symptom_Cough': [1]
})

prob = model.predict_proba(new_patient)[0][1]
prediction = model.predict(new_patient)[0]

print(f"\nProbability of having disease: {prob:.2f}")
print(f"Prediction: {'Has Disease' if prediction == 1 else 'No Disease'}")
