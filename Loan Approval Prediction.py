import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Custom Dataset
data = {
    'Income': [30, 60, 25, 80, 40, 90, 35, 50, 75, 20],
    'Employment_Status': [1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
    'Credit_Score': [700, 750, 620, 800, 680, 820, 610, 690, 770, 580],
    'Loan_Amount': [15, 25, 10, 20, 18, 30, 12, 20, 28, 10],
    'Loan_Approved': [1, 1, 0, 1, 1, 1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

# Step 2: Feature matrix and target
X = df.drop('Loan_Approved', axis=1)
y = df['Loan_Approved']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Predictions and Evaluation
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 6: Predict a new application
new_application = pd.DataFrame({
    'Income': [55],
    'Employment_Status': [1],
    'Credit_Score': [710],
    'Loan_Amount': [18]
})

approval_probability = model.predict_proba(new_application)[0][1]
approval_prediction = model.predict(new_application)[0]

print(f"\nApproval Probability: {approval_probability:.2f}")
print(f"Prediction: {'Approved' if approval_prediction == 1 else 'Rejected'}")
