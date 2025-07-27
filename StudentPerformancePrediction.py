import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create custom dataset
data = {
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Attendance_Rate': [60, 65, 70, 75, 80, 85, 90, 92, 95, 98],
    'Assignments_Completed': [2, 3, 4, 5, 5, 6, 7, 8, 9, 10],
    'Prior_Score': [40, 45, 50, 55, 60, 65, 70, 75, 80, 85],
    'Final_Exam_Score': [45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
}

df = pd.DataFrame(data)

# Step 2: Define features and target
X = df.drop('Final_Exam_Score', axis=1)
y = df['Final_Exam_Score']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict on test set
y_pred = model.predict(X_test)

# Step 6: Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Step 7: Predict new student's score
new_student = pd.DataFrame({
    'Study_Hours': [6],
    'Attendance_Rate': [88],
    'Assignments_Completed': [7],
    'Prior_Score': [68]
})

predicted_score = model.predict(new_student)
print("Predicted Final Exam Score for new student:", predicted_score[0])
