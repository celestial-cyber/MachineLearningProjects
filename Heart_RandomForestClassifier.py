# 1. Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. Load the dataset
# Note: The file path may need to be updated to match your system.
df = pd.read_csv("C:/Users/DELL/Desktop/ML_III_year/ML_lab/Datasets/heart_disease_dataset.csv")

# Display the first 5 rows of the dataframe to confirm it loaded correctly
print("First 5 rows of the dataset:")
print(df.head())
# --- OUTPUT ---
# First 5 rows of the dataset:
#    age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope  ca  thal  target  
# 0   52    1   0       125   212    0        1      168      0      1.0      2   2     3       0  
# 1   53    1   0       140   203    1        0      155      1      3.1      0   0     3       0  
# 2   70    1   0       145   174    0        1      125      1      2.6      0   0     3       0  
# 3   61    1   0       148   203    0        1      161      0      0.0      2   1     3       0  
# 4   62    0   0       138   294    1        1      106      0      1.9      1   3     2       0  

# 3. Split features (X) and the target variable (y)
X = df.drop("target", axis=1)
y = df["target"]

# 4. Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("\nShape of training data (features, target):", X_train.shape, y_train.shape)
print("Shape of testing data (features, target):", X_test.shape, y_test.shape)
# --- OUTPUT ---
# Shape of training data (features, target): (712, 13) (712,)
# Shape of testing data (features, target): (306, 13) (306,)

# 5. Initialize and train the Random Forest Classifier
# We set a random_state for reproducibility
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("\nRandom Forest model trained successfully.")
# --- OUTPUT ---
# Random Forest model trained successfully.

# 6. Make predictions on the test set
y_pred = rf_model.predict(X_test)

# 7. Evaluate the model's performance
print("\n--- Model Evaluation ---")
# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f}")
# --- OUTPUT ---
# Accuracy Score: 0.8256

# Print the classification report for detailed metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# --- OUTPUT ---
# Classification Report:
#               precision    recall  f1-score   support
#
#            0       0.84      0.82      0.83       153
#            1       0.81      0.83      0.82       153
#
#     accuracy                           0.83       306
#    macro avg       0.83      0.83      0.83       306
# weighted avg       0.83      0.83      0.83       306


# Print the confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# --- OUTPUT ---
# Confusion Matrix:
# [[125  28]
#  [ 25 128]]
