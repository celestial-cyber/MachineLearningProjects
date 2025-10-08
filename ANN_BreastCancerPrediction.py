# Importing essential libraries
import pandas as pd              # For handling and processing structured data (CSV)
import numpy as np               # For numerical operations
from sklearn.model_selection import train_test_split   # To split dataset into training and testing sets
from sklearn.preprocessing import StandardScaler       # To standardize feature values for better model performance
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # For model evaluation metrics
from sklearn.neural_network import MLPClassifier       # To build the Artificial Neural Network (ANN) model
from sklearn.preprocessing import LabelEncoder         # To convert categorical labels (diagnosis) into numeric form

# Load the dataset
data = pd.read_csv('breast_cancer_data.csv')  
print("Data loaded successfully")
print(data.head())   # To quickly view the first few rows and understand dataset structure
print(data.info())   # To check data types and null values
print(data.shape)    # To see the dataset dimensions (number of rows and columns)

# Separate features and target variable
# 'id' and 'diagnosis' are removed from features as:
#  - 'id' is just an identifier (not useful for learning)
#  - 'diagnosis' is the target variable we’re trying to predict
X = data.drop(columns=['id', 'diagnosis'])
y = data['diagnosis']

# Encode the target variable
# Convert categorical output ('M' for malignant, 'B' for benign) into numeric form (0 or 1)
le = LabelEncoder()
y = le.fit_transform(y)

# Split the dataset into training and testing sets
# Reason: To train the model on one subset (80%) and test performance on unseen data (20%)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature variables
# Reason: ANN models are sensitive to scale — normalization ensures all features contribute equally
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)  # Fit on training data, then transform
x_test_scaled = scaler.transform(x_test)        # Transform test data using same parameters

# Build the ANN model
# hidden_layer_sizes=(10,5) → two hidden layers with 10 and 5 neurons respectively
# activation='relu' → introduces non-linearity for complex pattern learning
# solver='adam' → optimization algorithm for faster convergence
# max_iter=1000 → ensures sufficient training iterations
# random_state=42 → ensures reproducibility
model = MLPClassifier(hidden_layer_sizes=(10,5), activation='relu', solver='adam', max_iter=1000, random_state=42)
model.fit(x_train_scaled, y_train)  # Train the ANN on scaled training data

# Make predictions on the test set
# Reason: To check how well the model generalizes on unseen data
y_pred = model.predict(x_test_scaled)

# Evaluate model performance
# accuracy_score → overall correctness
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

# Create a results dataframe to compare actual vs predicted outcomes
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.head(10))  # Display first 10 predictions for verification
