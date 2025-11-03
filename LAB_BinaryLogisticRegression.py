import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
import matplotlib.pyplot as plt 
 
# Step 1: Generate synthetic binary data 
np.random.seed(42) 
X = np.random.randn(100, 2)                 # 100 samples, 2 features y = (X[:, 0] + X[:, 1] > 0).astype(int)     # Class 0 or 1 
 
# Step 2: Train-Test Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
 
# Step 3: Train Logistic Regression model model = LogisticRegression() 
model.fit(X_train, y_train) 
 
# Step 4: Predictions 
y_pred = model.predict(X_test) 
 
# Step 5: Evaluation 
print("Accuracy:", accuracy_score(y_test, y_pred)) 
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred)) 
print("Classification Report:\n", classification_report(y_test, y_pred)) 
 
# Step 6: Plot decision boundary x_min, x_max = X[:,0].min()-1, X[:,0].max()+1 y_min, y_max = X[:,1].min()-1, X[:,1].max()+1 xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
np.linspace(y_min, y_max, 100)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape) 
 
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm') 
plt.scatter(X[:,0], X[:,1], c=y, edgecolor='k', cmap='coolwarm') 
plt.xlabel("Feature 1") 
plt.ylabel("Feature 2") 
plt.title("Binary Logistic Regression (Decision Boundary)") 
plt.show() 
