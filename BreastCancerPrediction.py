from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt 
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC

#load the datasets
cancer=load_breast_cancer()
X=cancer.data[:,:2]
Y= cancer.target

#build the model
svm=SVC(kernel="rbf", gamma=0.5, C=1.0)
#trained the model
svm.fit(X,Y)

#plot decision boundary 
DecisionBundaryDisplay.from_estimator(
    svm,
    X,
    response_method="predict",
    cmap=plt.cm.Spectral
    alpha=0.8,
    xlabel=cancer.feature_names[0],
    ylabel=cancer.feature_names[1],
         
)

#scatter plot 
plt.scanner(X[:,0],X[:,1],
            c=y,
           s=20, edgecolors="k")
plt.show()
