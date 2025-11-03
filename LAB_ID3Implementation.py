from collections import Counter
import numpy as np
import math 

class Node:
    def __init__(self,feature=None , value= None, reults=None, true_branch=None, false_branch=None):
        self.feature=feature
        self.value=value
        self.results=results
        self.true_branch=true_branch
        self.false_branch=false_branch

    def entropy(data):
        counts= np.bincount(data)
        probabilities=counts/len(data)
        entropy= np.sum([p* np.log2(p) for p in probabilites if p>0])
        return entropy

    def split_data(X,y,feature, value):
        true_indices=np.where(X[:, feature]<=value[0])
        
