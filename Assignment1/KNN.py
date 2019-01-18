import numpy as np 
import csv

class KNN(object):
    def __init__(self):
        pass
    
    def train(self,X,y):
        self.X_train = X
        self.y_train = y
    
    def predict(self,X,k=1):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test,num_train))
        dists = np.sqrt(- 2 * np.dot( X , self.X_train.T ) + np.sum( X * X , 1).reshape(num_test,1) + np.sum( self.X_train * self.X_train , 1))
        y_pred = np.zeros(num_test)
        for i in list(range(num_test)):
            closest_y = []
            closest_y = self.y_train[np.argsort(dists[i])][:k]
            if np.shape(np.shape(closest_y))[0] !=1:  
                closest_y=np.squeeze(closest_y)
            y_pred[i] = np.argmax(np.bincount(closest_y))
        return y_pred


    
