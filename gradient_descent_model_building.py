
import pandas as pd
import numpy as np

class GradientDescentRegressionModel:

    def __init__(self,X_train,y_train):

        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = list(X_train.columns)

    def build_model(self):

        X = self.X_train.to_numpy()
        Y = self.y_train.to_numpy()

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        
        self.feature_names = ["Intercept"] + self.feature_names

        alpha = 0.1
        b = [0]*X.shape[1]
        
        for _ in range(10000):
            b = b - alpha * (X.T @(X@b - Y)/X.shape[0])
            

        self.cofficients = b
        

    def get_prameters(self):

        
        coef_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Coefficient": self.cofficients.round(3)  
        })
        
        return coef_df
        

    
    def predict(self, X):

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        
        return X @ self.cofficients