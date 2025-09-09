import pandas as pd
import numpy as np

class ClosedFormRegressionModel:

    def __init__(self,X_train,y_train):

        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = list(X_train.columns)

        
    def build_model(self):

        X = self.X_train.to_numpy()
        Y = self.y_train.to_numpy()

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        
        self.feature_names = ["Intercept"] + self.feature_names

        try :
            
            X_transpose = np.linalg.inv(X.T @ X )

        except : 
            
            X_transpose = np.linalg.pinv(X.T @ X )

        self.cofficients = X_transpose@(X.T @ Y)
        

    
    def get_prameters(self):

        
        coef_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Coefficient": self.cofficients.round(3)  
        })
        
        return coef_df
        

    
    def predict(self, X):

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        
        return X @ self.cofficients