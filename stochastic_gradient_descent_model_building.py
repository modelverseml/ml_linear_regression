
import pandas as pd
import numpy as np

class StochasticGradientDescentRegressionModel:

    def __init__(self,X_train,y_train,epochs,batch_size):

        self.X_train = X_train
        self.y_train = y_train
        self.epochs = epochs
        self.batch_size = batch_size
        self.feature_names = list(X_train.columns)

    def build_model(self):

        X = self.X_train.to_numpy()
        Y = self.y_train.to_numpy()

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        
        self.feature_names = ["Intercept"] + self.feature_names

        b = np.zeros(X.shape[1])
        alpha=0.01

        n_samples = X.shape[0]

        for _ in range(self.epochs):
            # Shuffle the dataset each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
    
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]
                
                # Compute gradient for the batch
                gradient = X_batch.T @ (X_batch @ b - Y_batch) / X_batch.shape[0]
                b = b - alpha * gradient
            

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