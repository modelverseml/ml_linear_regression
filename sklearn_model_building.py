from sklearn.linear_model import LinearRegression
import pandas as pd

class SkLearnRegressionModel:

    def __init__(self, X_train,y_train):

        self.X_train = X_train
        self.y_train = y_train

    def build_model(self):

        lr = LinearRegression()

        self.lr = lr.fit(self.X_train,self.y_train)

    def predict(self,X):

        return self.lr.predict(X)

    def get_prameters(self):

        
        coef_df = pd.DataFrame({
            "Feature": self.X_train.columns,
            "Coefficient": self.lr.coef_.round(3)  
        })
        
        coef_df.loc[len(coef_df)] = ["Intercept", round(self.lr.intercept_, 3)]
        
        return coef_df
        