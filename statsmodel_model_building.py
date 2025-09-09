import statsmodels.api as sm
import pandas as pd

class SMRegressionModel:

    def __init__(self, X_train,y_train):

        self.X_train = X_train
        self.y_train = y_train

    def build_model(self):

        X_train_sm = sm.add_constant(self.X_train)

        lr = sm.OLS(self.y_train,X_train_sm)

        self.lr = lr.fit()


    def predict(self,X):

        return self.lr.predict(sm.add_constant(X))

    def get_prameters(self):

        
        coef_df = pd.DataFrame({
            "Feature": self.lr.params.index,
            "Coefficient": self.lr.params.values.round(3)
        })
        
        return coef_df