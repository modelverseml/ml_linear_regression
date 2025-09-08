

class RMClosedSolution:

    def __init__(self,X_train,y_train):

        self.X_train = X_train
        self.y_train = y_train

    def get_coefficients(self):

        X = self.X_train.to_numpy()
        Y = self.y_train.to_numpy()
        self.cofficients = np.linalg.inv(X.T @ X )@(X.T @ Y)

    def predict(self,X_train):

        return 