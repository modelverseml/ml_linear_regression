from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class RegressionMetrics:

    def __init__(self,actual,predict):

        self.actual = actual
        self.predict = predict

    def get_metrics(self):

        print("r2 Score ",round(r2_score(self.actual,self.predict),2))
        print("Mean Squared Error ",round(mean_squared_error(self.actual,self.predict),2))
        print("Mean Absolute Error ",round(mean_absolute_error(self.actual,self.predict),2))

    def plot_residuals(self):

        residuals = self.actual - self.predict

        plt.scatter(self.predict, residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Fitted")
        plt.show()


        sns.histplot(residuals, kde=True)
        plt.title("Residuals Distribution")
        plt.xlabel("Residual")
        plt.show()

        
        