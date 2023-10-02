import numpy as np
from sklearn.metrics import r2_score
class CostFunction:
    def __init__(self, type_of_cost):
        """
        Initialize the CostFunction class with the type of cost function to use.

        Parameters:
            type_of_cost (str): The type of cost function to use ("MAE", "MSE", "RMSE")
        """
        self.type = type_of_cost.upper()

    def compute_cost(self, y_true, y_pred):
        """
        Compute the cost given the true labels and predicted labels.

        Parameters:
            y_true (numpy.array): True labels
            y_pred (numpy.array): Predicted labels

        Returns:
            float: The computed cost
        """
        if self.type == "MAE":
            return self.mean_absolute_error(y_true, y_pred)
        elif self.type == "MSE":
            return self.mean_squared_error(y_true, y_pred)
        elif self.type == "RMSE":
            return self.root_mean_squared_error(y_true, y_pred)
        elif self.type == "R_SQUARED":
            return self.r_squared_error(y_true, y_pred)
        else:
            raise ValueError("Invalid cost function")

    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        """
        Compute the Mean Absolute Error.

        Parameters:
            y_true (numpy.array): True labels
            y_pred (numpy.array): Predicted labels

        Returns:
            float: The Mean Absolute Error
        """
        abso=np.abs(y_true-y_pred)
        mae=np.mean(abso)
        # mae=mean_absolute_error(y_true,y_pred)
        return mae
        raise NotImplementedError("The mean_absolute_error function not implemented yet")


    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """
        Compute the Mean Squared Error.

        Parameters:
            y_true (numpy.array): True labels
            y_pred (numpy.array): Predicted labels

        Returns:
            float: The Mean Squared Error
        """
        sq_rt=np.square(y_true-y_pred)
        mse=np.mean(sq_rt)
        # mse=mean_squared_error(y_true,y_pred)
        return mse
        raise NotImplementedError("The mean_squared_error function has not been implemented yet.")


    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        """
        Compute the Root Mean Squared Error.

        Parameters:
            y_true (numpy.array): True labels
            y_pred (numpy.array): Predicted labels

        Returns:
            float: The Root Mean Squared Error
        """
        sq_rt=np.square(y_true-y_pred)
        mse=np.mean(sq_rt)
        rmse=np.sqrt(mse)
        # mse=mean_squared_error(y_true,y_pred)
        # rmse=np.sqrt(mse)
        return rmse
        raise NotImplementedError("The root_mean_squared_error function has not been implemented yet.")


    @staticmethod
    def r_squared_error(y_true, y_pred):
        """
        Compute the Root Mean Squared Error.

        Parameters:
            y_true (numpy.array): True labels
            y_pred (numpy.array): Predicted labels

        Returns:
            float: The Root Mean Squared Error
        """
        
        r2=r2_score(y_true,y_pred)
        return r2
        raise NotImplementedError("The r_squared_error function has not been implemented yet.")

# Here is an example to make use of this class

# True and predicted labels
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

# Initialize CostFunction class
cost_fn = CostFunction("MAE")

# Compute cost
cost = cost_fn.compute_cost(y_true, y_pred)
print(f"Computed Cost (MAE): {cost}")

# Change the cost function type to MSE
cost_fn.type = "MSE"
cost = cost_fn.compute_cost(y_true, y_pred)
print(f"Computed Cost (MSE): {cost}")

# Change the cost function type to RMSE
cost_fn.type = "RMSE"
cost = cost_fn.compute_cost(y_true, y_pred)
print(f"Computed Cost (RMSE): {cost}")

cost_fn.type = "R_SQUARED"
cost = cost_fn.compute_cost(y_true, y_pred)
print(f"Computed Cost (R_SQUARED): {cost}")