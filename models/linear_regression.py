import numpy as np
from abc import ABC, abstractmethod
from models.cost_function import CostFunction

class Model(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, wandb_instance=None):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def cost_function(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

class LinearRegressionNormalEquation(Model):
    def __init__(self,cost_function_type="MSE"):
        self.theta = None
        self.cost_function_instance = CostFunction(cost_function_type)

    def fit(self, X: np.ndarray, y: np.ndarray, wandb_instance=None) -> None:
        # Adding a bias column of ones
        X_bias = np.insert(X, 0, 1, axis=1)
        
        # Normal Equation formula: theta = inv(X' * X) * X' * y
        X_transpose = np.transpose(X_bias)
        X_transpose_dot_X = np.dot(X_transpose, X_bias)
        inv_X_transpose_dot_X = np.linalg.inv(X_transpose_dot_X)
        X_transpose_dot_y = np.dot(X_transpose, y)

        self.theta = np.dot(inv_X_transpose_dot_X, X_transpose_dot_y)
        
        if wandb_instance:
            wandb_instance.log({"theta": self.theta.tolist()})

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_bias = np.insert(X, 0, 1, axis=1)
        return np.dot(X_bias, self.theta)
    
    def cost_function(self, y_true: np.ndarray, y_pred: np.ndarray, wandb_instance=None) -> float:
        m = len(y_true)
        cost = self.cost_function_instance.compute_cost(y_true, y_pred)

        if wandb_instance:
            wandb_instance.log({"cost": cost})
        
        return cost
    
class LinearRegressionSimple(Model):
    def __init__(self, learning_rate=0.01, num_iterations=20000):
        self.theta = None
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def fit(self, X: np.ndarray, y: np.ndarray, wandb_instance=None):
        m, n = X.shape
        self.theta = np.zeros(n + 1) 
        X_bias = np.insert(X, 0, 1, axis=1)
        
        for _ in range(self.num_iterations):
            y_pred = np.dot(X_bias, self.theta)
            gradient = (1 / m) * np.dot(X_bias.T, (y_pred - y))
            self.theta -= self.learning_rate * gradient
        print('Theta: ',self.theta.tolist())
        if wandb_instance:
            wandb_instance.log({"theta": self.theta.tolist()})
        # raise NotImplementedError("Method not implemented")

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_bias = np.insert(X, 0, 1, axis=1)
        return np.dot(X_bias, self.theta)
        raise NotImplementedError("Method not implemented")

    def cost_function(self, y_true: np.ndarray, y_pred: np.ndarray,wandb_instance=None) -> float:
        m = len(y_true)
        # Calculate Mean Squared Error as the cost
        cost = np.sum((y_pred - y_true) ** 2) / (2 * m)

        if wandb_instance:
            wandb_instance.log({"cost": cost})
        
        return cost
        raise NotImplementedError("Method not implemented")

class LinearRegressionAbsoluteTrick(Model):
    def __init__(self, learning_rate=0.01, num_iterations=20000):
        self.theta = None
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def fit(self, X: np.ndarray, y: np.ndarray, wandb_instance=None):
        m, n = X.shape
        self.theta = np.zeros(n + 1)
        X_bias = np.insert(X, 0, 1, axis=1)
        
        for _ in range(self.num_iterations):
            y_pred = np.dot(X_bias, self.theta)
            gradient = (1 / m) * np.dot(X_bias.T, np.sign(y_pred - y))
            self.theta -= self.learning_rate * gradient
        if wandb_instance:
            wandb_instance.log({"theta": self.theta.tolist()})

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_bias = np.insert(X, 0, 1, axis=1)
        return np.dot(X_bias, self.theta)
        raise NotImplementedError("LinearRegressionAbsoluteTrick :: Method not implemented")

    def cost_function(self, y_true: np.ndarray, y_pred: np.ndarray,wandb_instance=None) -> float:
        m = len(y_true)
        cost = np.sum(np.abs(y_pred - y_true)) / m
        if wandb_instance:
            wandb_instance.log({"cost": cost})
        return cost
        raise NotImplementedError("LinearRegressionAbsoluteTrick :: Method not implemented")

class LinearRegressionSquareTrick(Model):
    def __init__(self, learning_rate=0.01, num_iterations=20000):
        self.theta = None
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
    def fit(self, X: np.ndarray, y: np.ndarray, wandb_instance=None):
        m, n = X.shape
        self.theta = np.zeros(n + 1)
        X_bias = np.insert(X, 0, 1, axis=1)
        
        for _ in range(self.num_iterations):
            y_pred = np.dot(X_bias, self.theta)
            gradient = (1 / m) * np.dot(X_bias.T, (y_pred - y))
            self.theta -= self.learning_rate * gradient
        if wandb_instance:
            wandb_instance.log({"theta": self.theta.tolist()})

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_bias = np.insert(X, 0, 1, axis=1)
        return np.dot(X_bias, self.theta)
        raise NotImplementedError("LinearRegressionSquareTrick :: Method not implemented")

    def cost_function(self, y_true: np.ndarray, y_pred: np.ndarray,wandb_instance=None) -> float:
        m = len(y_true)
        cost = np.sum((y_pred - y_true) ** 2) / (2 * m)
        if wandb_instance:
            wandb_instance.log({"cost": cost})
        return cost
        raise NotImplementedError("LinearRegressionSquareTrick :: Method not implemented")
    
    
    
# This is extra credit
from sklearn.base import BaseEstimator, RegressorMixin

class CustomSkLearnLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def fit(self, X, y, sample_weight=None):
        # Custom fitting logic here
        print("Custom fit method called")
        return super().fit(X, y, sample_weight)
    
    def predict(self, X):
        # Custom prediction logic here
        print("Custom predict method called")
        return super().predict(X)
    
    def custom_cost_function(self, y_true, y_pred):
        # Custom cost function here
        return np.sum(np.abs(y_true - y_pred))    
    
    def score(self, X, y, sample_weight=None):
        # Custom scoring logic here
        print("Custom score method called")
        return super().score(X, y, sample_weight)


    
# Here is an use case scenario
# Generate some example training data
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2, 4, 3, 3.5, 5])

# Generate some example test data
X_test = np.array([[1.5], [2.5], [3.5], [4.5]])
y_test = np.array([2.5, 3.7, 3.5, 4.8])

# Initialize the linear regression model using the normal equation method
# with MSE as the cost function
linear_model = LinearRegressionNormalEquation(cost_function_type="MSE")

# Fit the model to the training data
linear_model.fit(X_train, y_train)

# Make predictions on test data
y_pred = linear_model.predict(X_test)

# Compute the cost of these predictions using the cost function specified (MSE)
cost = linear_model.cost_function(y_test, y_pred)

# Output the results
print("Predictions:", y_pred)
print("Cost:", cost)

# Initialize the linear regression model using MAE as the cost function
linear_model_mae = LinearRegressionNormalEquation(cost_function_type="MAE")

# Fit the model to the training data
linear_model_mae.fit(X_train, y_train)

# Make predictions on test data
y_pred_mae = linear_model_mae.predict(X_test)

# Compute the cost of these predictions using the cost function specified (MAE)
cost_mae = linear_model_mae.cost_function(y_test, y_pred_mae)

# Output the results
print("Predictions with MAE:", y_pred_mae)
print("Cost with MAE:", cost_mae)

linear_model = LinearRegressionSimple()

# Fit the model to the training data
linear_model.fit(X_train, y_train)

# Make predictions on test data
y_pred = linear_model.predict(X_test)

# Compute the cost of these predictions using the cost function specified (MSE)
cost = linear_model.cost_function(y_test, y_pred)

# Output the results
print("Predictions:", y_pred)
print("Cost:", cost)

# Initialize the linear regression model using MAE as the cost function
linear_model_mae = LinearRegressionSimple()

# Fit the model to the training data
linear_model_mae.fit(X_train, y_train)

# Make predictions on test data
y_pred_mae = linear_model_mae.predict(X_test)

# Compute the cost of these predictions using the cost function specified (MAE)
cost_mae = linear_model_mae.cost_function(y_test, y_pred_mae)

# Output the results
print("Predictions with simple trick:", y_pred_mae)
print("Cost with MSE:", cost_mae)

linear_model_mae = LinearRegressionAbsoluteTrick()

# Fit the model to the training data
linear_model_mae.fit(X_train, y_train)

# Make predictions on test data
y_pred_mae = linear_model_mae.predict(X_test)

# Compute the cost of these predictions using the cost function specified (MAE)
cost_mae = linear_model_mae.cost_function(y_test, y_pred_mae)

# Output the results
print("Predictions with Absolute trick:", y_pred_mae)
print("Cost with MSE:", cost_mae)

linear_model_mae = LinearRegressionSquareTrick()

# Fit the model to the training data
linear_model_mae.fit(X_train, y_train)

# Make predictions on test data
y_pred_mse = linear_model_mae.predict(X_test)

# Compute the cost of these predictions using the cost function specified (MAE)
cost_mae = linear_model_mae.cost_function(y_test, y_pred_mse)

# Output the results
print("Predictions with Square Trick:", y_pred_mse)
print("Cost with MSE:", cost_mae)