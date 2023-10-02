import matplotlib.pyplot as plt
from typing import Dict, Union, Any
from models.linear_regression import LinearRegressionNormalEquation
import numpy as np
from sklearn.metrics import mean_squared_error
import seaborn as sns

class Visualizer:
    @staticmethod
    def evaluate_and_plot(models, X_test, y_test):
        """
        *** Feel free to modify this. You can pass the cost instance from CostFunction class or numpy array for actual and predicted data.
        """
        for name, model in models.items():
            y_pred = model.predict(X_test)
            cost = mean_squared_error(y_test, y_pred) 
            print(f"{name} Cost: {cost}")
            
            plt.scatter(y_test, y_pred, label=f"{name} (Cost: {cost:.2f})")

        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.legend()
        plt.show()

    @staticmethod
    def visualize_metrics(all_metrics, metric_names):
        for metric_name in metric_names:
            labels = []
            values = []
            
            for key, metrics in all_metrics.items():
                labels.append(key)
                values.append(metrics.get(metric_name, 0))
                
            plt.barh(labels, values)
            plt.xlabel(metric_name)
            plt.title(f'Side-by-side comparison for {metric_name}')
            plt.show()

    @staticmethod
    def visualize_boxplots(all_metrics, metric_names):
        for metric_name in metric_names:
            labels = []
            data = []
            
            for key, metrics in all_metrics.items():
                labels.append(key)
                data.append(metrics.get(metric_name, []))  # Assuming metrics now holds lists of scores
                
            plt.boxplot(data, labels=labels)
            plt.xlabel('Model and Dataset')
            plt.ylabel(metric_name)
            plt.title(f'Boxplot for {metric_name}')
            plt.xticks(rotation=45)
            plt.show()

    @staticmethod
    def visualize_heatmap(all_metrics, metric_name):
        labels = list(set(key.split('_')[0] for key in all_metrics.keys()))
        model_names = list(set(key.split('_')[1] for key in all_metrics.keys()))
        
        data_matrix = np.zeros((len(labels), len(model_names)))
        
        for i, label in enumerate(labels):
            for j, model_name in enumerate(model_names):
                key = f"{label}_{model_name}"
                metrics = all_metrics.get(key, {})
                data_matrix[i, j] = metrics.get(metric_name, np.nan)
        
        sns.heatmap(data_matrix, annot=True, xticklabels=model_names, yticklabels=labels, cmap="coolwarm")
        plt.xlabel('Models')
        plt.ylabel('Datasets')
        plt.title(f'Heatmap for {metric_name}')
        plt.show()
        
# Sample usage
all_metrics = {'Dataset_1_Model_1': {'MSE': [1.1, 1.0, 1.2], 'MAE': [0.8, 0.9, 0.7]},
               'Dataset_1_Model_2': {'MSE': [2.1, 2.0, 2.2], 'MAE': [1.0, 1.1, 1.2]},
               'Dataset_2_Model_1': {'MSE': [0.4, 0.5, 0.45], 'MAE': [0.3, 0.25, 0.35]}}

visualizer = Visualizer()
visualizer.visualize_boxplots(all_metrics, ['MSE', 'MAE'])
visualizer.visualize_heatmap(all_metrics, 'MSE')
#visualizer.visualize_parallel_coordinates(all_metrics)        