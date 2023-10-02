from models.linear_regression import LinearRegressionNormalEquation
from models.linear_regression import LinearRegressionSimple
from models.linear_regression import LinearRegressionAbsoluteTrick
from models.linear_regression import LinearRegressionSquareTrick
from utils.data_loader import DataLoader
from utils.standard_scaler import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from utils.benchmarking import BasicBenchmark
#from utils.visualization.viz_benchmark import visualize_metrics, evaluate_and_plot
import wandb
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# def cost_function(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#     m = len(y_true)
#     return (1 / (2 * m)) * np.sum(np.square(y_pred - y_true))

def main() -> None:
    # Initialize wandb if you want to use it
    use_wandb: bool = False
    wandb_init = wandb.init(project='model_benchmarking', reinit=True) if use_wandb else None

    # # Load and preprocess the data
    data = pd.read_csv('data/advertising.csv', index_col=0)
    
    X = data.drop('target', axis=1).values
    y = data['target'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    datasets = [(X_train, X_test, y_train, y_test)]

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Custom Model
    custom_model = LinearRegressionNormalEquation()
    custom_model.fit(X_train, y_train)
    predictiond1= custom_model.predict(X_test)
    # print(predictiond1)
    # print(predictiond1.size)

    custom_model = LinearRegressionSimple()
    custom_model.fit(X_train, y_train)
    simple_predictiond1= custom_model.predict(X_test)
    # print(simple_predictiond1)
    # print(simple_predictiond1.size)
    
    custom_model = LinearRegressionSquareTrick()
    custom_model.fit(X_train, y_train)
    square_predictiond1= custom_model.predict(X_test)
    print(square_predictiond1)
    print(square_predictiond1.size)

    custom_model = LinearRegressionAbsoluteTrick()
    custom_model.fit(X_train, y_train)
    absolute_predictiond1= custom_model.predict(X_test)
    print(absolute_predictiond1)
    print(absolute_predictiond1.size)

    sk_model = LinearRegression()
    sk_model.fit(X_train, y_train)
    sklearn_predictiond1=sk_model.predict(X_test)
    print(sklearn_predictiond1)

    mae_simple = mean_absolute_error(y_test, sklearn_predictiond1)
    mae_absolute = mean_absolute_error(y_test, absolute_predictiond1)
    mae_square = mean_absolute_error(y_test, square_predictiond1)

    mse_simple = mean_squared_error(y_test, sklearn_predictiond1)
    mse_absolute = mean_squared_error(y_test, absolute_predictiond1)
    mse_square = mean_squared_error(y_test, square_predictiond1)

    r2_simple = r2_score(y_test, sklearn_predictiond1)
    r2_absolute = r2_score(y_test, absolute_predictiond1)
    r2_square = r2_score(y_test, square_predictiond1)

    plt.scatter(y_test, sklearn_predictiond1, label='Sklearn Regression advertising',)
    plt.scatter(y_test, absolute_predictiond1, label='Absolute Regression advertising')
    plt.scatter(y_test, square_predictiond1, label='Square Regression advertising',alpha=0.5)

    plt.xlabel('Actual Values advertising')
    plt.ylabel('Predicted Values advertising')
    plt.legend()
    plt.show()

    labels = ['MAE', 'MSE', 'R2']
    simple_metrics = [mae_simple, mse_simple, r2_simple]
    absolute_metrics = [mae_absolute, mse_absolute, r2_absolute]
    square_metrics = [mae_square, mse_square, r2_square]

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, simple_metrics, width, label='Sklearn')
    rects2 = ax.bar(x, absolute_metrics, width, label='Absolute')
    rects3 = ax.bar(x + width, square_metrics, width, label='Square')

    ax.set_ylabel('Scores')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()

    metrics = {'MSE': mean_squared_error}
    benchmark = BasicBenchmark()

    all_metrics = benchmark.evaluate_multi(models=[custom_model, sk_model], 
                                           datasets=datasets, 
                                           metric_funcs=metrics, 
                                           wandb_instance=wandb_init)
    
    # Load and preprocess the data
    data = pd.read_csv('data/housing.csv', index_col=0)

    # data=data.drop('ZN',axis=1)
    # print(data.head(15))
    X = data.drop('TARGET_MEDV', axis=1).values
    y = data['TARGET_MEDV'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    datasets = [(X_train, X_test, y_train, y_test)]

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Custom Model
    custom_model = LinearRegressionNormalEquation()
    custom_model.fit(X_train, y_train)
    normal_prediction= custom_model.predict(X_test)
    print(normal_prediction)
    print(normal_prediction.size)

    custom_model = LinearRegressionSimple()
    custom_model.fit(X_train, y_train)
    simple_prediction= custom_model.predict(X_test)
    print(simple_prediction)
    print(simple_prediction.size)
    
    custom_model = LinearRegressionSquareTrick()
    custom_model.fit(X_train, y_train)
    square_prediction= custom_model.predict(X_test)
    print(square_prediction)
    print(square_prediction.size)

    custom_model = LinearRegressionAbsoluteTrick()
    custom_model.fit(X_train, y_train)
    absolute_prediction= custom_model.predict(X_test)
    print(absolute_prediction)
    print(absolute_prediction.size)

    # Scikit-learn Model
    sk_model = LinearRegression()
    sk_model.fit(X_train, y_train)
    sklearn_predictiond2=sk_model.predict(X_test)

    # Calculate evaluation metrics
    mae_simple = mean_absolute_error(y_test, sklearn_predictiond2)
    mae_absolute = mean_absolute_error(y_test, absolute_prediction)
    mae_square = mean_absolute_error(y_test, square_prediction)

    mse_simple = mean_squared_error(y_test, sklearn_predictiond2)
    mse_absolute = mean_squared_error(y_test, absolute_prediction)
    mse_square = mean_squared_error(y_test, square_prediction)

    r2_simple = r2_score(y_test, sklearn_predictiond2)
    r2_absolute = r2_score(y_test, absolute_prediction)
    r2_square = r2_score(y_test, square_prediction)

    plt.scatter(y_test, sklearn_predictiond2, label='Sklearn Regression housing')
    plt.scatter(y_test, absolute_prediction, label='Absolute Regression housing')
    plt.scatter(y_test, square_prediction, label='Square Regression housing')

    plt.xlabel('Actual Values housing')
    plt.ylabel('Predicted Values housing')
    plt.legend()
    plt.show()

    labels = ['MAE', 'MSE', 'R2']
    simple_metrics = [mae_simple, mse_simple, r2_simple]
    absolute_metrics = [mae_absolute, mse_absolute, r2_absolute]
    square_metrics = [mae_square, mse_square, r2_square]

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, simple_metrics, width, label='Sklearn')
    rects2 = ax.bar(x, absolute_metrics, width, label='Absolute')
    rects3 = ax.bar(x + width, square_metrics, width, label='Square')

    ax.set_ylabel('Scores')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()
    # Add models to a dictionary
    # models = {'Custom Model': custom_model, 'Sklearn Model': sk_model}

    # Evaluate and plot
   # evaluate_and_plot(models, X_test, y_test)

    # Benchmarking
    metrics = {'MSE': mean_squared_error}
    benchmark = BasicBenchmark()

    all_metrics = benchmark.evaluate_multi(models=[custom_model, sk_model], 
                                           datasets=datasets, 
                                           metric_funcs=metrics, 
                                           wandb_instance=wandb_init)
    
    # # Load and preprocess the data
    data = pd.read_csv('data/real_estate_data.csv', index_col=0)
    X = data.drop('Y house price of unit area', axis=1).values
    y = data['Y house price of unit area'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    datasets = [(X_train, X_test, y_train, y_test)]

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Custom Model
    custom_model = LinearRegressionNormalEquation()
    custom_model.fit(X_train, y_train)
    predictiond3= custom_model.predict(X_test)
    print(predictiond3)
    print(predictiond3.size)

    custom_model = LinearRegressionSimple()
    custom_model.fit(X_train, y_train)
    simple_predictiond3= custom_model.predict(X_test)
    print(simple_predictiond3)
    print(simple_predictiond3.size)
    
    custom_model = LinearRegressionSquareTrick()
    custom_model.fit(X_train, y_train)
    square_predictiond3= custom_model.predict(X_test)
    print(square_predictiond3)
    print(square_predictiond3.size)

    custom_model = LinearRegressionAbsoluteTrick()
    custom_model.fit(X_train, y_train)
    absolute_predictiond3= custom_model.predict(X_test)
    print(absolute_predictiond3)
    print(absolute_predictiond3.size)

    sk_model = LinearRegression()
    sk_model.fit(X_train, y_train)
    sklearn_predictiond3=sk_model.predict(X_test)

    mae_simple = mean_absolute_error(y_test, sklearn_predictiond3)
    mae_absolute = mean_absolute_error(y_test, absolute_predictiond3)
    mae_square = mean_absolute_error(y_test, square_predictiond3)

    mse_simple = mean_squared_error(y_test, sklearn_predictiond3)
    mse_absolute = mean_squared_error(y_test, absolute_predictiond3)
    mse_square = mean_squared_error(y_test, square_predictiond3)

    r2_simple = r2_score(y_test, sklearn_predictiond3)
    r2_absolute = r2_score(y_test, absolute_predictiond3)
    r2_square = r2_score(y_test, square_predictiond3)

    plt.scatter(y_test, sklearn_predictiond3, label='Sklearn Regression REal Estate')
    plt.scatter(y_test, absolute_predictiond3, label='Absolute Regression REal Estate')
    plt.scatter(y_test, square_predictiond3, label='Square Regression REal Estate')

    plt.xlabel('Actual Values Real Estate')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.show()

    labels = ['MAE', 'MSE', 'R2']
    simple_metrics = [mae_simple, mse_simple, r2_simple]
    absolute_metrics = [mae_absolute, mse_absolute, r2_absolute]
    square_metrics = [mae_square, mse_square, r2_square]

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, simple_metrics, width, label='Sklearn')
    rects2 = ax.bar(x, absolute_metrics, width, label='Absolute')
    rects3 = ax.bar(x + width, square_metrics, width, label='Square')

    ax.set_ylabel('Scores')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()

    metrics = {'MSE': mean_squared_error}
    benchmark = BasicBenchmark()

    all_metrics = benchmark.evaluate_multi(models=[custom_model, sk_model], 
                                           datasets=datasets, 
                                           metric_funcs=metrics, 
                                           wandb_instance=wandb_init)

    data = pd.read_csv('data/winequality-red.csv',sep=';',quoting=1, index_col=0)
    X = data.drop('quality', axis=1).values
    y = data['quality'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    datasets = [(X_train, X_test, y_train, y_test)]

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Custom Model
    custom_model = LinearRegressionNormalEquation()
    custom_model.fit(X_train, y_train)
    predictiond4= custom_model.predict(X_test)
    print(predictiond4)
    print(predictiond4.size)

    custom_model = LinearRegressionSimple()
    custom_model.fit(X_train, y_train)
    simple_predictiond4= custom_model.predict(X_test)
    print(simple_predictiond4)
    print(simple_predictiond4.size)
    
    custom_model = LinearRegressionSquareTrick()
    custom_model.fit(X_train, y_train)
    square_predictiond4= custom_model.predict(X_test)
    print(square_predictiond4)
    print(square_predictiond4.size)

    custom_model = LinearRegressionAbsoluteTrick()
    custom_model.fit(X_train, y_train)
    absolute_predictiond4= custom_model.predict(X_test)
    print(absolute_predictiond4)
    print(absolute_predictiond4.size)

    sk_model = LinearRegression()
    sk_model.fit(X_train, y_train)
    sklearn_predictiond4=sk_model.predict(X_test)

    mae_simple = mean_absolute_error(y_test, sklearn_predictiond4)
    mae_absolute = mean_absolute_error(y_test, absolute_predictiond4)
    mae_square = mean_absolute_error(y_test, square_predictiond4)

    mse_simple = mean_squared_error(y_test, sklearn_predictiond4)
    mse_absolute = mean_squared_error(y_test, absolute_predictiond4)
    mse_square = mean_squared_error(y_test, square_predictiond4)

    r2_simple = r2_score(y_test, sklearn_predictiond4)
    r2_absolute = r2_score(y_test, absolute_predictiond4)
    r2_square = r2_score(y_test, square_predictiond4)

    plt.scatter(y_test, sklearn_predictiond4, label='Sklearn Regression Wine Quality red')
    plt.scatter(y_test, absolute_predictiond4, label='Absolute Regression Wine Quality red')
    plt.scatter(y_test, square_predictiond4, label='Square Regression Wine Quality red')

    plt.xlabel('Actual Values Wine Quality red')
    plt.ylabel('Predicted Values wine')
    plt.legend()
    plt.show()

    labels = ['MAE', 'MSE', 'R2']
    simple_metrics = [mae_simple, mse_simple, r2_simple]
    absolute_metrics = [mae_absolute, mse_absolute, r2_absolute]
    square_metrics = [mae_square, mse_square, r2_square]

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, simple_metrics, width, label='Sklearn')
    rects2 = ax.bar(x, absolute_metrics, width, label='Absolute')
    rects3 = ax.bar(x + width, square_metrics, width, label='Square')

    ax.set_ylabel('Scores')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()

    metrics = {'MSE': mean_squared_error}
    benchmark = BasicBenchmark()

    all_metrics = benchmark.evaluate_multi(models=[custom_model, sk_model], 
                                           datasets=datasets, 
                                           metric_funcs=metrics, 
                                           wandb_instance=wandb_init)
    
    data = pd.read_csv('data/winequality-white.csv',sep=';',quoting=1, index_col=0)
    X = data.drop('quality', axis=1).values
    y = data['quality'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    datasets = [(X_train, X_test, y_train, y_test)]

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Custom Model
    custom_model = LinearRegressionNormalEquation()
    custom_model.fit(X_train, y_train)
    predictiond5= custom_model.predict(X_test)
    print(predictiond5)
    print(predictiond5.size)

    custom_model = LinearRegressionSimple()
    custom_model.fit(X_train, y_train)
    simple_predictiond5= custom_model.predict(X_test)
    print(simple_predictiond5)
    print(simple_predictiond5.size)
    
    custom_model = LinearRegressionSquareTrick()
    custom_model.fit(X_train, y_train)
    square_predictiond5= custom_model.predict(X_test)
    print(square_predictiond5)
    print(square_predictiond5.size)

    custom_model = LinearRegressionAbsoluteTrick()
    custom_model.fit(X_train, y_train)
    absolute_predictiond5= custom_model.predict(X_test)
    print(absolute_predictiond5)
    print(absolute_predictiond5.size)

    sk_model = LinearRegression()
    sk_model.fit(X_train, y_train)
    sklearn_predictiond5=sk_model.predict(X_test)

    mae_simple = mean_absolute_error(y_test, sklearn_predictiond5)
    mae_absolute = mean_absolute_error(y_test, absolute_predictiond5)
    mae_square = mean_absolute_error(y_test, square_predictiond5)

    mse_simple = mean_squared_error(y_test, sklearn_predictiond5)
    mse_absolute = mean_squared_error(y_test, absolute_predictiond5)
    mse_square = mean_squared_error(y_test, square_predictiond5)

    r2_simple = r2_score(y_test, sklearn_predictiond5)
    r2_absolute = r2_score(y_test, absolute_predictiond5)
    r2_square = r2_score(y_test, square_predictiond5)

    plt.scatter(y_test, sklearn_predictiond5, label='Sklearn Regression Wine Quality white')
    plt.scatter(y_test, absolute_predictiond5, label='Absolute Regression Wine Quality white')
    plt.scatter(y_test, square_predictiond5, label='Square Regression Wine Quality white')

    plt.xlabel('Actual Values Wine Quality white')
    plt.ylabel('Predicted Values wine')
    plt.legend()
    plt.show()

    labels = ['MAE', 'MSE', 'R2']
    simple_metrics = [mae_simple, mse_simple, r2_simple]
    absolute_metrics = [mae_absolute, mse_absolute, r2_absolute]
    square_metrics = [mae_square, mse_square, r2_square]

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, simple_metrics, width, label='Sklearn')
    rects2 = ax.bar(x, absolute_metrics, width, label='Absolute')
    rects3 = ax.bar(x + width, square_metrics, width, label='Square')

    ax.set_ylabel('Scores')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()

    metrics = {'MSE': mean_squared_error}
    benchmark = BasicBenchmark()

    all_metrics = benchmark.evaluate_multi(models=[custom_model, sk_model], 
                                           datasets=datasets, 
                                           metric_funcs=metrics, 
                                           wandb_instance=wandb_init)
    # # Visualize the metrics
    # visualize_metrics(all_metrics, metrics.keys())


if __name__ == '__main__':
    main()
