import numpy as np
from sklearn.metrics import mean_squared_error
from typing import Callable, Union, List, Tuple, Dict, Any
import wandb
from abc import ABC, abstractmethod

class AbstractBenchmark(ABC):

    @abstractmethod
    def evaluate_simple(self, 
                        model: 'Model', 
                        X_train: np.ndarray, 
                        X_test: np.ndarray, 
                        y_train: np.ndarray, 
                        y_test: np.ndarray, 
                        metric_func: Callable[[np.ndarray, np.ndarray], float],
                        wandb_instance: Union[Any, None] = None
                        ) -> float:
        pass
    
    @abstractmethod
    def evaluate_multi(self, models: List['Model'], 
                       datasets: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], 
                       metric_funcs: Dict[str, Callable[[np.ndarray, np.ndarray], float]], 
                       wandb_instance: Union[Any, None] = None, 
                       iterations: int = 1
                       ) -> Dict[str, Dict[str, float]]:
        pass


class BasicBenchmark(AbstractBenchmark):

    def evaluate_multi(self, models: List['Model'], 
                       datasets: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], 
                       metric_funcs: Dict[str, Callable[[np.ndarray, np.ndarray], float]], 
                       wandb_instance: Union[Any, None] = None, 
                       iterations: int = 1
                       ) -> Dict[str, Dict[str, float]]:
        
        all_metrics = {}
        
        for model in models:
            model_name = type(model).__name__
            print(f"Benchmarking {model_name}...")
            
            if wandb_instance:
                wandb_instance.log({'Model': model_name})

            for i, (X_train, X_test, y_train, y_test) in enumerate(datasets):
                dataset_name = f"Dataset_{i+1}"
                metrics = {}

                for iter in range(iterations):
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    for metric_name, metric_func in metric_funcs.items():
                        value = metric_func(y_test, y_pred)
                        
                        if wandb_instance:
                            wandb_instance.log({f'{dataset_name}_{model_name}_{metric_name}': value})
                        
                        if metric_name not in metrics:
                            metrics[metric_name] = []
                        metrics[metric_name].append(value)

                avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
                all_metrics[f"{dataset_name}_{model_name}"] = avg_metrics
                
        return all_metrics

    def evaluate_simple(self, 
                        model: 'Model', 
                        X_train: np.ndarray, 
                        X_test: np.ndarray, 
                        y_train: np.ndarray, 
                        y_test: np.ndarray, 
                        metric_func: Callable[[np.ndarray, np.ndarray], float],
                        wandb_instance: Union[Any, None] = None
                        ) -> float:
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = metric_func(y_test, y_pred)
        
        if wandb_instance:
            wandb_instance.log({"mse": mse})

        return mse
