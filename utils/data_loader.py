import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, file_path: str):
        self.data = pd.read_csv(file_path, index_col=0)

    def train_test_split(self, test_size: float = 0.2) -> tuple:
        X = self.data.drop('target', axis=1).values
        y = self.data['target'].values
        return train_test_split(X, y, test_size=test_size, random_state=42)
