from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def score(self, X, y):
        pass
