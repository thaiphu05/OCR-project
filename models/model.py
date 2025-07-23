from abc import ABC, abstractmethod
class Model(ABC):
    @abstractmethod
    def predict(self, image):
        pass

    @abstractmethod
    def load_model(self, model_path):
        pass