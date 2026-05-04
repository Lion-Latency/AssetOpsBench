# Interchangeable Model Interface
# Reference: https://www.tutorialspoint.com/python/python_interfaces.htm
from abc import ABC, abstractmethod

class InterchangeableModelInterface(ABC):
    
    # Initialize the model adapter.
    def __init__(self, model_checkpoint: str, context_length: int, prediction_filter_length: int):
        self.model_checkpoint = model_checkpoint
        self.context_length = context_length
        self.prediction_filter_length = prediction_filter_length
        self.model = None

    # Load the model from checkpoint.
    @abstractmethod
    def load_model(self):
        pass

    # Run zero-shot forecasting on time series data.
    @abstractmethod
    def forecast(self):
        pass
    
    # Fine-tune the model on a dataset.
    @abstractmethod
    def finetune(self):
        pass

    # Run anomaly detection.
    @abstractmethod
    def anomaly_detection(self):
        pass

    # Run integrated anomaly detection (forecasting + anomaly detection in one call).
    @abstractmethod
    def integrated_anomaly_detection(self):
        pass