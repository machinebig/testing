from abc import ABC, abstractmethod

class BaseMetric(ABC):
    @abstractmethod
    def calculate_score(self, generated, ground_truth, **kwargs):
        pass

    @abstractmethod
    def validate_threshold(self, score, threshold):
        pass
