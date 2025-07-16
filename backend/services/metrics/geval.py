from .base_metric import BaseMetric

class GEvalMetric(BaseMetric):
    def calculate_score(self, generated, ground_truth, **kwargs):
        # Placeholder for GEval implementation
        return 0.0

    def validate_threshold(self, score, threshold):
        return score >= threshold, "Below GEval threshold" if score < threshold else ""
