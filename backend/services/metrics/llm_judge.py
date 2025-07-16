from .base_metric import BaseMetric

class LLMJudgeMetric(BaseMetric):
    def calculate_score(self, generated, ground_truth, **kwargs):
        # Placeholder for LLM-based qualitative evaluation
        return 0.0

    def validate_threshold(self, score, threshold):
        return score >= threshold, "Below LLM Judge threshold" if score < threshold else ""
