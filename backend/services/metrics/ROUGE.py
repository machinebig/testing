from .base_metric import BaseMetric
from rouge_score import rouge_scorer

class ROUGEMetric(BaseMetric):
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    def calculate_score(self, generated, ground_truth, **kwargs):
        scores = self.scorer.score(ground_truth, generated)
        return scores['rouge1'].f_measure

    def validate_threshold(self, score, threshold):
        return score >= threshold, "Below ROUGE threshold" if score < threshold else ""
