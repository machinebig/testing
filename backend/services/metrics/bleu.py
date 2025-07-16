from .base_metric import BaseMetric
from nltk.translate.bleu_score import sentence_bleu
import nltk

nltk.download('punkt')

class BLEUMetric(BaseMetric):
    def calculate_score(self, generated, ground_truth, **kwargs):
        return sentence_bleu([ground_truth.split()], generated.split())

    def validate_threshold(self, score, threshold):
        return score >= threshold, "Below BLEU threshold" if score < threshold else ""
