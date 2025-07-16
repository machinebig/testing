from .base_metric import BaseMetric
from sentence_transformers import SentenceTransformer
import numpy as np

class CosineMetric(BaseMetric):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def calculate_score(self, generated, ground_truth, **kwargs):
        embeddings = self.model.encode([generated, ground_truth])
        return float(np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-10))

    def validate_threshold(self, score, threshold):
        return score >= threshold, "Below Cosine similarity threshold" if score < threshold else ""
