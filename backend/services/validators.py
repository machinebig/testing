from services.metrics.bleu import BLEUMetric
from services.metrics.rouge import ROUGEMetric
# Import other metrics as needed
import pandas as pd

class ValidatorService:
    def __init__(self):
        self.metrics = {
            'bleu': BLEUMetric(),
            'rouge': ROUGEMetric(),
            # Add other metrics
        }

    def run(self, df):
        results = []
        for _, row in df.iterrows():
            for metric_name, metric in self.metrics.items():
                score = metric.calculate_score(row['Generated'], row['Ground Truth'])
                pass_fail, reason = metric.validate_threshold(score, 0.7)  # Example threshold
                results.append({
                    'Input': row['Input'],
                    'Generated': row['Generated'],
                    'Ground Truth': row['Ground Truth'],
                    'Metric': metric_name,
                    'Score': score,
                    'Pass/Fail': pass_fail,
                    'Reason': reason
                })
        return results
