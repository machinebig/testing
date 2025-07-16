import pytest
from services.metrics.bleu import BLEUMetric

def test_bleu_metric():
    metric = BLEUMetric()
    score = metric.calculate_score("hello world", "hello world")
    assert score == 1.0
    assert metric.validate_threshold(score, 0.7)[0]
