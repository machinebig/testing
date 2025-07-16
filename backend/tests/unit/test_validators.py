import pytest
from services.validators import ValidatorService

def test_validator_service():
    service = ValidatorService()
    df = pd.DataFrame([{
        'Input': 'test input',
        'Generated': 'hello world',
        'Ground Truth': 'hello world'
    }])
    results = service.run(df)
    assert len(results) > 0
    assert results[0]['Pass/Fail'] == True
