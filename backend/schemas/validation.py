from pydantic import BaseModel

class ValidationRequest(BaseModel):
    project_id: int
    input_data: str
    generated_data: str
    ground_truth: str
