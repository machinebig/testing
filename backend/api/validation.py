from fastapi import APIRouter, Depends, UploadFile, File
from sqlalchemy.orm import Session
from db import get_db
from services.validators import ValidatorService
from schemas.validation import ValidationRequest
import pandas as pd

router = APIRouter()

@router.post("/")
async def run_validation(file: UploadFile = File(...), db: Session = Depends(get_db)):
    df = pd.read_csv(file.file) if file.filename.endswith('.csv') else pd.read_excel(file.file)
    validator = ValidatorService()
    results = validator.run(df)
    return results
