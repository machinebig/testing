from fastapi import FastAPI
from api.auth import router as auth_router
from api.projects import router as projects_router
from api.validation import router as validation_router

app = FastAPI(title="GenAI Validator API")

app.include_router(auth_router, prefix="/auth")
app.include_router(projects_router, prefix="/projects")
app.include_router(validation_router, prefix="/validation")

@app.get("/")
def read_root():
    return {"message": "Welcome to GenAI Validator API"}
