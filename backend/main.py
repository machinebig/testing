from fastapi import FastAPI
from backend.api import auth
app = FastAPI()
app.include_router(auth.router, prefix='/auth')
@app.get('/')
def root():
    return {'message': 'GenAI Validator API'}