from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from core.security import create_access_token, verify_password
from schemas.user import UserCreate, UserLogin
from models.user import User
from sqlalchemy.orm import Session
from db import get_db

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

@router.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = User(username=user.username, hashed_password=create_access_token(user.password))
    db.add(db_user)
    db.commit()
    return {"message": "User registered"}

@router.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}
