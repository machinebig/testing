from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from db import get_db
from schemas.project import ProjectCreate, Project
from models.project import Project as ProjectModel

router = APIRouter()

@router.post("/")
def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    db_project = ProjectModel(name=project.name, description=project.description)
    db.add(db_project)
    db.commit()
    return Project(**db_project.__dict__)

@router.get("/{project_id}")
def get_project(project_id: int, db: Session = Depends(get_db)):
    project = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return Project(**project.__dict__)
