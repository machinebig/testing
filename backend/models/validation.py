from sqlalchemy import Column, Integer, String, ForeignKey
from db import Base

class Validation(Base):
    __tablename__ = "validations"
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    input_data = Column(String)
    generated_data = Column(String)
    ground_truth = Column(String)
