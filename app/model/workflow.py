from sqlmodel import SQLModel, Field
from sqlalchemy import Column, String, JSON
from pgvector.sqlalchemy import Vector

class WorkflowDocument(SQLModel, table=True):
    __tablename__ = "workflow_documents"
    
    id: str = Field(primary_key=True)
    document: str
    embedding: list[float] = Field(sa_column=Column(Vector(384)))
    type: str | None = None
    title: str | None = None
    category: str | None = None
    estimated_duration: str | None = None
    required_permissions: list[str] = Field(default=[], sa_column=Column(JSON))
    tags: list[str] = Field(default=[], sa_column=Column(JSON))
    created_date: str | None = None
    last_updated: str | None = None
