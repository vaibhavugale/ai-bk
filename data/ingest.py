"""
PostgreSQL (pgvector) Ingestion Script for Workflow Documents
Indexes workflow documents with embeddings into PostgreSQL using pgvector
"""

import json
import os
import sys
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine, Field, select
from sqlalchemy import Column, String, JSON
from pgvector.sqlalchemy import Vector
from datetime import datetime

# Add the root directory to sys.path to import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.settings import settings

load_dotenv()

BATCH_SIZE = 20
RATE_LIMIT_DELAY = 0.5  # seconds between batches

from app.model.workflow import WorkflowDocument


def get_engine():
    """Create and return SQLAlchemy engine"""
    engine = create_engine(settings.sqlalchemy_database_uri)
    return engine


def setup_database(engine):
    """Ensure pgvector extension exists and create tables"""
    print("Setting up database and ensuring pgvector is installed...")
    with Session(engine) as session:
        session.exec(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        session.commit()
    
    SQLModel.metadata.create_all(engine)


def format_workflow_text(workflow: Dict[str, Any]) -> str:
    """Format workflow document as text for embedding"""
    text_parts = [
        f"Workflow: {workflow['title']}",
        f"Type: {workflow['type']}",
        f"Category: {workflow['category']}",
        f"Description: {workflow['description']}",
        f"Estimated Duration: {workflow['estimated_duration']}",
        "\nSteps:"
    ]
    
    for step in workflow['steps']:
        step_text = f"\nStep {step['step_number']}: {step['description']}"
        if step.get('required_inputs'):
            step_text += f"\n  Required inputs: {', '.join(step['required_inputs'])}"
        if step.get('dependencies'):
            step_text += f"\n  Dependencies: Steps {', '.join(map(str, step['dependencies']))}"
        if step.get('completion_criteria'):
            step_text += f"\n  Completion criteria: {step['completion_criteria']}"
        text_parts.append(step_text)
    
    return "\n".join(text_parts)


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a batch of texts using a local Hugging Face model via LangChain"""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            # Optionally configure model_kwargs or encode_kwargs here
        )
        return embeddings.embed_documents(texts)
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        raise


def prepare_document(workflow: Dict[str, Any], embedding: List[float]) -> WorkflowDocument:
    """Prepare SQLModel document for database insertion"""
    return WorkflowDocument(
        id=workflow["id"],
        document=format_workflow_text(workflow),
        embedding=embedding,
        type=workflow.get("type"),
        title=workflow.get("title"),
        category=workflow.get("category"),
        estimated_duration=workflow.get("estimated_duration"),
        required_permissions=workflow.get("required_permissions", []),
        tags=workflow.get("tags", []),
        created_date=workflow.get("created_date"),
        last_updated=workflow.get("last_updated")
    )


def ingest_workflows(engine, workflows: List[Dict[str, Any]]):
    """Ingest workflows into PostgreSQL with embeddings"""
    total = len(workflows)
    print(f"\nIngesting {total} workflows into PostgreSQL database...")
    
    # Process in batches
    for i in range(0, total, BATCH_SIZE):
        batch = workflows[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
        
        # Generate text representations
        texts = [format_workflow_text(wf) for wf in batch]
        
        # Generate embeddings
        try:
            embeddings = get_embeddings(texts)
        except Exception as e:
            print(f"Failed to generate embeddings for batch {batch_num}: {e}")
            continue
        
        if not embeddings or len(embeddings) != len(batch):
            print(f"Failed to get embeddings for batch {batch_num}: Got {len(embeddings)} but expected {len(batch)}")
            continue
            
        # Insert into Database
        try:
            with Session(engine) as session:
                for wf, emb in zip(batch, embeddings):
                    # Check if document already exists to avoid duplication
                    existing = session.exec(select(WorkflowDocument).where(WorkflowDocument.id == wf["id"])).first()
                    if existing:
                        # Update existing
                        doc = prepare_document(wf, emb)
                        for key, value in doc.model_dump().items():
                            setattr(existing, key, value)
                    else:
                        # Add new
                        doc = prepare_document(wf, emb)
                        session.add(doc)
                session.commit()
            print(f"  ✓ Batch {batch_num} indexed successfully")
        except Exception as e:
            print(f"  ✗ Failed to index batch {batch_num}: {e}")
        
        # Rate limiting
        if i + BATCH_SIZE < total:
            time.sleep(RATE_LIMIT_DELAY)
    
    with Session(engine) as session:
        count = session.exec(select(WorkflowDocument)).all()
        print(f"\n✓ Ingestion complete! {len(count)} documents in table")


def main():
    """Main ingestion workflow"""
    print("=" * 60)
    print("Workflow Document Ingestion (PostgreSQL pgvector)")
    print("=" * 60)
    
    # Load workflow data
    data_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflow_dataset.json")
    if not os.path.exists(data_file_path):
        print(f"Error: Data file not found: {data_file_path}")
        print("Please run data_generator.py first to generate workflow data")
        return
    
    print(f"\nLoading workflows from {data_file_path}...")
    with open(data_file_path, 'r') as f:
        workflows = json.load(f)
    print(f"Loaded {len(workflows)} workflows")
    
    print("\nConnecting to PostgreSQL database...")
    try:
        engine = get_engine()
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("Connected to PostgreSQL successfully")
    except Exception as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        return
    
    setup_database(engine)
    ingest_workflows(engine, workflows)
    
    print("\n" + "=" * 60)
    print("Ingestion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

# Made with Bob
