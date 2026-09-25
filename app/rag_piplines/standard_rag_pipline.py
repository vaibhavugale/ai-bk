from sqlmodel import select
from app.model.workflow import WorkflowDocument
from app.core.db import get_pgvector_session
from app.core.ai_models import ModelFactory
from app.core.db import OpenSearchDep
from typing import Any, Dict, List

from app.rag_piplines.base_rag_pipeline import BaseRAGPipeline
class StandardRAG(BaseRAGPipeline):
    # take the query
    
    # Create Embeddings
    # Retrieve document from open search with that embedding
    # return the output
    @classmethod
    def retrieve(cls, query: str) -> List[Dict[str, Any]]:
        # 1. Create embedding for the query
        embedding_client = ModelFactory.get_local_embedding_client()
        
        if embedding_client is None:
            print("❌ Embedding client is offline. Returning empty context.")
            return []

        try:
            query_embeddings = embedding_client.embed_query(query)
            print("✅ Successfully generated embeddings!")
            print(query_embeddings)
        except Exception as e:
            print(f"❌ Failed to connect to embedding model: {e}")
            return []
        # 2. Get relevant documents from pgvector
        try:
            
            # Use the session generator defined in db.py
            session = next(get_pgvector_session())
            
            # Query for top 5 most similar documents using L2 distance
            stmt = select(WorkflowDocument).order_by(
                WorkflowDocument.embedding.l2_distance(query_embeddings)
            ).limit(5)
            
            results = session.exec(stmt).all()
            print(results)
            
            # Format results to match expected output
            return [{"document": doc.document, "title": doc.title} for doc in results]
                
        except Exception as e:
            print(f"❌ Database query failed: {e}")
            return []
    def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
        return super().generate(query, context)
    pass

pipeline = StandardRAG() 