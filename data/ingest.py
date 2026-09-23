"""
OpenSearch Ingestion Script for Workflow Documents
Indexes workflow documents with embeddings into OpenSearch
"""

import json
import os
import sys
import time
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv
from opensearchpy import OpenSearch, helpers

# Add the root directory to sys.path to import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.settings import settings

load_dotenv()

# Configuration
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "workflow_documents")
LITNG_EMBEDDING_API = "https://8001-01krc32prg8r3e6sd3v76vscg9.cloudspaces.litng.ai/v1"

BATCH_SIZE = 20
RATE_LIMIT_DELAY = 0.5  # seconds between batches


def get_opensearch_client() -> OpenSearch:
    """Create and return OpenSearch client"""
    client = OpenSearch(
        hosts=[{'host': settings.OPENSEARCH_HOST, 'port': settings.OPENSEARCH_PORT}],
        http_compress=True, # enables gzip compression for request bodies
        use_ssl=True,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False
    )
    return client


def get_or_create_index(client: OpenSearch, index_name: str):
    """Create OpenSearch index if it doesn't exist"""
    print(f"Ensuring index exists: {index_name}")
    if not client.indices.exists(index=index_name):
        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100
                }
            },
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 384, # BAAI/bge-small-en-v1.5 produces 384d vectors
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "lucene"
                        }
                    },
                    "document": {"type": "text"},
                    "type": {"type": "keyword"},
                    "title": {"type": "text"},
                    "category": {"type": "keyword"},
                    "estimated_duration": {"type": "text"},
                    "required_permissions": {"type": "keyword"},
                    "tags": {"type": "keyword"},
                    "created_date": {"type": "date"},
                    "last_updated": {"type": "date"}
                }
            }
        }
        client.indices.create(index=index_name, body=index_body)
    return index_name


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
    """Generate embeddings for a batch of texts using Litng API via LangChain"""
    try:
        from langchain_openai import OpenAIEmbeddings
        from pydantic import SecretStr
        embeddings = OpenAIEmbeddings(
            model="BAAI/bge-small-en-v1.5",
            api_key=SecretStr("my-super-secret-token"),
            base_url=LITNG_EMBEDDING_API,
            check_embedding_ctx_length=False,
        )
        return embeddings.embed_documents(texts)
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        raise


def prepare_document(workflow: Dict[str, Any], embedding: List[float], index_name: str) -> Dict[str, Any]:
    """Prepare document for OpenSearch bulk ingestion"""
    return {
        "_index": index_name,
        "_id": workflow["id"],
        "_source": {
            "document": format_workflow_text(workflow),
            "embedding": embedding,
            "type": workflow.get("type"),
            "title": workflow.get("title"),
            "category": workflow.get("category"),
            "estimated_duration": workflow.get("estimated_duration"),
            "required_permissions": workflow.get("required_permissions", []),
            "tags": workflow.get("tags", []),
            "created_date": workflow.get("created_date"),
            "last_updated": workflow.get("last_updated")
        }
    }


def ingest_workflows(
    client: OpenSearch,
    index_name: str,
    workflows: List[Dict[str, Any]]
):
    """Ingest workflows into OpenSearch with embeddings"""
    total = len(workflows)
    print(f"\nIngesting {total} workflows into OpenSearch...")
    
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
            
        # Prepare documents for bulk index
        actions = [
            prepare_document(wf, emb, index_name)
            for wf, emb in zip(batch, embeddings)
        ]
        
        # Insert into OpenSearch
        try:
            success, failed = helpers.bulk(client, actions)
            print(f"  ✓ Batch {batch_num} indexed successfully ({success} items)")
            if failed:
                print(f"  ⚠ Batch {batch_num} had {len(failed)} failed items")
        except Exception as e:
            print(f"  ✗ Failed to index batch {batch_num}: {e}")
        
        # Rate limiting
        if i + BATCH_SIZE < total:
            time.sleep(RATE_LIMIT_DELAY)
    
    # Get document count
    client.indices.refresh(index=index_name)
    count = client.count(index=index_name)['count']
    print(f"\n✓ Ingestion complete! {count} documents in index")


def main():
    """Main ingestion workflow"""
    print("=" * 60)
    print("Workflow Document Ingestion (OpenSearch)")
    print("=" * 60)
    
    # Load workflow data
    data_file = "../data/workflow_dataset.json"
    data_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflow_dataset.json")
    if not os.path.exists(data_file_path):
        print(f"Error: Data file not found: {data_file_path}")
        print("Please run data_generator.py first to generate workflow data")
        return
    
    print(f"\nLoading workflows from {data_file_path}...")
    with open(data_file_path, 'r') as f:
        workflows = json.load(f)
    print(f"Loaded {len(workflows)} workflows")
    
    # Initialize OpenSearch client
    print("\nConnecting to OpenSearch...")
    client = get_opensearch_client()
    
    # Test connection
    try:
        info = client.info()
        print(f"Connected to OpenSearch cluster: {info['cluster_name']}")
    except Exception as e:
        print(f"Error connecting to OpenSearch: {e}")
        return
    
    # Create index
    get_or_create_index(client, OPENSEARCH_INDEX)
    
    # Ingest workflows
    ingest_workflows(client, OPENSEARCH_INDEX, workflows)
    
    print("\n" + "=" * 60)
    print("Ingestion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

# Made with Bob
