from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import SecretStr
from app.core.settings import settings



class ModelFactory:
      # 1. You MUST declare this here first!
    _embeddings_client: OpenAIEmbeddings | None = None
    _local_embeddings_client: HuggingFaceEmbeddings | None = None
    
    @classmethod
    def get_embedding_client(cls) -> OpenAIEmbeddings | None:
        if cls._embeddings_client is None:
            try:
                cls._embeddings_client = OpenAIEmbeddings(
                    model="BAAI/bge-small-en-v1.5",
                    # Provide a dummy string if the API key is empty to prevent an immediate crash
                    api_key=SecretStr(settings.LITNG_EMBEDDING_API_KEY or "missing-key"),
                    base_url=settings.LITNG_EMBEDDING_API,
                    check_embedding_ctx_length=False,
                )
            except Exception as e:
                print(f"⚠️ Warning: Could not initialize Embedding Client: {e}")
                return None
        return cls._embeddings_client

    @classmethod
    def get_local_embedding_client(cls) -> HuggingFaceEmbeddings | None:
        if cls._local_embeddings_client is None:
            try:
                print("⏳ Loading local HuggingFace embeddings (~133MB)...")
                cls._local_embeddings_client = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-small-en-v1.5"
                )
            except Exception as e:
                print(f"⚠️ Warning: Could not initialize Local Embedding Client: {e}")
                return None
        return cls._local_embeddings_client
