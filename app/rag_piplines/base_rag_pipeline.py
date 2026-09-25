from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRAGPipeline(ABC):
    """
    Abstract Base Class for all RAG pipelines.
    Defines the standard interface every pipeline must implement.
    """


    @abstractmethod
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context based on the user's query.
        """
        pass

    @abstractmethod
    def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Generate a response using the LLM and the retrieved context.
        """
        pass

    def run(self, query: str) -> str:
        """
        The main execution flow. 
        Note: This is NOT abstract, it provides a default implementation 
        that orchestrates retrieve() and generate().
        """
        context = self.retrieve(query)
        response = self.generate(query, context)
        return response
