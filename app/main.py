from app.rag_piplines.standard_rag_pipline import StandardRAG
from fastapi import FastAPI

retriver_result = StandardRAG.retrieve("What is machine learning?")
app = FastAPI()

# uv run fastapi dev app/main.py
# uv run fastapi dev
# 