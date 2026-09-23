from langchain_text_splitters import RecursiveJsonSplitter

class JSONSplitter:
    splitter = RecursiveJsonSplitter(max_chunk_size=300)

    pass