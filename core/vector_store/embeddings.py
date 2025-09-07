from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from typing import List

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text], show_progress_bar=False)[0]
        return embedding.tolist()