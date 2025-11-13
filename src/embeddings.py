from typing import List

from sentence_transformers import SentenceTransformer


class SBERTEmbeddings:
    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.m = SentenceTransformer(name)

    def embed_documents(self, texts: List[str]):
        return self.m.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str):
        return self.m.encode([text], normalize_embeddings=True)[0].tolist()

    def __call__(self, text: str):
        return self.embed_query(text)

