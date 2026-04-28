from sentence_transformers import SentenceTransformer
from typing import List

# Load model once
model = SentenceTransformer('BAAI/bge-large-en-v1.5')

def embed_query(text: str) -> List[float]:
    """
    Converts a single query string into an embedding vector.
    """
    vector = model.encode(text, normalize_embeddings=False)
    return vector

# Example usage
print(embed_query("Hello world"))