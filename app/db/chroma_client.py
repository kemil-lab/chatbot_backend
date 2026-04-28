from functools import lru_cache
from typing import Optional, List, Any
import uuid
import numpy as np

import chromadb
from chromadb.api.models.Collection import Collection

from app.core.config import settings


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.PersistentClient:
    """
    Returns a single persistent Chroma client for the app.
    """
    return chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)


@lru_cache(maxsize=1)
def get_collection() -> Collection:
    """
    Returns the main Chroma collection.
    Creates it if it does not exist.
    """
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        metadata={"description": "Main knowledge base collection for chatbot"},
    )


def reset_collection() -> None:
    """
    Deletes and recreates the collection.
    Useful only in development/testing.
    """
    client = get_chroma_client()

    try:
        client.delete_collection(name=settings.CHROMA_COLLECTION_NAME)
    except Exception:
        pass

    get_collection.cache_clear()
    client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        metadata={"description": "Main knowledge base collection for chatbot"},
    )


def add_documents(documents: List[Any], embeddings: np.ndarray):
    """Add documents and their embeddings to the vector store"""

    if len(documents) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(documents)} documents vs {len(embeddings)} embeddings"
        )

    print(f"Adding {len(documents)} documents to vector store")

    collection = get_collection()  

    ids = []
    metadatas = []
    documents_text = []
    embeddings_list = []

    for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
        doc_id = f"doc_{uuid.uuid4()}"
        ids.append(doc_id)

        metadata = dict(doc.metadata) if getattr(doc, "metadata", None) else {}
        metadata["doc_index"] = i
        metadata["content_length"] = len(doc.page_content)
        metadatas.append(metadata)

        documents_text.append(doc.page_content)
        embeddings_list.append(embedding)

    print("Final lengths:")
    print("ids:", len(ids))
    print("embeddings:", len(embeddings_list))
    print("documents:", len(documents_text))
    print("metadatas:", len(metadatas))

    try:
        collection.add(  
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=documents_text,
        )
        print("✅ Successfully added documents")
    except Exception as e:
        print(f"❌ Error adding documents: {e}")
        raise