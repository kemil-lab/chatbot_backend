from typing import Any, Dict, List, Optional

from app.db.Chroma_clientV2 import get_collection
from app.services.embedding_service import embed_query


def retrieve_relevant_chunks(
    query: str,
    top_k: int = 3,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    collection = get_collection()
    query_embedding = embed_query(query)
   
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results= top_k,
        # query_texts=[query]
        # where_document={"$contains":query},
        include=["documents", "metadatas", "distances",],
    )
    print("restults")
    print(results)
    print("restults")
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved_chunks = []

    for idx, doc in enumerate(documents):
        retrieved_chunks.append(
            {
                "id": ids[idx] if idx < len(ids) else None,
                "content": doc,
                "metadata": metadatas[idx] if idx < len(metadatas) else {},
                "distance": distances[idx] if idx < len(distances) else None,
            }
        )

    return retrieved_chunks