# from sentence_transformers import CrossEncoder

# reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# def rerank(query: str, documents: list[dict], top_k: int = 3):


#     pairs = [(query, doc["content"]) for doc in documents]

#     scores = reranker.predict(pairs)

#     for i, doc in enumerate(documents):
#         doc["rerank_score"] = float(scores[i])

#     documents = sorted(documents, key=lambda x: x["rerank_score"], )

#     return documents[:top_k]


   
