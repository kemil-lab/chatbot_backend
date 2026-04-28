from pathlib import Path
from typing import List, Dict, Any

from app.db.Chroma_clientV2 import get_collection
from app.rag.chunking import chunk_text,paragraph_chunk_text
from app.services.embedding_service import embed_documents
from app.services.pdf_service import extract_text_from_pdf


def ingest_pdf(pdf_path: str) -> Dict[str, Any]:
    
    """
    Full ingestion flow:
    PDF -> page text -> chunks -> embeddings -> Chroma
    """
    collection = get_collection()
    pages = extract_text_from_pdf(pdf_path)
    source_name = Path(pdf_path).name
    print(pages)
    all_ids: List[str] = []
    all_docs: List[str] = []
    all_metadatas: List[dict] = []

    for page_data in pages:
        page_number = page_data["page"]
        page_text = page_data["text"]

        # chunks = chunk_text(page_text, chunk_size=300, overlap=50)
        chunks = paragraph_chunk_text(page_text, max_words=250, overlap_words=40)
        print(chunks)
        for chunk_index, chunk in enumerate(chunks):
            chunk_id = f"{source_name}_p{page_number}_c{chunk_index}"
            metadata = {
                "source": source_name,
                "page": page_number,
                "chunk_index": chunk_index,
            }

            all_ids.append(chunk_id)
            all_docs.append(chunk)
            all_metadatas.append(metadata)

    if not all_docs:
        return {
            "source": source_name,
            "chunks_added": 0,
            "message": "No text found in PDF.",
        }

    embeddings = embed_documents(all_docs)

    collection.add(
        ids=all_ids,
        documents=all_docs,
        embeddings=embeddings,
        metadatas=all_metadatas,
    )

    return {
        "source": source_name,
        "chunks_added": len(all_docs),
        "message": "PDF ingested successfully.",
    }