from typing import Any, Dict, List
import re
import torch
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    KnowledgeGraphIndex,
)
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# 🔥 NEW (Neo4j)
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.llms.openai import OpenAI

from app.core.config import settings as config
from app.db.Chroma_clientV2 import get_collection

from app.contexts.neo import graph_store
from app.contexts.mongo import docstore
# ---------------- CLEANING ---------------- #

def clean_metadata(raw_meta: dict) -> dict:
    source_path = raw_meta.get("source") or raw_meta.get("file_path") or ""
    file_name = Path(source_path).name if source_path else raw_meta.get("file_name", "")

    return {
        "file_name": file_name,
        "page": raw_meta.get("page", ""),
        "total_pages": raw_meta.get("total_pages", ""),
        "author": raw_meta.get("author", ""),
    }


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    text = re.sub(r"\b(obj|endobj|stream|endstream|xref)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if re.search(r"\.{2,}\s*\d+\s*$", line):
            continue

        if re.fullmatch(r"[\W_]+", line):
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()


# ---------------- INGEST ---------------- #

def ingest_pharma_data_hybrid_Graph(input_dir: str = "data/raw") -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=config.EMBEDDING_MODEL,
        device=device,
    )

    Settings.llm = OpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=config.OPENAI_API_KEY,
    )

    loader = DirectoryLoader(
        input_dir,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True,
    )

    raw_docs = loader.load()

    if not raw_docs:
        return {"status": "error", "message": "No PDF documents loaded"}

    documents: List[Document] = []

    for doc in raw_docs:
        cleaned = clean_text(doc.page_content)

        if not cleaned or len(cleaned.split()) < 50:
            continue

        if "obj" in cleaned or "stream" in cleaned:
            continue

        metadata = clean_metadata(doc.metadata or {})

        documents.append(
            Document(
                text=cleaned,
                metadata=metadata,
            )
        )

    if not documents:
        return {"status": "error", "message": "No clean text found"}


    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[1500, 800]  
    )

    all_nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(all_nodes)


    chroma_collection = get_collection()
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)


   
    docstore.add_documents(all_nodes)

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        graph_store=graph_store,  
        docstore=docstore,
    )



    vector_index = VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
    )



    graph_index = KnowledgeGraphIndex.from_documents(
        documents,   # 🔥 use full docs (important)
        max_triplets_per_chunk=25,  # 🔥 better graph
        storage_context=storage_context,
    )

    # ---------------- PERSIST ---------------- #

    storage_context.persist(persist_dir=config.CHROMA_PERSIST_DIR)

    return {
        "status": "success",
        "documents_loaded": len(documents),
        "all_nodes": len(all_nodes),
        "leaf_nodes": len(leaf_nodes),
        "graph": "neo4j_created",
    }