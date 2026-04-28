from pathlib import Path
from typing import Any, Dict
import logging
import time

import chromadb
import torch
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker

from app.services.embedding_service import embedding_model, embed_documents
import uuid
from app.db.Chroma_clientV2 import get_collection,add_documents
from app.core.config import settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def ingest_with_semantic_chunking(
    input_dir: str = "data/raw",

) -> Dict[str, Any]:
    pdfLoader = DirectoryLoader(
       input_dir,
       glob="**/*.pdf",
       loader_cls=PyMuPDFLoader, ##loaderclas to us
       show_progress=False
        )
    embmodel =embedding_model()
    pdfloader = pdfLoader.load()
    text_splitter = SemanticChunker(embmodel)
    chunked_docs = text_splitter.split_documents(pdfloader)
    print(chunked_docs)
    texts = [doc.page_content for doc in chunked_docs]
    embeddings = embed_documents(texts)
    add_documents(chunked_docs,embeddings)
    # print(embeddings)
   