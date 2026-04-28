from app.services.langchain_semantic_ingest_service import ingest_with_semantic_chunking


def main():
    result = ingest_with_semantic_chunking("data/raw", )
    print(result)


if __name__ == "__main__":
    main()