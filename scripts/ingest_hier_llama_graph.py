from app.services.graphingestion import ingest_pharma_data_hybrid_Graph


def main():
    result = ingest_pharma_data_hybrid_Graph("data/raw")
    print(result)


if __name__ == "__main__":
    main()