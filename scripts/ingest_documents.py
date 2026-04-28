from pathlib import Path

from app.services.ingest_service import ingest_pdf


def main():
    raw_dir = Path("data/raw")
    pdf_files = list(raw_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in data/raw")
        return

    for pdf_file in pdf_files:
        result = ingest_pdf(str(pdf_file))
        print(result)


if __name__ == "__main__":
    main()
