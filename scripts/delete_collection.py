from app.db.chroma_client import reset_collection

def main():
    reset_collection()
    print("Collection deleted successfully.")

# if __name__ == "__main__":
#     main()