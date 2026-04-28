import chromadb

# create client
client = chromadb.Client()

# create collection
collection = client.get_or_create_collection(name="test")

# add data
collection.add(
    documents=["This is a test document","tnother"],
    ids=["1","2"]
)

# query
results = collection.query(
    query_texts=["another"],
    n_results=1
)

print(results)