from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Same embedding model used while storing
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load existing vector database
db = Chroma(
    persist_directory="vectorstore",
    embedding_function=embedding
)

# Ask a question
query = "What skills does Ashfaq have?"

# Search for the most relevant chunks
results = db.similarity_search(query, k=3)

print(f"Found {len(results)} relevant chunks\n")

for i, doc in enumerate(results, start=1):
    print(f"----- Chunk {i} -----")
    print(doc.page_content)
    print("\n" + "=" * 60 + "\n")