from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load PDF
loader = PyPDFLoader("data/eesa102.pdf")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=150
)

chunks = splitter.split_documents(documents)
print("\n========== ALL CHUNKS ==========\n")

for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}")
    print("-" * 50)
    print(chunk.page_content)
    print()
# Load embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Store vectors
db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="vectorstore"
)

print("Vector database created successfully!")
print(f"Stored {len(chunks)} chunks.")