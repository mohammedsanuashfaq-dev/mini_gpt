from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Load PDF
loader = PyPDFLoader("data/MohammedAshfaq_Resume.pdf")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Number of chunks: {len(chunks)}")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert first chunk into an embedding
embedding = model.encode(chunks[0].page_content)

print("\nEmbedding dimension:", len(embedding))
print("\nFirst 10 values:")
print(embedding[:10])