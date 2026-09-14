import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# ==========================================
# Load ALL PDFs
# ==========================================

data_folder = "data"

documents = []

for filename in os.listdir(data_folder):

    if filename.endswith(".pdf"):

        pdf_path = os.path.join(data_folder, filename)

        print("Loading:", filename)

        loader = PyPDFLoader(pdf_path)

        documents.extend(loader.load())


print("\nTotal pages loaded:", len(documents))


# ==========================================
# Split into chunks
# ==========================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=150
)

chunks = splitter.split_documents(documents)


print("\nTotal chunks created:", len(chunks))


# ==========================================
# Load embedding model
# ==========================================

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================
# Store vectors in ChromaDB
# ==========================================

db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="vectorstore"
)


print("\nVector database created successfully!")
print(f"Stored {len(chunks)} chunks.")