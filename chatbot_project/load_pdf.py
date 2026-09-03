from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("data/MohammedAshfaq_Resume.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))
print()
print(chunks[0].page_content)
for i, chunk in enumerate(chunks):
    print("=" * 60)
    print(f"Chunk {i+1}")
    print(chunk.page_content)