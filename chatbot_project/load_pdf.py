import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


documents = []

data_folder = "data"

for filename in os.listdir(data_folder):

    if filename.endswith(".pdf"):

        pdf_path = os.path.join(data_folder, filename)

        print("Loading:", filename)

        loader = PyPDFLoader(pdf_path)

        documents.extend(loader.load())


print("\nTotal pages loaded:", len(documents))


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)


print("Number of chunks:", len(chunks))
print()


for i, chunk in enumerate(chunks):

    print("=" * 60)
    print(f"Chunk {i + 1}")

    print("Source:", chunk.metadata.get("source"))

    print(chunk.page_content)