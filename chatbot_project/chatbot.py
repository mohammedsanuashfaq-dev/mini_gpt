import os

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Load embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector database
db = Chroma(
    persist_directory="vectorstore",
    embedding_function=embedding
)

# Create retriever
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 8
    }
)

# Load Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)

while True:

    question = input("\nAsk a question (or type 'exit'): ")

    if question.lower() == "exit":
        break

    docs = retriever.invoke(question)
    print(len(docs))
    print("\n------Retrieved Chunks------\n")

    for i, doc in enumerate(docs, 1):
        print(f"Chunk {i}")
        print("-" * 40)
        print(doc.page_content)
        print()

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are an assistant answering questions using ONLY the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    print("\n" + "=" * 60)
    print("Answer:\n")
    print(response.content)
    print("=" * 60)
