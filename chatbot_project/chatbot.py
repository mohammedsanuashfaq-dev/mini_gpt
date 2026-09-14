
import os

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


# ==========================================
# Load embedding model
# ==========================================

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================
# Load vector database
# ==========================================

db = Chroma(
    persist_directory="vectorstore",
    embedding_function=embedding
)


# ==========================================
# Create retriever
# ==========================================

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 4
    }
)

# ==========================================
# Load Gemini
# ==========================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)


# ==========================================
# Chat loop
# ==========================================

while True:

    question = input("\nAsk a question (or type 'exit'): ")

    if question.lower() == "exit":
        break


    # ==========================================
    # Retrieve relevant chunks
    # ==========================================

    docs = retriever.invoke(question)

    print(f"\nRetrieved {len(docs)} chunks")

    print("\n------ Retrieved Chunks ------\n")


    for i, doc in enumerate(docs, 1):

        source = doc.metadata.get("source", "Unknown")

        print(f"Chunk {i}")
        print(f"Source: {source}")
        print("-" * 40)
        print(doc.page_content)
        print()


    # ==========================================
    # Create context
    # ==========================================

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"{doc.page_content}"
        for doc in docs
    )


    # ==========================================
    # Create prompt
    # ==========================================

    prompt = f"""
You are an assistant answering questions using ONLY the provided context.

Rules:
- Answer only using information present in the context.
- If the answer cannot be found in the context, say:
  "I couldn't find that information in the provided documents."
- Do not make up information.
- Give a clear and concise answer.
- Mention the source PDF when appropriate.

Context:
{context}

Question:
{question}

Answer:
"""


    # ==========================================
    # Generate answer
    # ==========================================

    response = llm.invoke(prompt)


    print("\n" + "=" * 60)

    print("Answer:\n")


    # ==========================================
    # Clean Gemini response
    # ==========================================

    if isinstance(response.content, list):

        answer = ""

        for item in response.content:

            if isinstance(item, dict) and item.get("type") == "text":

                answer += item.get("text", "")

    else:

        answer = response.content


    print(answer)

    print("=" * 60)