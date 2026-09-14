import os

import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


# ==========================================
# Project paths
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")
ENV_FILE = os.path.join(BASE_DIR, ".env")


# ==========================================
# Load environment variables
# ==========================================

load_dotenv(ENV_FILE)


# ==========================================
# Page configuration
# ==========================================

st.set_page_config(
    page_title="Multi-PDF RAG Chatbot",
    page_icon="📚",
    layout="centered"
)


# ==========================================
# Title
# ==========================================

st.title("📚 Multi-PDF RAG Chatbot")

st.write(
    "Ask questions about the information contained in the documents."
)


# ==========================================
# Load embedding model
# ==========================================

@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embedding = load_embedding_model()


# ==========================================
# Load ChromaDB
# ==========================================

@st.cache_resource
def load_vector_database():

    return Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embedding
    )


db = load_vector_database()


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

@st.cache_resource
def load_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0
    )


llm = load_llm()


# ==========================================
# Question input
# ==========================================

question = st.text_input(
    "Ask a question:",
    placeholder="Example: What is Papa's Spectacles?"
)


# ==========================================
# Generate answer
# ==========================================

if question:

    with st.spinner("Searching documents..."):

        docs = retriever.invoke(question)

        context = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            f"{doc.page_content}"
            for doc in docs
        )

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

        response = llm.invoke(prompt)

        if isinstance(response.content, list):

            answer = ""

            for item in response.content:

                if isinstance(item, dict) and item.get("type") == "text":

                    answer += item.get("text", "")

        else:

            answer = response.content


    # ==========================================
    # Display answer
    # ==========================================

    st.subheader("🤖 Answer")

    st.write(answer)


    # ==========================================
    # Display sources
    # ==========================================

    st.subheader("📄 Sources")

    sources = set()

    for doc in docs:

        source = doc.metadata.get("source", "Unknown")

        sources.add(source)


    for source in sources:

        st.write(f"- {source}")