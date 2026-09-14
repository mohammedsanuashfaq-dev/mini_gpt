import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Multi-PDF RAG Chatbot",
    page_icon="📚",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("📚 Multi-PDF RAG Chatbot")

st.write(
    "Ask questions about the information contained in the documents."
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embedding = load_embedding_model()


# ============================================================
# LOAD CHROMA VECTOR DATABASE
# ============================================================

@st.cache_resource
def load_vector_database():

    return Chroma(
        persist_directory="vectorstore",
        embedding_function=embedding
    )


db = load_vector_database()


# ============================================================
# CREATE RETRIEVER
# ============================================================

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 4
    }
)


# ============================================================
# LOAD GEMINI
# ============================================================

@st.cache_resource
def load_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0
    )


llm = load_llm()


# ============================================================
# USER QUESTION
# ============================================================

question = st.text_input(
    "Ask a question:",
    placeholder="Example: What is Papa's Spectacles?"
)


# ============================================================
# RAG PIPELINE
# ============================================================

if question:

    with st.spinner("Searching documents..."):

        # ----------------------------------------------------
        # STEP 1: RETRIEVE RELEVANT DOCUMENT CHUNKS
        # ----------------------------------------------------

        docs = retriever.invoke(question)


        # ----------------------------------------------------
        # STEP 2: CREATE CONTEXT
        # ----------------------------------------------------

        context = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            f"{doc.page_content}"
            for doc in docs
        )


        # ----------------------------------------------------
        # STEP 3: CREATE PROMPT
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # STEP 4: ASK GEMINI
        # ----------------------------------------------------

        try:

            response = llm.invoke(prompt)

        except Exception as e:

            error_message = str(e)

            # Handle Gemini rate-limit / quota errors
            if "429" in error_message or "rate" in error_message.lower():

                st.error(
                    "⚠️ Gemini API daily limit reached. "
                    "Please try again after the quota resets."
                )

                st.stop()

            # Handle other API errors
            else:

                st.error(
                    "❌ Something went wrong while generating the answer."
                )

                st.stop()


        # ----------------------------------------------------
        # STEP 5: EXTRACT ANSWER
        # ----------------------------------------------------

        if isinstance(response.content, list):

            answer = ""

            for item in response.content:

                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                ):

                    answer += item.get("text", "")

        else:

            answer = response.content


    # ========================================================
    # DISPLAY ANSWER
    # ========================================================

    st.subheader("🤖 Answer")

    st.write(answer)


    # ========================================================
    # DISPLAY SOURCES
    # ========================================================

    st.subheader("📄 Sources")

    sources = set()

    for doc in docs:

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        sources.add(source)


    for source in sources:

        st.write(f"- {source}")