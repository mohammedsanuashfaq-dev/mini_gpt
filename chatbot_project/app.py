import os
import re
import streamlit as st
from pathlib import Path
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types

# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="Multi-PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# =========================
# GEMINI
# =========================

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    try:
        API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        API_KEY = None

if API_KEY:
    client = genai.Client(api_key=API_KEY)
else:
    client = None


# =========================
# PAGE TITLE
# =========================

st.title("📚 Multi-PDF RAG Chatbot")
st.write("Ask questions about the information contained in the documents.")


# =========================
# PDF PROCESSING
# =========================

def extract_pdf_text(file_path):

    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        try:
            text = page.extract_text()
        except Exception:
            text = ""

        if text:
            text = text.strip()

            if text:
                pages.append({
                    "text": text,
                    "source": file_path.name,
                    "page": page_number
                })

    return pages


def split_text(text, chunk_size=1000, overlap=200):

    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= chunk_size:
        return [text]

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


def build_documents():

    documents = []

    pdf_files = list(DATA_DIR.glob("*.pdf"))

    for pdf_file in pdf_files:

        pages = extract_pdf_text(pdf_file)

        for page_data in pages:

            chunks = split_text(page_data["text"])

            for chunk in chunks:

                if len(chunk.strip()) > 20:

                    documents.append({
                        "text": chunk,
                        "source": page_data["source"],
                        "page": page_data["page"]
                    })

    return documents


# =========================
# LOAD DOCUMENTS
# =========================

@st.cache_data
def load_documents():

    return build_documents()


documents = load_documents()


# =========================
# PDF UPLOAD
# =========================

with st.sidebar:

    st.header("📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        changed = False

        for uploaded_file in uploaded_files:

            file_path = DATA_DIR / uploaded_file.name

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            changed = True

        if changed:
            load_documents.clear()
            st.rerun()

    st.divider()

    st.write(f"**PDF files:** {len(list(DATA_DIR.glob('*.pdf')))}")
    st.write(f"**Text chunks:** {len(documents)}")

    if st.button("🔄 Reload PDFs"):

        load_documents.clear()
        st.rerun()


# =========================
# CHECK DOCUMENTS
# =========================

if not documents:

    st.warning(
        "No readable PDF documents found. "
        "Upload one or more PDFs from the sidebar."
    )

    st.stop()


# =========================
# CREATE TF-IDF INDEX
# =========================

@st.cache_resource
def create_index(texts):

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(texts)

    return vectorizer, matrix


texts = [doc["text"] for doc in documents]

vectorizer, document_matrix = create_index(texts)


# =========================
# RETRIEVAL
# =========================

def retrieve_documents(question, top_k=5):

    question_vector = vectorizer.transform([question])

    scores = cosine_similarity(
        question_vector,
        document_matrix
    )[0]

    ranked_indexes = scores.argsort()[::-1]

    results = []

    for index in ranked_indexes[:top_k]:

        score = float(scores[index])

        if score > 0:

            doc = documents[index].copy()

            doc["score"] = score

            results.append(doc)

    return results


# =========================
# GEMINI ANSWER
# =========================

def generate_answer(question, retrieved_docs):

    if not retrieved_docs:

        return (
            "I couldn't find relevant information in the provided documents."
        )

    context_parts = []

    for i, doc in enumerate(retrieved_docs, start=1):

        context_parts.append(
            f"""
SOURCE {i}
File: {doc['source']}
Page: {doc['page']}

Content:
{doc['text']}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question ONLY using the information contained
in the provided document context.

If the answer is not present in the context, say:
"I couldn't find that information in the provided documents."

Do not make up information.

When possible, give a clear and concise answer.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    if client is None:

        return (
            "Gemini API key is not configured. "
            "Set GEMINI_API_KEY and try again."
        )

    try:

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:

        return f"Gemini error: {str(e)}"


# =========================
# QUESTION
# =========================

question = st.text_input(
    "Ask a question:",
    placeholder="What is Papa's Spectacles?"
)


# =========================
# ANSWER
# =========================

if question:

    with st.spinner("Searching the documents..."):

        retrieved_docs = retrieve_documents(
            question,
            top_k=5
        )

    st.subheader("🤖 Answer")

    if not retrieved_docs:

        st.error(
            "I couldn't find relevant information in the provided documents."
        )

    else:

        with st.spinner("Generating answer..."):

            answer = generate_answer(
                question,
                retrieved_docs
            )

        st.write(answer)

        # =========================
        # SOURCES
        # =========================

        st.subheader("📄 Sources")

        for i, doc in enumerate(retrieved_docs, start=1):

            with st.expander(
                f"{i}. {doc['source']} — Page {doc['page']}"
            ):

                st.write(
                    f"**Relevance score:** {doc['score']:.3f}"
                )

                st.write(doc["text"])


# =========================
# FOOTER
# =========================

st.divider()

st.caption(
    f"📚 {len(list(DATA_DIR.glob('*.pdf')))} PDF(s) • "
    f"🔎 {len(documents)} searchable chunks"
)