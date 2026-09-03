# Mini GPT - Local RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built using Python, LangChain, ChromaDB, HuggingFace Embeddings, and Google's Gemini API.

This project allows users to ask questions about PDF documents. The chatbot retrieves the most relevant sections from the document using vector search and generates answers using an LLM.

---

## Features

- Load PDF documents
- Split PDFs into text chunks
- Generate embeddings using HuggingFace
- Store embeddings in ChromaDB
- Semantic search over documents
- Answer questions using Gemini
- View retrieved chunks for debugging

---

## Tech Stack

- Python
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Google Gemini API
- PyPDF

---

## Project Structure

```
mini_gpt/
│
├── chatbot_project/
│   ├── app/
│   ├── data/
│   ├── models/
│   ├── vectorstore/
│   │
│   ├── load_pdf.py
│   ├── store_vectors.py
│   ├── retrieve.py
│   ├── chatbot.py
│   ├── requirements.txt
│   └── .env
│
├── README.md
└── .gitignore
```

---

## How it Works

1. Place a PDF inside the `data/` folder.
2. Run `store_vectors.py` to:
   - Load the PDF
   - Split it into chunks
   - Generate embeddings
   - Store vectors in ChromaDB
3. Run `chatbot.py`
4. Ask questions about the document.

---

## Installation

Clone the repository

```bash
git clone https://github.com/mohammedsanuashfaq-dev/mini_gpt.git
```

Go into the project

```bash
cd mini_gpt/chatbot_project
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```env
GOOGLE_API_KEY=YOUR_API_KEY
```

Generate the vector database

```bash
python store_vectors.py
```

Start chatting

```bash
python chatbot.py
```

---

## Current Limitations

- Works with one PDF at a time
- Console-based interface
- No conversation memory
- No source citations
- No web interface

---

## Future Improvements

- Multiple PDF support
- Chat history and memory
- Streamlit web interface
- Source citations
- Better retrieval with reranking
- Hybrid search (BM25 + Vector Search)
- Upload PDFs directly from the UI
- Docker deployment
- API support using FastAPI

---

## Learning Goals

This project is part of my journey into:

- Artificial Intelligence
- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- Vector Databases
- LangChain
- Prompt Engineering

---

## Author

**Mohammed Ashfaq**

Computer Vision & AI Enthusiast

GitHub: https://github.com/mohammedsanuashfaq-dev