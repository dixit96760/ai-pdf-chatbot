# AI PDF Question Answering System

A Retrieval-Augmented Generation (RAG) application built using Python, Gemini, and ChromaDB.

## Features

* Extracts text from PDF documents
* Splits content into chunks
* Creates vector embeddings using Gemini
* Stores embeddings in ChromaDB
* Performs semantic search
* Generates context-aware answers using Gemini 2.5 Flash

## Tech Stack

* Python
* Google Gemini API
* ChromaDB
* PyPDF
* python-dotenv

## How It Works

1. Read PDF document
2. Extract text
3. Generate embeddings
4. Store vectors in ChromaDB
5. Convert user question into embedding
6. Retrieve relevant chunks
7. Generate answer using Gemini

## Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

Run:

```bash
python main.py
```

## Future Improvements

* Streamlit Web Interface
* Multi-PDF Support
* Chat History
* Source Citations
* Better Retrieval Strategy
