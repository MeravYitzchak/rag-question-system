# RAG QA System
A simple Retrieval-Augmented Generation (RAG) system for document-based question answering.

## What It Does
You can ask natural language questions about a collection of text documents, and the system will find relevant parts and answer them using a QA model.

## Project Overview

This project enables:
- Preprocessing of plain text documents
- Chunking, embedding, and indexing with FAISS
- Semantic search of document chunks
- Question answering using a Hugging Face QA model

## Setup
- pip install sentence-transformers
- pip install faiss-cpu
- pip install llama-cpp-python
- pip install tf-keras
- pip install torch
- pip install transformers
- CMD -> huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.1-GGUF mistral-7b-instruct-v0.1.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False


rag-qa-system/
- │
- ├── documents/                # Input .txt files
- ├── processed_documents.json  # Chunked and cleaned output
- ├── embeddings.pkl            # Saved embeddings and metadata
- ├── faiss_index.bin           # Vector store
- ├── preprocessing.py          # Step 1: Clean and chunk docs
- ├── build_embeddings.py       # Step 2: Generate sentence embeddings
- ├── build_faiss_index.py      # Step 3: Build FAISS index from embeddings
- ├── query_retriever.py        # Step 4: Retrieve and answer questions
- ├── ARCHITECTURE.md           # System design overview
- └── README.md                 # This file

### 1. Clone the Repository
bash
git clone https://github.com/MeravYitzchak/rag-qa-system
cd rag-qa-system

### 2. python -m venv venv
source venv\Scripts\activate     # or venv/bin/activate on mac

### 3. Install dependencies
pip install -r requirements.txt


## ▶ How to Run the System

### 1: Preprocess the Documents
   Clean, normalize, and split your `.txt` files into small chunks for embedding:
 - python preprocessing.py
### 2: Generate Embeddings
   Use a SentenceTransformer to convert text chunks into vector representations:
 - python build_embeddings.py 
### 3: Build the FAISS Index
   Create a FAISS index from the generated embeddings for efficient similarity search:
 - python build_faiss_index.py
### 4: Start the Question Answering Interface
   Run the main loop to ask questions and get answers based on your documents:
 - python query_retriever.py 


## Example Questions
Here are some useful questions you can ask the system:

    1. What were the main topics discussed at the engineering synchronization meeting?
    2. What are the rules that apply to working from home?
    3. What are the specific requirements for creating a strong password according to the companys IT security policy?
    4. How many paid vacation days are employees entitled to each year?
    5. Is there a specific deadline for completing the mandatory legal forms?





