# RAG QA System – Architecture Overview

This document provides an overview of the architecture for the Retrieval-Augmented Generation (RAG) system used to answer questions based on internal company documents

The code is implemented in an object-oriented (OOP) structure using clear classes for each step in the process: document processing, embedding creation, indexing with FAISS, and retrieval + response.

---

## System Components

### 1. **Document Preprocessing**

* **Input:** `.txt` files from the `./documents` folder
* **Process:**

  * Clean the text
  * Split into chunks (\~100 words)
  * Store each chunk with metadata (`document` name, `chunk_id`, `text`)

### 2. **Embedding & Indexing**

* **Embedding Model:** `all-MiniLM-L6-v2` from Sentence-Transformers
* **Steps:**

  * Generate vector embeddings for each chunk
  * Store the embeddings in a FAISS index (`faiss_index.bin`)
  * Save text + metadata to `embeddings.pkl`

### 3. **Query Retrieval**

* **Process:**

  * User types a question
  * The system embeds the query
  * FAISS retrieves the top relevant chunks
  * Relevant chunks are shown as a preview

### 4. **Answer Generation**

* **Model:** A QA model like `distilbert-base-uncased-distilled-squad` from Hugging Face
* **Process:**

  * Use retrieved chunks as context
  * Generate an answer to the question
  * Return the final answer to the user

---

## Workflow Summary

1. **Preprocessing**: Clean and split documents into chunks.
2. **Embedding**: Convert chunks into vectors using Sentence-Transformers.
3. **Indexing**: Store vectors in FAISS for fast retrieval.
4. **Querying**: Retrieve relevant chunks and pass to the LLM.
5. **Answer Generation**: Generate answers using a lightweight QA pipeline with Hugging Face's model.

---

## Directory Structure

```
rag-qa-system/
│
├── documents/                # Input .txt files
├── processed_documents.json  # Chunked and cleaned output
├── embeddings.pkl            # Saved embeddings and metadata
├── faiss_index.bin           # Vector store
├── preprocess.py             # Step 1: Clean and chunk docs
├── build_faiss_index.py      # Step 2: Generate embeddings and index
├── query_retriever.py        # Step 3: Retrieve and answer
├── README.md                 # Setup & usage instructions
└── ARCHITECTURE.md           # This file
```

---

## Technologies Used

* Python 3.11
* sentence-transformers
* faiss-cpu
* transformers (for QA model)
* scikit-learn, numpy, json, pickle

---

## Possible Improvements

* Add a web or chatbot UI
* Use a local LLM (like LLaMA or Mistral) for offline generation
* Integrate vector databases like Pinecone or Weaviate
* Handle multilingual documents

---

## RAG Methodology – Comparison and Justification

RAG systems combine two steps:

1. Retrieval of relevant document chunks
2. Generation of an answer based on them

### Comparison of RAG Methodologies:

* **RAG-Sequence**: Sends all retrieved chunks to the model. Simple to implement but limited by token size.
* **RAG-Token**: Weighs each token’s importance. More accurate but requires fine-tuning.
* **Fusion-in-Decoder**: Integrates chunks during generation. Powerful but complex and resource-heavy.
* **Multi-stage Retrieval**: Two-step retrieval for precision. More accurate but slower.

### My Choice: **RAG-Sequence**

I chose RAG-Sequence for its simplicity and compatibility with local models. It’s suitable for a prototype without requiring cloud infrastructure.

---

## Technology Choices – Summary

1. **Embedding Model**: `all-MiniLM-L6-v2` (via Sentence-Transformers)

   * **Reason**: Strong balance between speed, accuracy, and low resource usage. Ideal for semantic embeddings in document chunk retrieval.

2. **Vector Store**: FAISS

   * **Reason**: Efficient for similarity search over dense vectors. Supports scalable indexing and fast retrieval, essential for performance.

3. **Document Preprocessing & Chunking**

   * **Reason**: Splitting documents into small chunks allows for accurate retrieval and avoids input length limitations.

4. **No LangChain**

   * **Reason**: LangChain is powerful but adds unnecessary complexity for a simple prototype. A custom pipeline gives full control and keeps the setup lightweight.

---

## Data Flow Summary

1. **Preprocessing**: Clean and split documents into chunks.
2. **Embedding**: Convert chunks into vectors using Sentence-Transformers.
3. **Indexing**: Store vectors in FAISS for fast retrieval.
4. **Querying**: Retrieve relevant chunks and pass to the LLM.
5. **Answer Generation**: Use a lightweight QA pipeline with Hugging Face's `distilbert-base-uncased-distilled-squad`.

---

## Deployment Considerations

* **On-premise**: Recommended for sensitive internal data—more control and better compliance.
* **Cloud**: Suitable for scale and flexibility if data sensitivity is lower. Offers easier maintenance and scalability.



