import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline


class RAGQuerySystem:
    def __init__(self, embeddings_file="embeddings.pkl", index_file="faiss_index.bin", embedding_model_name="all-MiniLM-L6-v2"):
        self.embeddings_file = embeddings_file
        self.index_file = index_file
        self.embedding_model_name = embedding_model_name
        self.index = None
        self.texts = []
        self.metadata = []
        self.embedding_model = None
        self.qa_pipeline = None

        # למנוע טעינה של TensorFlow (אם יש בעיות תאימות)
        os.environ["USE_TF"] = "0"

    def load_system(self):
        print("Loading FAISS index and metadata...")
        self.index = faiss.read_index(self.index_file)
        with open(self.embeddings_file, "rb") as f:
            data = pickle.load(f)
        self.texts = data["texts"]
        self.metadata = data["metadata"]
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
        print("System loaded successfully.")

    def retrieve(self, query, top_k=3):
        if not self.embedding_model or not self.index:
            raise RuntimeError("System not loaded. Call load_system() first.")
        
        print(f"\n Retrieving top {top_k} chunks for query: '{query}'")
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i in indices[0]:
            results.append({
                "text": self.texts[i],
                "meta": self.metadata[i]
            })
        return results

    def generate_answer(self, query, context):
        if not self.qa_pipeline:
            raise RuntimeError("QA pipeline not initialized.")
        return self.qa_pipeline(question=query, context=context)

    def interactive_loop(self):
        self.load_system()
        while True:
            query = input("\n Ask something (or type 'exit'): ")
            if query.lower() == "exit":
                print("Goodbye!")
                break

            results = self.retrieve(query)

            print("\n Top relevant chunks:")
            for r in results:
                print(f"- [{r['meta']['document']}] {r['text'][:200]}...\n")

            combined_context = results[0]["text"]

            try:
                answer = self.generate_answer(query, combined_context)
                print("\n Answer:", answer["answer"])
            except Exception as e:
                print("\n Error generating answer:", str(e))


if __name__ == "__main__":
    system = RAGQuerySystem()
    system.interactive_loop()
