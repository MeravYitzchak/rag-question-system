import os
import json
import re
from typing import List


class DocumentPreprocessor:
    def __init__(self, docs_path="./documents", chunk_size=100, output_file="processed_documents.json"):
        self.docs_path = docs_path
        self.chunk_size = chunk_size
        self.output_file = output_file
        self.documents = []
        self.processed_chunks = []

    def load_documents(self):
        print(f"Loading documents from {self.docs_path}...")
        for filename in os.listdir(self.docs_path):
            if filename.endswith(".txt"):
                with open(os.path.join(self.docs_path, filename), "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    self.documents.append({
                        "title": filename.replace(".txt", ""),
                        "content": content
                    })
        print(f"Loaded {len(self.documents)} documents.")

    def clean_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()

    def split_into_chunks(self, text: str) -> List[str]:
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            words = paragraph.split()
            if len(current_chunk.split()) + len(words) <= self.chunk_size:
                current_chunk += "\n\n" + paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def preprocess(self):
        print("Preprocessing and chunking documents...")
        for doc in self.documents:
            cleaned = self.clean_text(doc["content"])
            chunks = self.split_into_chunks(cleaned)
            for i, chunk in enumerate(chunks):
                self.processed_chunks.append({
                    "document": doc["title"],
                    "chunk_id": f"{doc['title']}_chunk_{i+1}",
                    "text": chunk
                })
        print(f"Created {len(self.processed_chunks)} chunks.")

    def save_to_json(self):
        print(f"Saving processed chunks to {self.output_file}...")
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(self.processed_chunks, f, ensure_ascii=False, indent=2)
        print("Save complete.")

    def run(self):
        self.load_documents()
        self.preprocess()
        self.save_to_json()


if __name__ == "__main__":
    processor = DocumentPreprocessor()
    processor.run()
