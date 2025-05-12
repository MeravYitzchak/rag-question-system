import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingBuilder:
    def __init__(self, input_file, output_file, model_name="all-MiniLM-L6-v2"):
        self.input_file = input_file
        self.output_file = output_file
        self.model_name = model_name
        self.model = None
        self.chunks = []
        self.texts = []
        self.embeddings = None

    def load_chunks(self):
        with open(self.input_file, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        print(f"Loaded {len(self.chunks)} chunks from {self.input_file}")

    def build_model(self):
        print(f"Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

    def encode_chunks(self):
        self.texts = [chunk["text"] for chunk in self.chunks]
        print(f"Encoding {len(self.texts)} chunks...")
        self.embeddings = self.model.encode(
            self.texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

    def save(self):
        data = {
            "embeddings": self.embeddings,
            "texts": self.texts,
            "metadata": self.chunks
        }
        with open(self.output_file, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved embeddings to {self.output_file}")

    def run(self):
        self.load_chunks()
        self.build_model()
        self.encode_chunks()
        self.save()


if __name__ == "__main__":
    builder = EmbeddingBuilder(
        input_file="processed_documents.json",
        output_file="embeddings.pkl",
        model_name="all-MiniLM-L6-v2"
    )
    builder.run()
