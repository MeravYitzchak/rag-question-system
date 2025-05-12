import faiss
import pickle
import numpy as np

class FaissIndexer:
    def __init__(self, embeddings_file, index_file):
        self.embeddings_file = embeddings_file
        self.index_file = index_file
        self.embeddings = None
        self.texts = []
        self.metadata = []
        self.index = None

    def load_embeddings(self):
        print(f"Loading embeddings from {self.embeddings_file}...")
        with open(self.embeddings_file, "rb") as f:
            data = pickle.load(f)
            self.embeddings = data["embeddings"]
            self.texts = data["texts"]
            self.metadata = data["metadata"]
        print(f"Loaded {len(self.embeddings)} embeddings.")

    def build_index(self):
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded.")
        dim = self.embeddings.shape[1]
        print(f"Building FAISS index (dimension: {dim})...")
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)
        print("Index built.")

    def save_index(self):
        if self.index is None:
            raise ValueError("Index not built.")
        faiss.write_index(self.index, self.index_file)
        print(f"Index saved to {self.index_file}")

    def run(self):
        self.load_embeddings()
        self.build_index()
        self.save_index()


if __name__ == "__main__":
    indexer = FaissIndexer(
        embeddings_file="embeddings.pkl",
        index_file="faiss_index.bin"
    )
    indexer.run()
