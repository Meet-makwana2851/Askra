import os
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Load the embedding model (runs locally, downloads once ~80MB)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Set up a persistent Chroma vector database (saved to disk)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="askra_docs")

def chunk_text(text, chunk_size=800, overlap=150):
    """Split text into overlapping chunks (in characters)."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
from pypdf import PdfReader

def extract_pdf_text(filepath):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def index_documents(folder="docs"):
    chunk_id = 0
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)

        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        elif filename.endswith(".pdf"):
            text = extract_pdf_text(filepath)
        else:
            continue

        chunks = chunk_text(text)
        for chunk in chunks:
            embedding = embedder.encode(chunk).tolist()
            collection.add(
                ids=[f"chunk_{chunk_id}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": filename}]
            )
            chunk_id += 1
        print(f"Indexed {filename} into {len(chunks)} chunks")

if __name__ == "__main__":
    index_documents()
    print("Indexing complete.")
    
    