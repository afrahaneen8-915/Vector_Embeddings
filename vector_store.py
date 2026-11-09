import os
from sentence_transformers import SentenceTransformer
import chromadb

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="education_documents"
)

# Read documents
documents = []
ids = []

folder = "dataset/documents"

for filename in os.listdir(folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(folder, filename)

        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read()

        documents.append(content)
        ids.append(filename.replace(".txt", ""))

# Generate embeddings
embeddings = model.encode(documents).tolist()

# Store documents and embeddings
collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)

print("Documents successfully converted into embeddings!")
print("Total documents:", len(documents))
print("Embedding dimensions:", len(embeddings[0]))