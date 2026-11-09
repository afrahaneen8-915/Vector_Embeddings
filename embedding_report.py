import chromadb
import pandas as pd

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("education_documents")

data = collection.get(include=["embeddings"])

report = []

for doc_id, embedding in zip(data["ids"], data["embeddings"]):
    report.append({
        "Document_ID": doc_id,
        "Embedding_Dimensions": len(embedding)
    })

df = pd.DataFrame(report)
df.to_csv("embedding_report.csv", index=False)

print("Embedding report generated successfully!")
print("Total documents:", len(report))
print("Embedding dimensions:", len(data["embeddings"][0]))