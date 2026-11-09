import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("education_documents")

# Read queries
queries_df = pd.read_csv("dataset/queries.csv")

results = []

# Search documents
for query in queries_df["Query"]:
    query_embedding = model.encode([query]).tolist()

    search_result = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    for i in range(len(search_result["ids"][0])):
        results.append({
            "Query": query,
            "Rank": i + 1,
            "Document_ID": search_result["ids"][0][i],
            "Retrieved_Document": search_result["documents"][0][i],
            "Distance": search_result["distances"][0][i]
        })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("search_results.csv", index=False)

print("Document search completed successfully!")
print("Total queries:", len(queries_df))
print("Total results:", len(results_df))
print("Results saved to search_results.csv")