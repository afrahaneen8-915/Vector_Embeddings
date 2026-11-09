# Vector Embeddings – AI in Education

## Project Description
This project demonstrates how vector embeddings can be generated
from educational documents and stored using ChromaDB.

## Domain
AI in Education

## Technologies Used
- Python
- Sentence Transformers
- ChromaDB
- Pandas
- NumPy

## Embedding Model
all-MiniLM-L6-v2

## Dataset
10 text documents related to AI in Education.

## Features
- Document loading
- Text embedding generation
- Vector storage using ChromaDB
- Semantic search
- Top 3 relevant document retrieval
- Embedding report generation

## Output Files
- search_results.csv
- embedding_report.csv

## How to Run
1. Install dependencies using:
   pip install -r requirements.txt

2. Generate embeddings:
   python vector_store.py

3. Search documents:
   python search_documents.py

4. Generate report:
   python embedding_report.py