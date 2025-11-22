import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Step 1: The "Encode" Phase ---
# Load a pre-trained model capable of understanding text.
# 'all-MiniLM-L6-v2' is a great, lightweight model to start with.
model = SentenceTransformer('all-MiniLM-L6-v2')

# Our library of documents we want to search through.
documents = [
    "The crew is preparing the spacecraft for its mission to Mars.",
    "A chef is carefully slicing vegetables for a gourmet meal.",
    "The launch window for the rocket is closing soon.",
    "Ancient Roman recipes often included exotic spices.",
    "The astronaut is conducting experiments in zero gravity."
]

# Use the model to convert our documents into numerical embeddings.
doc_embeddings = model.encode(documents)

# The variable `doc_embeddings` now holds a list of vectors (numerical fingerprints).
# Each vector represents the semantic meaning of a document.

# --- Step 2: The "Index" Phase ---
# ChromaDB handles this internally when you add documents and their embeddings using HNSW or similar methods.

# --- Step 3: The "Search" Phase ---
# The user's query. Notice it doesn't share keywords like "rocket" or "spacecraft".
query = "Anything about spices?"

# Encode the query using the exact same model.
query_embedding = model.encode([query])[0]

# Now, we calculate the "closeness" between our query's vector and all
# the document vectors.
similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

# Find the index of the top 2 most similar documents.
top_k_indices = np.argsort(similarities)[-2:][::-1]

print(f"Query: '{query}'\n")
print("Top Search Results:")
for index in top_k_indices:
    print(f"- Document: '{documents[index]}' (Similarity Score: {similarities[index]:.4f})")