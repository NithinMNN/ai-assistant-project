import google.generativeai as genai
import os
from dotenv import load_dotenv
import chromadb

# --- Setup ---
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("Error: GOOGLE_API_KEY not found in .env file.")
    exit()

genai.configure(api_key=API_KEY)
embedding_model = "models/text-embedding-004"

# --- Connect to ChromaDB ---
base_dir = os.path.dirname(__file__)
db_path = os.path.join(base_dir, "my_chroma_db")
client = chromadb.PersistentClient(path=db_path)
collection = client.get_or_create_collection("random_facts")

# --- The Learning Process ---
knowledge_folder = os.path.join(base_dir, "memory")

print("Starting study session...")

for filename in os.listdir(knowledge_folder):
    # We only want to read text files
    if filename.endswith(".txt"):
        file_path = os.path.join(knowledge_folder, filename)

        print(f"Reading: {filename}...")

        with open(file_path, "r") as f:
            knowledge_text = f.read()

        # Create the embedding for the file content
        embedding = genai.embed_content(model=embedding_model, content=knowledge_text)["embedding"]

        # Use upsert to add or update the memory.
        # We'll use the filename as the unique ID for each memory.
        collection.upsert(ids=[filename], embeddings=[embedding], documents=[knowledge_text])
        print(f"  -> Memory for '{filename}' is stored.")

print("\nStudy session complete! I've learned everything in the knowledge folder.")
