import os
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from chromadb.errors import NotFoundError
import numpy as np

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("Error: GOOGLE_API_KEY not found in .env file.")
    exit()

genai.configure(api_key=API_KEY)

base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, "memory", "something.txt")

try:
    with open(file_path, "r") as f:
        knowledge_text = f.read()
except FileNotFoundError:
    print("Error: The 'memory/something.txt' file was not found.")
    exit()

embedding_model = "models/text-embedding-004"
embedding_resp = genai.embed_content(model=embedding_model, content=knowledge_text)
raw_vector = embedding_resp["embedding"]

vector = np.asarray(raw_vector, dtype=float)

base_dir = os.path.dirname(__file__)
db_path = os.path.join(base_dir, "my_chroma_db")
db_exists = os.path.exists(db_path)

client = chromadb.PersistentClient(path=db_path)

try:
    if db_exists:
        collection = client.get_collection("random_facts")
        print("Found existing memory bank collection.")
    else:
        collection = client.create_collection("random_facts")
        print("Created new memory bank collection.")
except NotFoundError:
    collection = client.create_collection("random_facts")
    print("Created new memory bank collection.")

try:
    collection.add(documents=[knowledge_text], embeddings=[vector], ids=["fact_1"])
    print("Success! Your first memory has been stored in the memory bank.")
except Exception as e:
    if "already exists" in str(e).lower():
        print("Memory with ID 'fact_1' already exists in the memory bank.")
        collection.update(ids=["fact_1"], documents=[knowledge_text], embeddings=[vector])
        print("Updated existing memory in the memory bank.")
    else:
        print(f"Error adding memory to the bank: {e}")
