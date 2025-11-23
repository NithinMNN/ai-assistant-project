# exp/test4_query.py
import os
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise SystemExit("ERROR: GOOGLE_API_KEY missing in .env")

genai.configure(api_key=API_KEY)

base_dir = os.path.dirname(__file__)
db_path = os.path.join(base_dir, "my_chroma_db")
client = chromadb.PersistentClient(path=db_path)
collection_name = "random_facts"

try:
    collection = client.get_collection(collection_name)
except ValueError:
    raise SystemExit("ERROR: collection 'random_facts' not found. Run add script first.")

question = "What is the day today?"
embedding_model = "models/text-embedding-004"
resp = genai.embed_content(model=embedding_model, content=question)
query_vec = np.array(resp["embedding"], dtype=float)

result = collection.query(query_embeddings=[query_vec], n_results=1)

docs = result.get("documents")
if docs and len(docs) > 0 and len(docs[0]) > 0:
    found_memory = docs[0][0]
    print(f"I asked: '{question}'")
    print("The most relevant memory I found is:")
    print(found_memory)
else:
    print(f"I asked: '{question}'")
    print("I couldn't find a relevant memory for that question.")
