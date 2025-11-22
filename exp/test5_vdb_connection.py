import google.generativeai as genai
import os
from dotenv import load_dotenv
import chromadb

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("Error: GOOGLE_API_KEY not found in .env file.")
    exit()

genai.configure(api_key=API_KEY)

client = chromadb.PersistentClient(path="my_chroma_db") 

try:
    collection = client.get_collection("random_facts")
    print("Successfully connected to the 'random_facts' memory bank.")
except ValueError:
    print("Error: The 'random_facts' collection was not found.")
    print("Please run the script from the previous chapter first to create and store your first memory.")
    exit()


question = "What is the day today?"

embedding_model = 'models/text-embedding-004'
question_embedding = genai.embed_content(
    model=embedding_model,
    content=question
)['embedding']

result = collection.query(
    query_embeddings=[question_embedding],
    n_results=1
)

if result and result['documents'] and result['documents'][0]:
    found_memory = result['documents'][0][0]
    print(f"\nI asked: '{question}'")
    print("...")
    print(f"The most relevant memory I found is: '{found_memory}'")
else:
    print(f"\nI asked: '{question}'")
    print("...")
    print("I couldn't find a relevant memory for that question.")     