import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("Error: 'GOOGLE_API_KEY' not found in environment variables.")
    print("Please create a .env file with GOOGLE_API_KEY='Your_Key_Here' and try again.")
    exit()
genai.configure(api_key=API_KEY)
try:
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, 'memory', 'something.txt')
    with open(file_path, 'r') as f:
        memory_txt = f.read()
except FileNotFoundError:
    print("Error: The 'memory/something.txt' file was not found.")
    print("Please make sure you've created the folder and file as described.")
    exit()
print(f"Original text from your note: {memory_txt}")
embedding_model = 'models/text-embedding-004'
try:
    embedding = genai.embed_content(
        model=embedding_model,
        content=memory_txt
    )
    vector = embedding['embedding']
    print("\n✅ I've turned that text into a 'vector' or 'embedding'.")
    print("Here's a small sample of it:")
    print(str(vector[:5]) + "...")
    print(f"\nIn total, it's a list of {len(vector)} numbers.")
    print("This is the 'GPS coordinate' for your note's meaning!")
except Exception as e:
    print(f"\n❌ An error occurred while generating the embedding: {e}")