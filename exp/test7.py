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
llm = genai.GenerativeModel("gemini-2.5-flash-preview-09-2025")

# --- Connect to ChromaDB ---
base_dir = os.path.dirname(__file__)
db_path = os.path.join(base_dir, "my_chroma_db")
client = chromadb.PersistentClient(path=db_path)
try:
    collection = client.get_collection("random_facts")
except ValueError:
    print("Error: The 'random_facts' collection does not exist.")
    print("Please run the script from Chapter 3 to create and store your first memory.")
    exit()

print("Hello! I'm your personal AI assistant. I now have a memory of our facts.")
print("Type 'exit' or 'quit' to end the chat.")

# --- The Main Chat Loop ---
while True:
    user_question = input("\nYou: ").strip()

    if user_question.lower() in ["quit", "exit"]:
        print("Goodbye! It was nice chatting with you.")
        break

    # Check for empty input
    if not user_question:
        print("Please enter a question or type 'exit' to quit.")
        continue

    # --- Step 1: Look up relevant facts in our memory ---
    question_embedding = genai.embed_content(model=embedding_model, content=user_question)["embedding"]

    results = collection.query(query_embeddings=[question_embedding], n_results=1)

    if results and results["documents"] and results["documents"][0]:
        retrieved_memory = results["documents"][0][0]
    else:
        retrieved_memory = None

    # --- Step 2: Formulate the answer ---
    if retrieved_memory:
        prompt_with_context = (
            "You are a helpful personal assistant. "
            "Please respond in plain text without any formatting like bold or italics. "
            "Based on this fact I'm providing you: "
            f"'{retrieved_memory}'"
            "\nPlease answer the following question: "
            f"'{user_question}'"
        )
        print(f"AI (thinking with memory): I found a relevant fact... '{retrieved_memory}'")
    else:
        prompt_with_context = user_question
        print("AI (thinking): I don't have a specific memory for this, but I'll answer from my general knowledge.")

    response = llm.generate_content(prompt_with_context)

    print(f"AI: {response.text}")
