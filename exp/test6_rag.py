import google.generativeai as genai
import os
from dotenv import load_dotenv
import chromadb
from chromadb.errors import NotFoundError

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("Error: GOOGLE_API_KEY not found in .env file.")
    exit()

genai.configure(api_key=API_KEY)
embedding_model = "models/text-embedding-004"
llm = genai.GenerativeModel("gemini-2.5-flash-preview-09-2025")

base_dir = os.path.dirname(__file__)
db_path = os.path.join(base_dir, "my_chroma_db")

client = chromadb.PersistentClient(path=db_path)
try:
    collection = client.get_collection("random_facts")
except NotFoundError:
    print("Error: The 'random_facts' collection does not exist.")
    print("Please run test4_vectordb.py first to create and store your first memory.")
    exit()

print("Hello! I'm your personal AI assistant. I now have a memory of our facts.")
print("Type 'exit' or 'quit' to end the chat.")

# --- The Main Chat Loop ---
while True:
    user_question = input("\nYou: ")

    if user_question.lower() in ["quit", "exit"]:
        print("Goodbye! It was nice chatting with you.")
        break

    # --- Step 1: Look up relevant facts in our memory ---
    # First, we create an embedding for the user's question
    question_embedding = genai.embed_content(model=embedding_model, content=user_question)["embedding"]

    # Then, we query our collection to find the most relevant memory
    results = collection.query(
        query_embeddings=[question_embedding], n_results=1  # We only want the single most relevant fact
    )

    # Let's get the text of the most relevant memory
    if results and results["documents"] and results["documents"][0]:
        retrieved_memory = results["documents"][0][0]
    else:
        retrieved_memory = None  # No memory was found

    # --- Step 2: Formulate the answer ---
    if retrieved_memory:
        # We found a relevant memory! Let's use it to augment our prompt.
        prompt_with_context = (
            "You are a helpful personal assistant. "
            "Based on this fact I'm providing you: "
            f"'{retrieved_memory}'"
            "\nPlease answer the following question: "
            f"'{user_question}'"
        )
        print(f"AI (thinking with memory): I found a stored fact -  '{retrieved_memory}'")
    else:
        # We didn't find a relevant memory. We'll just ask the AI the question directly.
        prompt_with_context = user_question
        print("AI (thinking): I don't have a specific memory for this, but I'll answer from my general knowledge.")

    # Now, send the (potentially augmented) prompt to the Gemini LLM
    response = llm.generate_content(prompt_with_context)

    print(f"AI: {response.text}")
