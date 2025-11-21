import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')

print("Hello! I'm your personal AI assistant. Ask me anything!")
print("Type '--exit' or '--quit' to end the chat.")

while True:
    prompt = input("You: ")
    
    if prompt in ["--quit", "--exit"]:
        print("Goodbye! It was nice chatting with you.")
        break
    
    response = model.generate_content(prompt)

    print(f"AI: {response.text}")        