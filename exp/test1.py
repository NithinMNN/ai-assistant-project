import google.generativeai as genai

API_KEY = "AIzaSyBv-h93UQX5QRlFzlQmBhRaTEj5RpnOXmU"

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash-preview-09-2025")

prompt = "In simple terms, what is a solar eclipse?"

response = model.generate_content(prompt)

print(response.text)
