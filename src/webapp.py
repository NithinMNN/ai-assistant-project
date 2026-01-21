"""Personal AI Assistant Web Application.

A Streamlit-based chatbot with RAG (Retrieval-Augmented Generation)
that uses ChromaDB for memory storage and Google Gemini for responses.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import chromadb
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection

# Configure the page - this must be the first Streamlit command
st.set_page_config(
    page_title="My Personal AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Constants ---
EMBEDDING_MODEL = "models/text-embedding-004"
LLM_MODEL = "gemini-2.0-flash"
CHROMA_DB_PATH = "exp/my_chroma_db"
COLLECTION_NAME = "random_facts"

# Distance thresholds for confidence levels
THRESHOLD_HIGH = 0.8
THRESHOLD_MEDIUM = 1.2
THRESHOLD_MAX = 2.0

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Check for API key
if not API_KEY:
    st.error("🚨 **Google API key not found!** Please check your .env file.")
    st.write("Make sure you have a file called `.env` with your Google API key:")
    st.code("GOOGLE_API_KEY=your_actual_key_here")
    st.stop()

# Configure AI
genai.configure(api_key=API_KEY)
llm = genai.GenerativeModel(LLM_MODEL)


# --- Helper Functions ---
@st.cache_resource
def connect_to_memory() -> Collection | None:
    """Connect to the ChromaDB memory bank.

    Returns:
        The ChromaDB collection if found, None otherwise.
    """
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        return client.get_or_create_collection(COLLECTION_NAME)
    except Exception as e:
        st.error(f"Error connecting to memory: {e}")
        return None


def get_confidence_level(distance: float) -> str:
    """Determine confidence level based on embedding distance.

    Args:
        distance: The distance score from ChromaDB query.

    Returns:
        Confidence level string: "high", "medium", "low", or "none".
    """
    if distance < THRESHOLD_HIGH:
        return "high"
    elif distance < THRESHOLD_MEDIUM:
        return "medium"
    elif distance < THRESHOLD_MAX:
        return "low"
    return "none"


def get_knowledge_indicator(confidence_level: str, has_memory: bool) -> str:
    """Get the UI indicator for knowledge confidence.

    Args:
        confidence_level: The confidence level string.
        has_memory: Whether a memory was retrieved.

    Returns:
        An emoji-prefixed indicator string.
    """
    indicators = {
        "high": "🎯 *Based on what I know about you*",
        "medium": "💡 *Using some personal context*",
        "low": "📝 *With limited personal context*",
    }
    if has_memory and confidence_level in indicators:
        return indicators[confidence_level]
    return "🤔 *General advice (no specific personal info found)*"


def build_prompt(user_question: str, retrieved_memory: str | None) -> str:
    """Build the prompt for the LLM based on available context.

    Args:
        user_question: The user's question.
        retrieved_memory: The retrieved memory context, if any.

    Returns:
        The formatted prompt string.
    """
    if retrieved_memory:
        return f"""You are a helpful personal AI assistant. \
You have access to personal information about the user from their knowledge base.

Relevant personal information: {retrieved_memory}

User question: {user_question}

Instructions:
- Use the personal information to give a tailored, relevant response
- Be conversational and friendly
- If the personal information is very relevant, reference it naturally
- If the personal information is only somewhat relevant, use it as context but don't force it
- Always be helpful and engaging

Respond as the user's personal assistant who knows them well."""

    return f"""You are a helpful personal AI assistant.

User question: {user_question}

I don't have specific personal information about this topic in my knowledge base, \
so I'll provide helpful general advice. Be conversational, friendly, and as helpful as possible."""


def search_memory(collection: Collection, prompt: str) -> tuple[str | None, str]:
    """Search the memory bank for relevant information.

    Args:
        collection: The ChromaDB collection to search.
        prompt: The user's question to search for.

    Returns:
        A tuple of (retrieved_memory, confidence_level).
    """
    try:
        question_embedding = genai.embed_content(model=EMBEDDING_MODEL, content=prompt)["embedding"]

        results = collection.query(query_embeddings=[question_embedding], n_results=3)

        if results and results["documents"] and results["documents"][0]:
            best_memory = results["documents"][0][0]
            best_distance = results["distances"][0][0]

            # Debug info (can be removed in production)
            st.write(f"Debug: Best distance: {best_distance:.3f} (threshold < {THRESHOLD_MAX})")

            confidence = get_confidence_level(best_distance)
            if confidence != "none":
                return best_memory, confidence

    except Exception as e:
        st.error(f"Error searching memory: {e}")

    return None, "none"


def render_sidebar(collection: Collection | None) -> None:
    """Render the sidebar with memory status and tips.

    Args:
        collection: The ChromaDB collection, or None if not connected.
    """
    with st.sidebar:
        st.header("🧠 Memory Status")

        if collection:
            memories = collection.get()
            memory_count = len(memories["documents"])

            if memory_count > 0:
                st.success(f"💾 **{memory_count} memories** stored")

                st.subheader("Recent memories:")
                for memory in memories["documents"][:5]:
                    display_memory = memory[:100] + "..." if len(memory) > 100 else memory
                    st.write(f"• {display_memory}")

                if memory_count > 5:
                    st.write(f"*...and {memory_count - 5} more*")
            else:
                st.warning("🔄 Memory bank connected but empty")
        else:
            st.error("❌ **No memory bank found**")
            st.write("Your AI doesn't have any memories yet!")
            st.write("Run `python load_memory.py` to set up your memory bank.")

        st.markdown("---")
        st.subheader("💡 Tips")
        st.write("• Ask about your preferences, goals, or habits")
        st.write("• Your AI remembers previous conversations")
        st.write("• Try asking 'What do you know about me?'")


# --- Main Application ---
collection = connect_to_memory()

# App header
st.title("🤖 My Personal AI Assistant")
st.markdown("*An AI that actually knows you and remembers what matters*")

# Render sidebar
render_sidebar(collection)

# Main chat interface
st.header("💬 Chat with Your Assistant")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    welcome_msg = (
        "👋 Hi! I'm your personal AI assistant. I know things about you and can help "
        "with questions based on what I've learned. What would you like to talk about?"
    )
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

# Display all previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            retrieved_memory: str | None = None
            confidence_level = "none"

            if collection:
                retrieved_memory, confidence_level = search_memory(collection, prompt)

            # Build and send prompt to LLM
            enhanced_prompt = build_prompt(prompt, retrieved_memory)

            try:
                response = llm.generate_content(enhanced_prompt)
                ai_response = response.text

                # Display the response with confidence indicator
                st.markdown(ai_response)
                indicator = get_knowledge_indicator(confidence_level, retrieved_memory is not None)
                st.caption(indicator)

            except Exception as e:
                st.error(f"Error generating response: {e}")
                ai_response = "Sorry, I encountered an error while processing your question. Please try again."
                st.markdown(ai_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})

# Add a clear chat history button
if len(st.session_state.messages) > 1:
    if st.sidebar.button("🗑️ Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.rerun()
