# 🤖 AI Assistant Project

A personal AI assistant with **memory capabilities** using RAG (Retrieval-Augmented Generation). Built to learn AI implementation concepts including Vector Databases, Embeddings, and LLM integration.

## ✨ Features

- **Conversational AI** - Powered by Google Gemini LLM
- **Persistent Memory** - ChromaDB vector database for storing personal facts
- **Semantic Search** - Find relevant memories based on meaning, not keywords
- **RAG Pipeline** - Augments LLM responses with personal knowledge
- **Web Interface** - Streamlit-based chat UI

## 🏗️ Project Structure

```
ai-assistant-project/
├── src/
│   └── webapp.py           # Streamlit web application
├── exp/                    # Experimental/learning scripts
│   ├── test1.py            # Basic LLM API call
│   ├── test2.py            # Interactive chatbot
│   ├── test3.py            # Embeddings introduction
│   ├── test4_vectordb.py   # ChromaDB storage
│   ├── test5_*.py          # Semantic search experiments
│   ├── test6_rag.py        # Complete RAG pipeline
│   ├── test7.py            # Polished RAG chatbot
│   ├── load_memory.py      # Batch memory loader
│   ├── my_chroma_db/       # Vector database (gitignored)
│   └── memory/             # Knowledge files (.txt)
├── .env                    # API keys (gitignored)
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Tool configurations
├── .pre-commit-config.yaml # Git hooks for code quality
└── .github/workflows/      # CI/CD pipeline
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Google API Key ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd ai-assistant-project

# Create virtual environment
python -m venv myenv
myenv\Scripts\activate  # Windows
# source myenv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

### Load Memories

```bash
# Add your knowledge files to exp/memory/*.txt
# Then run:
python exp/load_memory.py
```

### Run the Web App

```bash
streamlit run src/webapp.py
```

## 🛠️ Development

### Code Quality Tools

This project uses:
- **Black** - Code formatting
- **Flake8** - Linting
- **MyPy** - Type checking
- **Typos** - Spell checking
- **Pre-commit** - Git hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run all checks manually
pre-commit run --all-files
```

### Running Tests

```bash
# Individual tools
black src/ --check
flake8 src/
mypy src/
```

## 📚 Learning Path

The `exp/` folder contains progressive learning scripts:

| Script | Concept |
|--------|---------|
| `test1.py` | Basic LLM API call |
| `test2.py` | Interactive chatbot with env vars |
| `test3.py` | Text embeddings (semantic GPS) |
| `test4_vectordb.py` | ChromaDB vector storage |
| `test5_*.py` | Semantic search & similarity |
| `test6_rag.py` | Full RAG pipeline |
| `test7.py` | Chatbot |

## 🔧 Tech Stack

| Technology | Purpose |
|------------|---------|
| Google Gemini | LLM for text generation |
| text-embedding-004 | Embedding model |
| ChromaDB | Vector database |
| Streamlit | Web interface |
| Pre-commit | Code quality automation |