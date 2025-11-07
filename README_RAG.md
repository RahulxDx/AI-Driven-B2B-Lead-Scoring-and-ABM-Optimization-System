# RAG-Based AI Chatbot for B2B Lead Scoring System

This document explains how to use the RAG (Retrieval-Augmented Generation) based AI chatbot feature.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys (Optional but Recommended)

For best performance, set up an LLM API key. Choose one:

**Option A: OpenAI**
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

**Option B: Groq (Fast & Free)**
```bash
export GROQ_API_KEY="your-groq-api-key"
```

**Option C: Google Gemini**
```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

If no API key is set, the system will use template-based responses (limited functionality).

### 3. Start the FastAPI Server

**Terminal 1:**
```bash
python start_api.py
```

The API will run at `http://localhost:8000`

### 4. Start the Streamlit App

**Terminal 2:**
```bash
streamlit run app.py
```

### 5. Use the Chatbot

1. Upload your CSV file in the "Lead Predictions" tab
2. Wait for predictions to complete (RAG context will be built automatically)
3. Go to the "AI Chat Assistant" tab
4. Start asking questions!

## 📁 Project Structure

```
B2B Project/
├── app.py                    # Main Streamlit app with tabs
├── start_api.py              # FastAPI server startup script
├── rag/
│   ├── __init__.py
│   ├── context_builder.py    # Builds ChromaDB vector store
│   └── query_engine.py       # Handles RAG queries
├── api/
│   ├── __init__.py
│   └── chatbot_api.py        # FastAPI endpoints
└── rag/vector_store/         # ChromaDB persistent storage (created automatically)
```

## 🤖 How It Works

1. **Context Building**: When you upload data, the system converts each lead into descriptive text chunks
2. **Vector Storage**: Text chunks are embedded using sentence-transformers and stored in ChromaDB
3. **Query Processing**: When you ask a question:
   - Your question is embedded
   - Similar chunks are retrieved from ChromaDB
   - Context + question is sent to LLM
   - Response is generated and displayed

## 💬 Example Questions

- "Which companies have the highest lead scores?"
- "Summarize performance for Tech industry"
- "What's the average engagement score?"
- "Which regions have the most high-quality leads?"
- "What percentage of leads are High quality?"
- "Show me insights about companies in Finance industry"

## 🔧 Configuration

### Change LLM Provider

Edit `api/chatbot_api.py` line 38:
```python
model_provider = "openai"  # or "groq", "gemini", "template"
```

### Adjust Retrieval Settings

Edit `rag/query_engine.py` in the `query()` method:
```python
contexts = self._retrieve_context(user_query, top_k=5)  # Change top_k
```

## 🐛 Troubleshooting

### API Connection Error

If you see "Error connecting to chatbot":
1. Make sure the FastAPI server is running (`python start_api.py`)
2. Check that port 8000 is not in use
3. The app will fallback to direct query engine if API is unavailable

### RAG Not Initializing

If RAG context doesn't build:
1. Check that you have uploaded and processed data in Tab 1
2. Ensure ChromaDB is installed: `pip install chromadb`
3. Check console for error messages

### No API Key

If you haven't set an API key:
- The system will use template-based responses
- Functionality is limited but still works
- For best results, set up an API key (see Step 2 above)

## 📊 Features

✅ **Automatic Context Building**: Context is built when data is uploaded
✅ **Persistent Storage**: ChromaDB stores embeddings locally
✅ **Smart Caching**: Skips rebuilding if data hasn't changed
✅ **Multiple LLM Support**: OpenAI, Groq, Gemini, or template fallback
✅ **Beautiful UI**: Chat interface with Streamlit chat components
✅ **Context Sources**: Shows how many sources were used for each answer

## 🎯 Advanced Usage

### Rebuild Context Manually

If you want to force a context rebuild, you can modify the code to call:
```python
builder.build_context(df, force_rebuild=True)
```

### Custom Embeddings

To use a different embedding model, edit `rag/context_builder.py`:
```python
self.embedding_model = SentenceTransformer('your-model-name')
```

## 📝 Notes

- The vector store is persisted in `./rag/vector_store/`
- Context is automatically rebuilt when data changes
- The chat history is stored in Streamlit session state
- API runs on port 8000 by default (change in `start_api.py` if needed)

## 🔐 Security

- API keys are read from environment variables (never hardcode)
- CORS is enabled for local development (restrict in production)
- Vector store is stored locally (not sent to external services)

---

**Built with**: ChromaDB, Sentence Transformers, FastAPI, Streamlit, and your favorite LLM!

