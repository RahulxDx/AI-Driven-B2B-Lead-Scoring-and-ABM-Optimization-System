# Complete Setup Guide - B2B Lead Scoring with RAG Chatbot

## 🎯 Overview

This project now includes:
1. ✅ Lead Scoring System (XGBoost)
2. ✅ ABM Dashboard
3. ✅ **NEW: RAG-based AI Chatbot**

## 📦 Installation

### Step 1: Install All Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train the Model (First Time Only)

```bash
python b2b_lead_scoring.py
```

This will:
- Generate synthetic data
- Train 5 ML models
- Save the best model as `best_model.pkl`
- Create feature importance visualization

### Step 3: Set Up API Key (Optional but Recommended)

For the best chatbot experience, set up an LLM API key:

**Groq (Recommended - Fast & Free):**
```bash
# Windows
set GROQ_API_KEY=your-api-key-here

# Linux/Mac
export GROQ_API_KEY=your-api-key-here
```

Get your free API key at: https://console.groq.com/

**Alternative: OpenAI**
```bash
export OPENAI_API_KEY=your-api-key-here
```

**Alternative: Google Gemini**
```bash
export GEMINI_API_KEY=your-api-key-here
```

> **Note:** If no API key is set, the chatbot will use template-based responses (limited but functional).

## 🚀 Running the Application

### Option A: Run Both Services (Recommended)

**Terminal 1 - Start FastAPI Server:**
```bash
python start_api.py
```
You should see: `API will be available at: http://localhost:8000`

**Terminal 2 - Start Streamlit App:**
```bash
streamlit run app.py
```
The app will open in your browser automatically.

### Option B: Run Streamlit Only (Fallback Mode)

If you don't want to run the API server, you can just run:
```bash
streamlit run app.py
```

The chatbot will use direct query engine (slightly slower but works).

## 📱 Using the Application

### Tab 1: Lead Predictions
1. Upload your CSV file (use `demo_leads.csv` for testing)
2. Wait for processing and predictions
3. View results, top leads, and download predictions

### Tab 2: ABM Dashboard
- View ABM metrics and visualizations
- Analyze industry and region breakdowns
- Engagement analysis

### Tab 3: AI Chat Assistant
1. **Important:** Process data in Tab 1 first
2. Ask questions about your leads:
   - "Which companies have the highest lead scores?"
   - "Summarize performance for Tech industry"
   - "What's the average engagement score?"
   - "Which regions have the most high-quality leads?"

## 🧪 Testing

### Test the Complete Flow

1. **Start API:**
   ```bash
   python start_api.py
   ```

2. **Start Streamlit:**
   ```bash
   streamlit run app.py
   ```

3. **Upload Data:**
   - Go to "Lead Predictions" tab
   - Upload `demo_leads.csv`
   - Wait for predictions

4. **Test Chatbot:**
   - Go to "AI Chat Assistant" tab
   - Try: "What are the top industries?"

### Verify API is Running

Open in browser: http://localhost:8000/docs

You should see the FastAPI documentation interface.

## 🐛 Troubleshooting

### Issue: "Model file not found"
**Solution:** Run `python b2b_lead_scoring.py` first to train and save the model.

### Issue: "Error connecting to chatbot"
**Solutions:**
1. Make sure `python start_api.py` is running in another terminal
2. Check port 8000 is not in use
3. The app has fallback mode - it will still work but may be slower

### Issue: "RAG not initializing"
**Solutions:**
1. Ensure data is uploaded and processed in Tab 1
2. Check ChromaDB is installed: `pip install chromadb`
3. Check console for specific error messages

### Issue: "API key errors"
**Solutions:**
1. Verify API key is set correctly: `echo $GROQ_API_KEY` (Linux/Mac) or `echo %GROQ_API_KEY%` (Windows)
2. Check API key is valid
3. System will use template mode if no key is set

## 📁 Project Structure

```
B2B Project/
├── app.py                    # Main Streamlit app (3 tabs)
├── b2b_lead_scoring.py       # Model training script
├── start_api.py              # FastAPI server startup
├── best_model.pkl            # Trained model (generated)
├── demo_leads.csv            # Sample data
├── requirements.txt          # All dependencies
├── rag/
│   ├── __init__.py
│   ├── context_builder.py    # ChromaDB context builder
│   └── query_engine.py       # RAG query processor
├── api/
│   ├── __init__.py
│   └── chatbot_api.py        # FastAPI endpoints
└── rag/vector_store/         # ChromaDB storage (auto-created)
```

## 🔧 Configuration

### Change LLM Provider

Edit `api/chatbot_api.py` line 38:
```python
model_provider = "groq"  # or "openai", "gemini", "template"
```

### Change API Port

Edit `start_api.py`:
```python
uvicorn.run(..., port=8000)  # Change port number
```

### Adjust Retrieval

Edit `rag/query_engine.py`:
```python
contexts = self._retrieve_context(user_query, top_k=5)  # Change top_k
```

## ✅ Checklist

Before using the system:

- [ ] Installed all dependencies: `pip install -r requirements.txt`
- [ ] Trained the model: `python b2b_lead_scoring.py`
- [ ] (Optional) Set API key: `export GROQ_API_KEY=your-key`
- [ ] Started API server: `python start_api.py`
- [ ] Started Streamlit: `streamlit run app.py`
- [ ] Uploaded test data in Tab 1
- [ ] Tested chatbot in Tab 3

## 🎉 You're All Set!

The complete system is now ready:
- ✅ Lead scoring with XGBoost
- ✅ ABM dashboard
- ✅ RAG-based AI chatbot

Enjoy your AI-powered B2B lead scoring system!

## 📞 Support

For issues:
1. Check the console/terminal for error messages
2. Verify all dependencies are installed
3. Ensure API server is running (if using API mode)
4. Check README_RAG.md for detailed RAG documentation

