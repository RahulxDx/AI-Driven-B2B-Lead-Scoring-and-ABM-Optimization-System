"""
AI-Driven B2B Lead Scoring and Account-Based Marketing (ABM) Optimization System - Streamlit App
Production-ready web application with RAG-based AI Chatbot
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import matplotlib.pyplot as plt
import requests
import json
from pathlib import Path
import sys

warnings.filterwarnings('ignore')

# Add paths for RAG modules
sys.path.append(str(Path(__file__).parent))

# Page configuration
st.set_page_config(
    page_title="B2B Lead Scoring & ABM System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 20%;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.title("🎯 B2B Lead Scoring")
    st.markdown("---")
    
    st.subheader("📋 Project Info")
    st.markdown("""
    **Project:** AI-Driven B2B Lead Scoring and ABM Optimization System
    
    **Model Type:** XGBoost Classifier
    
    **Purpose:** Predict lead quality and get AI-powered insights
    """)
    
    st.markdown("---")
    
    st.subheader("🤖 AI Assistant Info")
    st.markdown("""
    This assistant uses **Retrieval-Augmented Generation (RAG)** to answer 
    questions based on your uploaded leads and ABM insights.
    
    Ask questions like:
    - "Which companies have the highest lead scores?"
    - "Summarize performance for Tech industry"
    - "What's the engagement trend?"
    """)
    
    st.markdown("---")
    
    st.subheader("📊 Required Features")
    st.markdown("""
    **Categorical:**
    - Company_Size, Industry
    - Lead_Source, Region
    
    **Numerical:**
    - Annual_Revenue, Website_Visits
    - Email_Opens, Ad_Clicks
    - Engagement_Score
    - Previous_Purchases
    """)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_model():
    """Load the saved model package"""
    try:
        with open('best_model.pkl', 'rb') as f:
            model_package = pickle.load(f)
        return model_package
    except FileNotFoundError:
        st.error("❌ Model file 'best_model.pkl' not found. Run b2b_lead_scoring.py first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

def initialize_rag_context(df):
    """Initialize RAG context from dataframe"""
    try:
        import warnings
        warnings.filterwarnings('ignore')
        from rag.context_builder import ContextBuilder
        builder = ContextBuilder()
        builder.build_context(df, force_rebuild=False)
        return builder
    except ImportError as e:
        error_msg = str(e)
        if "cached_download" in error_msg or "Version compatibility" in error_msg:
            st.error(
                "❌ **Version Compatibility Issue Detected**\n\n"
                "The sentence-transformers and huggingface_hub packages have a version conflict.\n\n"
                "**Quick Fix:**\n"
                "1. Run: `python fix_imports.py`\n"
                "2. Or manually run:\n"
                "   ```bash\n"
                "   pip uninstall -y sentence-transformers huggingface-hub\n"
                "   pip install 'huggingface-hub>=0.16.0,<0.20.0' 'sentence-transformers>=2.2.0,<3.0.0'\n"
                "   ```\n"
                "3. Restart the application"
            )
        else:
            st.error(f"❌ Import Error: {error_msg}")
        return None
    except Exception as e:
        # Only show warning for other errors
        st.warning(f"⚠️ RAG initialization warning: {str(e)}")
        return None

def call_chatbot_api(query: str, api_url: str = "http://localhost:8000/chat"):
    """Call the FastAPI chatbot endpoint"""
    try:
        response = requests.post(
            api_url,
            json={"query": query, "top_k": 5},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 500:
            # Try to get error details
            try:
                error_detail = response.json().get("detail", "Unknown error")
                return {"response": f"API Error: {error_detail}. Please check if data is uploaded and processed.", "contexts_used": 0}
            except:
                return {"response": f"API Error 500: Internal server error. Check API server logs for details.", "contexts_used": 0}
        else:
            return {"response": f"API Error: {response.status_code} - {response.text}", "contexts_used": 0}
    except requests.exceptions.ConnectionError:
        # Fallback to direct query engine if API not available
        try:
            from rag.context_builder import ContextBuilder
            from rag.query_engine import QueryEngine
            
            builder = ContextBuilder()
            collection = builder.get_collection()
            
            # Check if collection has data
            count = collection.count()
            if count == 0:
                return {"response": "No data available. Please upload and process lead data first.", "contexts_used": 0}
            
            engine = QueryEngine(collection, model_provider="template")
            result = engine.query(query)
            return result
        except Exception as e:
            return {"response": f"Error: {str(e)}. Please ensure data is uploaded and processed in Tab 1.", "contexts_used": 0}
    except Exception as e:
        return {"response": f"Error connecting to chatbot: {str(e)}", "contexts_used": 0}

# ============================================================================
# MAIN CONTENT
# ============================================================================
st.markdown('<p class="main-header">🎯 B2B Lead Scoring & ABM Optimization System</p>', 
            unsafe_allow_html=True)

# Load model
model_package = load_model()
model = model_package['model']
scaler = model_package['scaler']
feature_names = model_package['feature_names']
model_name = model_package.get('model_name', 'XGBoost')

# Initialize session state
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'rag_initialized' not in st.session_state:
    st.session_state.rag_initialized = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Lead Predictions", "📈 ABM Dashboard", "🤖 AI Chat Assistant"])

# ============================================================================
# TAB 1: LEAD PREDICTIONS
# ============================================================================
with tab1:
    st.info(f"✅ **Model Loaded:** {model_name} | **Features:** {len(feature_names)}")
    
    st.markdown("---")
    st.subheader("📤 Upload Lead Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file containing lead data with the required features",
        key="uploader_tab1"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df)} leads from CSV file")
            
            # Store in session state
            st.session_state.df_processed = df
            
            # Display preview
            with st.expander("📋 Preview Uploaded Data", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
                st.caption(f"Total rows: {len(df)} | Total columns: {len(df.columns)}")
            
            # Preprocessing
            st.markdown("---")
            st.subheader("🔧 Data Preprocessing")
            
            required_categorical = ['Company_Size', 'Industry', 'Lead_Source', 'Region']
            required_numerical = ['Annual_Revenue', 'Website_Visits', 'Email_Opens', 
                                 'Ad_Clicks', 'Engagement_Score', 'Previous_Purchases']
            required_cols = required_categorical + required_numerical
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            else:
                df_processed = df.copy()
                
                # Handle missing values
                for col in required_numerical:
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
                for col in required_categorical:
                    df_processed[col].fillna(df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else 'Unknown', inplace=True)
                
                # Encode and scale
                df_encoded = pd.get_dummies(
                    df_processed[required_categorical + required_numerical],
                    columns=required_categorical,
                    prefix=required_categorical
                )
                df_encoded[required_numerical] = scaler.transform(df_encoded[required_numerical])
                
                for feature in feature_names:
                    if feature not in df_encoded.columns:
                        df_encoded[feature] = 0
                df_encoded = df_encoded[feature_names]
                
                # Predictions
                st.markdown("---")
                st.subheader("🔮 Predictions")
                
                predictions = model.predict(df_encoded)
                prediction_proba = model.predict_proba(df_encoded)
                
                score_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
                df['Predicted_Lead_Score'] = [score_labels[pred] for pred in predictions]
                df['Probability_Low'] = prediction_proba[:, 0]
                df['Probability_Medium'] = prediction_proba[:, 1]
                df['Probability_High'] = prediction_proba[:, 2]
                df['Confidence'] = np.max(prediction_proba, axis=1)
                
                # Update session state
                st.session_state.df_processed = df
                
                # Initialize RAG context
                if not st.session_state.rag_initialized:
                    with st.spinner("🔍 Initializing AI context..."):
                        initialize_rag_context(df)
                        st.session_state.rag_initialized = True
                
                st.markdown('<div class="success-box">✅ <strong>Predictions completed!</strong></div>', 
                           unsafe_allow_html=True)
                
                # Summary statistics
                st.markdown("---")
                st.subheader("📊 Summary Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                high_count = (df['Predicted_Lead_Score'] == 'High').sum()
                medium_count = (df['Predicted_Lead_Score'] == 'Medium').sum()
                low_count = (df['Predicted_Lead_Score'] == 'Low').sum()
                
                with col1:
                    st.metric("Total Leads", len(df))
                with col2:
                    st.metric("High Quality", high_count, delta=f"{high_count/len(df)*100:.1f}%")
                with col3:
                    st.metric("Medium Quality", medium_count, delta=f"{medium_count/len(df)*100:.1f}%")
                with col4:
                    st.metric("Low Quality", low_count, delta=f"{low_count/len(df)*100:.1f}%")
                
                # Best leads
                st.markdown("---")
                st.subheader("🏆 Top Leads")
                
                df['Lead_Rank_Score'] = 0
                score_weights = {'High': 1000, 'Medium': 500, 'Low': 0}
                df['Lead_Rank_Score'] = df['Predicted_Lead_Score'].map(score_weights)
                df['Lead_Rank_Score'] += df['Probability_High'] * 100
                df['Lead_Rank_Score'] += df['Confidence'] * 50 * (df['Predicted_Lead_Score'].isin(['High', 'Medium']).astype(int))
                
                if 'Annual_Revenue' in df.columns:
                    rev_max, rev_min = df['Annual_Revenue'].max(), df['Annual_Revenue'].min()
                    if rev_max > rev_min:
                        df['Lead_Rank_Score'] += ((df['Annual_Revenue'] - rev_min) / (rev_max - rev_min)) * 30
                
                if 'Engagement_Score' in df.columns:
                    eng_max, eng_min = df['Engagement_Score'].max(), df['Engagement_Score'].min()
                    if eng_max > eng_min:
                        df['Lead_Rank_Score'] += ((df['Engagement_Score'] - eng_min) / (eng_max - eng_min)) * 20
                
                df_ranked = df.sort_values('Lead_Rank_Score', ascending=False).copy()
                df_ranked['Rank'] = range(1, len(df_ranked) + 1)
                
                top_n = st.slider("Show Top N Leads", min_value=5, max_value=min(30, len(df)), value=10, step=5)
                
                top_cols = ['Rank', 'Company_Size', 'Industry', 'Lead_Source', 'Region',
                           'Annual_Revenue', 'Engagement_Score', 'Predicted_Lead_Score',
                           'Probability_High', 'Confidence']
                available_cols = [col for col in top_cols if col in df_ranked.columns]
                df_top = df_ranked[available_cols].head(top_n).copy()
                
                # Format
                for col in ['Annual_Revenue', 'Engagement_Score']:
                    if col in df_top.columns:
                        df_top[col] = df_top[col].apply(lambda x: f"{x:.0f}" if isinstance(x, (int, float)) else x)
                for col in ['Probability_High', 'Confidence']:
                    if col in df_top.columns:
                        df_top[col] = df_top[col].apply(lambda x: f"{x:.1%}" if isinstance(x, (int, float)) else x)
                
                def highlight(row):
                    if row['Predicted_Lead_Score'] == 'High':
                        return ['background-color: #90EE90'] * len(row)
                    elif row['Predicted_Lead_Score'] == 'Medium':
                        return ['background-color: #FFE4B5'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(df_top.style.apply(highlight, axis=1), use_container_width=True, height=350)
                
                # Download
                st.markdown("---")
                csv = df.to_csv(index=False)
                st.download_button("📥 Download Predictions as CSV", data=csv, 
                                 file_name="lead_predictions.csv", mime="text/csv")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

# ============================================================================
# TAB 2: ABM DASHBOARD
# ============================================================================
with tab2:
    st.subheader("📈 Account-Based Marketing Dashboard")
    
    if st.session_state.df_processed is not None:
        df = st.session_state.df_processed
        
        # ABM Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'Industry' in df.columns:
                top_industry = df['Industry'].value_counts().index[0]
                st.metric("Top Industry", top_industry)
        
        with col2:
            if 'Region' in df.columns:
                top_region = df['Region'].value_counts().index[0]
                st.metric("Top Region", top_region)
        
        with col3:
            if 'Annual_Revenue' in df.columns:
                avg_revenue = df['Annual_Revenue'].mean()
                st.metric("Avg Revenue", f"${avg_revenue:,.0f}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Lead Score Distribution")
            if 'Predicted_Lead_Score' in df.columns:
                score_counts = df['Predicted_Lead_Score'].value_counts()
                st.bar_chart(score_counts)
        
        with col2:
            st.markdown("#### 🏢 Industry Breakdown")
            if 'Industry' in df.columns:
                industry_counts = df['Industry'].value_counts().head(10)
                st.bar_chart(industry_counts)
        
        # Engagement Analysis
        if 'Engagement_Score' in df.columns:
            st.markdown("---")
            st.markdown("#### 📈 Engagement Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                avg_engagement = df['Engagement_Score'].mean()
                st.metric("Average Engagement", f"{avg_engagement:.1f}/100")
            
            with col2:
                high_engagement = (df['Engagement_Score'] > 70).sum()
                st.metric("High Engagement Leads", high_engagement)
        
    else:
        st.info("👆 Please upload and process lead data in the 'Lead Predictions' tab first.")

# ============================================================================
# TAB 3: AI CHAT ASSISTANT
# ============================================================================
with tab3:
    st.subheader("🤖 AI Chat Assistant")
    st.markdown("Ask questions about your leads, predictions, and ABM insights!")
    
    # Check if data is available
    if st.session_state.df_processed is None:
        st.warning("⚠️ Please upload and process lead data in the 'Lead Predictions' tab first to enable the AI assistant.")
    else:
        # Chat interface
        st.markdown("---")
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("#### 💬 Conversation:")
            for msg in st.session_state.chat_history:
                if msg['role'] == 'user':
                    with st.chat_message("user"):
                        st.write(msg['content'])
                else:
                    with st.chat_message("assistant"):
                        st.write(msg['content'])
                        if 'contexts_used' in msg and msg['contexts_used'] > 0:
                            st.caption(f"📚 Used {msg['contexts_used']} context sources")
        else:
            st.info("💡 Start a conversation by asking a question below!")
        
        # Chat input - using text_input instead of chat_input (since it can't be in tabs)
        st.markdown("---")
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_query = st.text_input(
                "Ask a question about your leads...",
                key="chat_input",
                placeholder="Type your question here and click Send or press Enter",
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.button("Send ➤", use_container_width=True, type="primary")
        
        # Process query if submitted via button
        if send_button and user_query and user_query.strip():
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Get AI response
            with st.spinner("🤔 Thinking..."):
                result = call_chatbot_api(user_query)
                response = result.get("response", "Sorry, I couldn't generate a response.")
                contexts_used = result.get("contexts_used", 0)
            
            # Add assistant response to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "contexts_used": contexts_used
            })
            
            # Clear input and rerun
            st.session_state.chat_input = ""
            st.rerun()
        
        # Reset chat button
        if st.session_state.chat_history and st.button("🔄 Reset Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.last_processed_query = ""
            st.rerun()
        
        # Example questions
        st.markdown("---")
        st.markdown("#### 💡 Example Questions:")
        example_questions = [
            "Which companies have the highest lead scores?",
            "Summarize performance for Tech industry",
            "What's the average engagement score?",
            "Which regions have the most high-quality leads?",
            "What percentage of leads are High quality?"
        ]
        
        # Display example questions as clickable buttons
        for i, question in enumerate(example_questions):
            if st.button(question, key=f"example_{i}", use_container_width=True):
                # Add to chat input (will be processed by the chat_input handler)
                st.session_state.example_question = question
                st.rerun()
        
        # Handle example question if set
        if 'example_question' in st.session_state and st.session_state.example_question:
            example_q = st.session_state.example_question
            del st.session_state.example_question
            
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": example_q})
            
            # Get AI response
            with st.spinner("🤔 Thinking..."):
                result = call_chatbot_api(example_q)
                response = result.get("response", "Sorry, I couldn't generate a response.")
                contexts_used = result.get("contexts_used", 0)
            
            # Add assistant response
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "contexts_used": contexts_used
            })
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "AI-Driven B2B Lead Scoring & ABM System | Powered by XGBoost & RAG"
    "</div>",
    unsafe_allow_html=True
)
