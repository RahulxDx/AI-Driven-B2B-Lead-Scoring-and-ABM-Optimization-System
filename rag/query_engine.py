"""
RAG Query Engine for B2B Lead Scoring System
Handles query processing, retrieval, and LLM response generation
"""

import os
from typing import List, Dict, Optional
from chromadb import Collection
import json
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    error_msg = str(e)
    if "cached_download" in error_msg:
        raise ImportError(
            f"Version compatibility issue with sentence-transformers and huggingface_hub.\n"
            f"Please run: python fix_imports.py\n"
            f"Or manually: pip install 'huggingface-hub>=0.16.0,<0.20.0' 'sentence-transformers>=2.2.0,<3.0.0'"
        )
    else:
        raise ImportError(f"Failed to import SentenceTransformer: {e}. Please install: pip install sentence-transformers")

# Try to import LLM providers (fallback to OpenAI if others not available)
try:
    import groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class QueryEngine:
    """Handles RAG queries and LLM responses"""
    
    def __init__(self, collection: Collection, model_provider: str = "openai"):
        """
        Initialize query engine
        
        Args:
            collection: ChromaDB collection
            model_provider: LLM provider ("openai", "groq", or "gemini")
        """
        self.collection = collection
        self.model_provider = model_provider.lower()
        
        # Initialize embedding model with error handling
        print("Loading embedding model for queries...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Embedding model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load embedding model: {e}")
            raise
        
        # Initialize LLM based on provider
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM based on provider"""
        if self.model_provider == "groq":
            if not GROQ_AVAILABLE:
                raise ImportError("groq package not installed. Install with: pip install groq")
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set")
            self.llm_client = groq.Groq(api_key=api_key)
            self.model_name = "llama-3.1-70b-versatile"
        
        elif self.model_provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError("google-generativeai package not installed")
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            genai.configure(api_key=api_key)
            self.llm_client = genai.GenerativeModel('gemini-pro')
            self.model_name = "gemini-pro"
        
        else:  # Default to OpenAI
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package not installed. Install with: pip install openai")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                # Fallback to a simple template-based response
                print("Warning: OPENAI_API_KEY not set. Using template-based responses.")
                self.llm_client = None
                self.model_name = "template"
            else:
                self.llm_client = openai.OpenAI(api_key=api_key)
                self.model_name = "gpt-3.5-turbo"
    
    def _retrieve_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant context from ChromaDB
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        contexts = []
        if results['documents'] and len(results['documents']) > 0:
            for i, doc in enumerate(results['documents'][0]):
                contexts.append({
                    "text": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else 0
                })
        
        return contexts
    
    def _format_context(self, contexts: List[Dict]) -> str:
        """
        Format retrieved contexts into a single string
        
        Args:
            contexts: List of context dictionaries
            
        Returns:
            Formatted context string
        """
        if not contexts:
            return "No relevant context found."
        
        formatted = "Relevant Information:\n\n"
        for i, ctx in enumerate(contexts, 1):
            formatted += f"{i}. {ctx['text']}\n\n"
        
        return formatted
    
    def _generate_llm_response(self, query: str, context: str) -> str:
        """
        Generate response using LLM
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            LLM-generated response
        """
        # Build prompt
        prompt = f"""You are an AI assistant for a B2B Lead Scoring and Account-Based Marketing (ABM) system.
You help users understand their lead data, predictions, and provide insights.

Use the following context to answer the user's question. Be concise, accurate, and helpful.
If the context doesn't contain enough information, say so politely.

Context:
{context}

User Question: {query}

Provide a helpful, professional response:"""

        # If no LLM client available, use template-based response
        if self.llm_client is None:
            return self._template_response(query, context)
        
        try:
            if self.model_provider == "groq":
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful B2B marketing analytics assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                return response.choices[0].message.content.strip()
            
            elif self.model_provider == "gemini":
                response = self.llm_client.generate_content(prompt)
                return response.text.strip()
            
            else:  # OpenAI
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful B2B marketing analytics assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                return response.choices[0].message.content.strip()
        
        except Exception as e:
            return f"Error generating response: {str(e)}. Please check your API key and try again."
    
    def _template_response(self, query: str, context: str) -> str:
        """
        Fallback template-based response when LLM is not available
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Template-based response
        """
        query_lower = query.lower()
        
        # Simple keyword-based responses
        if "high" in query_lower or "best" in query_lower or "top" in query_lower:
            if "lead" in context.lower() or "high" in context.lower():
                return "Based on the data, high-quality leads typically have higher engagement scores, larger company sizes, and higher annual revenue. Focus on leads with 'High' predicted scores and high confidence levels."
            else:
                return "I found some relevant information, but please check the 'High Quality' metric in the dashboard for the most accurate count."
        
        elif "industry" in query_lower:
            if "industry" in context.lower():
                lines = [line for line in context.split('\n') if 'industry' in line.lower()]
                return "Based on the data:\n" + "\n".join(lines[:3])
            else:
                return "Please check the 'Top Industries' chart in the dashboard for industry breakdown."
        
        elif "revenue" in query_lower or "revenue" in query_lower:
            if "revenue" in context.lower():
                lines = [line for line in context.split('\n') if 'revenue' in line.lower()]
                return "Revenue information:\n" + "\n".join(lines[:3])
            else:
                return "Please check the 'Annual Revenue' column in the results table for revenue details."
        
        elif "engagement" in query_lower:
            if "engagement" in context.lower():
                lines = [line for line in context.split('\n') if 'engagement' in line.lower()]
                return "Engagement metrics:\n" + "\n".join(lines[:3])
            else:
                return "Please check the 'Engagement Score' column in the results table for engagement details."
        
        else:
            return f"Based on the available data: {context[:200]}...\n\nFor more detailed information, please check the dashboard metrics and tables."
    
    def query(self, user_query: str, top_k: int = 5) -> Dict:
        """
        Process a user query and return response
        
        Args:
            user_query: User's question
            top_k: Number of context chunks to retrieve
            
        Returns:
            Dictionary with response and metadata
        """
        # Retrieve relevant context
        contexts = self._retrieve_context(user_query, top_k=top_k)
        
        # Format context
        context_str = self._format_context(contexts)
        
        # Generate LLM response
        response = self._generate_llm_response(user_query, context_str)
        
        return {
            "response": response,
            "contexts_used": len(contexts),
            "sources": [ctx['metadata'] for ctx in contexts]
        }


if __name__ == "__main__":
    # Test query engine
    from context_builder import ContextBuilder
    
    print("Testing Query Engine...")
    builder = ContextBuilder()
    collection = builder.get_collection()
    
    engine = QueryEngine(collection, model_provider="template")
    result = engine.query("What are the top industries?")
    print(f"Response: {result['response']}")

