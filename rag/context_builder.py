"""
RAG Context Builder for B2B Lead Scoring System
Converts lead data into text chunks and stores them in ChromaDB for retrieval
"""

import pandas as pd
import numpy as np
from pathlib import Path
import chromadb
from chromadb.config import Settings
import hashlib
import json
import warnings
from typing import List, Dict, Optional

# Suppress huggingface_hub warnings
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

class ContextBuilder:
    """Builds and manages vector store for RAG-based chatbot"""
    
    def __init__(self, collection_name: str = "b2b_leads", persist_directory: str = "./rag/vector_store"):
        """
        Initialize the context builder
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model with error handling
        print("Loading embedding model...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Embedding model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load embedding model: {e}")
            raise
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "B2B Lead Scoring and ABM data"}
            )
            print(f"Created new collection: {collection_name}")
    
    def _generate_data_hash(self, df: pd.DataFrame) -> str:
        """Generate hash of dataframe to detect changes"""
        data_str = df.to_string()
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _convert_lead_to_text(self, row: pd.Series) -> str:
        """
        Convert a single lead row into a descriptive text chunk
        
        Args:
            row: Pandas Series representing a single lead
            
        Returns:
            Formatted text description of the lead
        """
        parts = []
        
        # Company information
        if 'Company_Size' in row and pd.notna(row['Company_Size']):
            parts.append(f"Company Size: {row['Company_Size']}")
        
        if 'Industry' in row and pd.notna(row['Industry']):
            parts.append(f"Industry: {row['Industry']}")
        
        if 'Region' in row and pd.notna(row['Region']):
            parts.append(f"Region: {row['Region']}")
        
        # Financial metrics
        if 'Annual_Revenue' in row and pd.notna(row['Annual_Revenue']):
            parts.append(f"Annual Revenue: ${row['Annual_Revenue']:,.0f}")
        
        # Engagement metrics
        if 'Engagement_Score' in row and pd.notna(row['Engagement_Score']):
            parts.append(f"Engagement Score: {row['Engagement_Score']:.1f}/100")
        
        if 'Website_Visits' in row and pd.notna(row['Website_Visits']):
            parts.append(f"Website Visits: {row['Website_Visits']}")
        
        if 'Email_Opens' in row and pd.notna(row['Email_Opens']):
            parts.append(f"Email Opens: {row['Email_Opens']}")
        
        if 'Ad_Clicks' in row and pd.notna(row['Ad_Clicks']):
            parts.append(f"Ad Clicks: {row['Ad_Clicks']}")
        
        # Lead source
        if 'Lead_Source' in row and pd.notna(row['Lead_Source']):
            parts.append(f"Lead Source: {row['Lead_Source']}")
        
        # Previous purchases
        if 'Previous_Purchases' in row and pd.notna(row['Previous_Purchases']):
            parts.append(f"Previous Purchases: {row['Previous_Purchases']}")
        
        # Predictions
        if 'Predicted_Lead_Score' in row and pd.notna(row['Predicted_Lead_Score']):
            score = row['Predicted_Lead_Score']
            parts.append(f"Predicted Lead Score: {score} quality")
            
            # Add confidence if available
            if 'Confidence' in row and pd.notna(row['Confidence']):
                conf_pct = row['Confidence'] * 100
                parts.append(f"Prediction Confidence: {conf_pct:.1f}%")
        
        # Probability scores
        if 'Probability_High' in row and pd.notna(row['Probability_High']):
            prob = row['Probability_High'] * 100
            parts.append(f"Probability of High Quality: {prob:.1f}%")
        
        # Combine into readable text
        text = f"This lead is a {row.get('Company_Size', 'company')} company in the {row.get('Industry', 'unknown')} industry. "
        text += " ".join(parts[2:])  # Skip first two since we already mentioned them
        
        return text
    
    def _generate_summary_stats(self, df: pd.DataFrame) -> List[str]:
        """
        Generate summary statistics as text chunks
        
        Args:
            df: DataFrame with lead data
            
        Returns:
            List of summary text chunks
        """
        summaries = []
        
        # Overall statistics
        total_leads = len(df)
        summaries.append(f"Total leads analyzed: {total_leads}")
        
        # Lead score distribution
        if 'Predicted_Lead_Score' in df.columns:
            score_counts = df['Predicted_Lead_Score'].value_counts()
            for score, count in score_counts.items():
                pct = (count / total_leads) * 100
                summaries.append(f"{count} leads ({pct:.1f}%) are predicted as {score} quality")
        
        # Industry breakdown
        if 'Industry' in df.columns:
            industry_counts = df['Industry'].value_counts().head(5)
            top_industries = ", ".join([f"{ind} ({count})" for ind, count in industry_counts.items()])
            summaries.append(f"Top industries: {top_industries}")
        
        # Revenue insights
        if 'Annual_Revenue' in df.columns:
            avg_revenue = df['Annual_Revenue'].mean()
            max_revenue = df['Annual_Revenue'].max()
            summaries.append(f"Average annual revenue: ${avg_revenue:,.0f}, Maximum: ${max_revenue:,.0f}")
        
        # Engagement insights
        if 'Engagement_Score' in df.columns:
            avg_engagement = df['Engagement_Score'].mean()
            summaries.append(f"Average engagement score: {avg_engagement:.1f}/100")
        
        # Region breakdown
        if 'Region' in df.columns:
            region_counts = df['Region'].value_counts()
            for region, count in region_counts.items():
                pct = (count / total_leads) * 100
                summaries.append(f"{pct:.1f}% of leads are from {region}")
        
        return summaries
    
    def build_context(self, df: pd.DataFrame, force_rebuild: bool = False) -> bool:
        """
        Build context from dataframe and store in ChromaDB
        
        Args:
            df: DataFrame with lead data and predictions
            force_rebuild: If True, rebuild even if data hasn't changed
            
        Returns:
            True if context was built/updated, False if skipped
        """
        # Check if we need to rebuild
        current_hash = self._generate_data_hash(df)
        hash_file = self.persist_directory / "data_hash.txt"
        
        if not force_rebuild and hash_file.exists():
            stored_hash = hash_file.read_text().strip()
            if stored_hash == current_hash:
                print("Data unchanged, skipping context rebuild")
                return False
        
        print("Building context from lead data...")
        
        # Clear existing collection
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "B2B Lead Scoring and ABM data"}
            )
        except:
            pass
        
        # Generate lead-level chunks
        documents = []
        metadatas = []
        ids = []
        
        for idx, row in df.iterrows():
            text = self._convert_lead_to_text(row)
            documents.append(text)
            
            # Create metadata
            metadata = {
                "type": "lead",
                "index": str(idx)
            }
            if 'Industry' in row and pd.notna(row['Industry']):
                metadata["industry"] = str(row['Industry'])
            if 'Company_Size' in row and pd.notna(row['Company_Size']):
                metadata["company_size"] = str(row['Company_Size'])
            if 'Predicted_Lead_Score' in row and pd.notna(row['Predicted_Lead_Score']):
                metadata["lead_score"] = str(row['Predicted_Lead_Score'])
            
            metadatas.append(metadata)
            ids.append(f"lead_{idx}")
        
        # Generate summary chunks
        summaries = self._generate_summary_stats(df)
        for idx, summary in enumerate(summaries):
            documents.append(summary)
            metadatas.append({"type": "summary", "index": str(idx)})
            ids.append(f"summary_{idx}")
        
        # Add to collection
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Added {len(documents)} chunks to vector store")
        
        # Save hash
        hash_file.write_text(current_hash)
        
        return True
    
    def get_collection(self):
        """Get the ChromaDB collection"""
        return self.collection


if __name__ == "__main__":
    # Test the context builder
    print("Testing Context Builder...")
    builder = ContextBuilder()
    
    # Load sample data
    df = pd.read_csv("../demo_leads.csv")
    df['Predicted_Lead_Score'] = 'Medium'  # Mock prediction
    
    builder.build_context(df, force_rebuild=True)
    print("Context built successfully!")

