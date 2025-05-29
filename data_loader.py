import os
import json
import pandas as pd
import requests
from pathlib import Path
import streamlit as st

class FakeNewsNetLoader:
    """Load and process FakeNewsNet dataset."""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_fakenewsnet_sample(self):
        """Download sample data from FakeNewsNet repository."""
        # GitHub raw URLs for sample data
        urls = {
            'gossipcop_fake': 'https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/gossipcop_fake.csv',
            'gossipcop_real': 'https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/gossipcop_real.csv',
            'politifact_fake': 'https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_fake.csv',
            'politifact_real': 'https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_real.csv'
        }
        
        datasets = {}
        
        for name, url in urls.items():
            try:
                st.write(f"Downloading {name}...")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Save raw data
                file_path = self.data_dir / f"{name}.csv"
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                # Load into DataFrame
                df = pd.read_csv(file_path)
                datasets[name] = df
                st.success(f"Downloaded {name}: {len(df)} articles")
                
            except Exception as e:
                st.error(f"Error downloading {name}: {str(e)}")
                # Create fallback with minimal structure
                datasets[name] = pd.DataFrame(columns=['id', 'news_url', 'title', 'tweet_ids'])
        
        return datasets
    
    def load_or_download_data(self):
        """Load existing data or download if not available."""
        # Check if data already exists
        existing_files = list(self.data_dir.glob("*.csv"))
        
        if len(existing_files) >= 4:
            st.info("Loading existing FakeNewsNet data...")
            datasets = {}
            for file_path in existing_files:
                name = file_path.stem
                try:
                    datasets[name] = pd.read_csv(file_path)
                except Exception as e:
                    st.error(f"Error loading {name}: {str(e)}")
            return datasets
        else:
            st.info("FakeNewsNet data not found locally. Downloading...")
            return self.download_fakenewsnet_sample()
    
    def prepare_training_data(self, datasets):
        """Prepare training data from loaded datasets."""
        training_data = []
        
        for dataset_name, df in datasets.items():
            if df.empty:
                continue
                
            # Determine label from dataset name
            label = 0 if 'fake' in dataset_name else 1  # 0 = fake, 1 = real
            
            for _, row in df.iterrows():
                # Use title as text content (in real implementation, you'd fetch full article text)
                text_content = str(row.get('title', ''))
                
                if text_content and len(text_content) > 10:  # Basic validation
                    training_data.append({
                        'text': text_content,
                        'label': label,
                        'source': dataset_name,
                        'id': row.get('id', ''),
                        'url': row.get('news_url', '')
                    })
        
        training_df = pd.DataFrame(training_data)
        
        # Shuffle the data
        training_df = training_df.sample(frac=1).reset_index(drop=True)
        
        return training_df
    
    def get_article_content(self, url):
        """Fetch full article content from URL (placeholder for real implementation)."""
        # In a real implementation, you would use web scraping to get full article text
        # For now, return placeholder
        return f"Article content from {url}"

class DataValidator:
    """Validate and clean training data."""
    
    @staticmethod
    def validate_training_data(df):
        """Validate the training data quality."""
        validation_results = {
            'total_samples': len(df),
            'fake_samples': len(df[df['label'] == 0]),
            'real_samples': len(df[df['label'] == 1]),
            'empty_texts': len(df[df['text'].str.len() < 10]),
            'duplicate_texts': df['text'].duplicated().sum(),
            'avg_text_length': df['text'].str.len().mean(),
            'min_text_length': df['text'].str.len().min(),
            'max_text_length': df['text'].str.len().max()
        }
        
        return validation_results
    
    @staticmethod
    def clean_training_data(df):
        """Clean the training data."""
        # Remove empty or very short texts
        df = df[df['text'].str.len() >= 10]
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['text'])
        
        # Remove any null values
        df = df.dropna(subset=['text', 'label'])
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df