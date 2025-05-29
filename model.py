import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import streamlit as st

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class FakeNewsClassifier:
    def __init__(self):
        """Initialize the fake news classifier with multiple models."""
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        # Initialize models
        self.models = {
            'logistic': LogisticRegression(random_state=42, max_iter=1000),
            'naive_bayes': MultinomialNB(alpha=0.1),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        self.ensemble_model = None
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Train the model with sample data
        self._train_initial_model()
    
    def preprocess_text(self, text):
        """Preprocess text for analysis."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and stem
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                stemmed_token = self.stemmer.stem(token)
                processed_tokens.append(stemmed_token)
        
        return ' '.join(processed_tokens)
    
    def _generate_sample_data(self):
        """Generate sample training data for the model."""
        # Sample real news patterns
        real_news_samples = [
            "The stock market experienced volatility today as investors reacted to new economic data released by the federal reserve.",
            "Scientists at the university published peer-reviewed research showing promising results in cancer treatment trials.",
            "Local authorities confirmed that the new infrastructure project will begin construction next month after environmental approval.",
            "The weather service issued a warning for severe storms expected to hit the region this weekend.",
            "City council voted unanimously to approve the new budget allocation for public education and healthcare services.",
            "International trade negotiations continued this week with representatives from multiple countries participating in discussions.",
            "The technology company announced quarterly earnings that exceeded analyst expectations due to strong product sales.",
            "Medical experts recommend following established health guidelines during the ongoing public health situation.",
            "The university's research team published findings in a scientific journal after extensive peer review process.",
            "Emergency services responded quickly to the incident and confirmed that safety protocols were followed properly."
        ]
        
        # Sample fake news patterns
        fake_news_samples = [
            "SHOCKING: Scientists REFUSE to tell you this ONE WEIRD TRICK that governments don't want you to know!",
            "BREAKING: Celebrity spotted with aliens in secret government facility - PHOTOS LEAKED!",
            "This MIRACLE cure will eliminate all diseases but Big Pharma is hiding it from the public!",
            "URGENT: New world order plans revealed - they are coming for your freedom next week!",
            "EXPOSED: Everything you know about history is a LIE - see the REAL TRUTH they hide!",
            "ALERT: Government mind control satellites activated - protect yourself with this simple method!",
            "CONSPIRACY: Major news outlets caught fabricating stories to control your thoughts!",
            "SECRET documents reveal that all celebrities are actually robots controlled by shadow government!",
            "DEVASTATING: The real reason why they banned this natural remedy that cures everything!",
            "FORBIDDEN knowledge: Ancient civilization technology that could solve world problems but elites suppress it!"
        ]
        
        # Create training data
        texts = real_news_samples + fake_news_samples
        labels = ['Real'] * len(real_news_samples) + ['Fake'] * len(fake_news_samples)
        
        # Add some variations and more samples
        extended_texts = []
        extended_labels = []
        
        for text, label in zip(texts, labels):
            extended_texts.append(text)
            extended_labels.append(label)
            
            # Create variations
            if label == 'Real':
                # Add more formal variations for real news
                variations = [
                    f"According to official sources, {text.lower()}",
                    f"Reports indicate that {text.lower()}",
                    f"Officials confirmed that {text.lower()}"
                ]
                for var in variations:
                    extended_texts.append(var)
                    extended_labels.append(label)
            else:
                # Add more sensational variations for fake news
                variations = [
                    f"MUST READ: {text}",
                    f"They don't want you to see this: {text}",
                    f"VIRAL: {text}"
                ]
                for var in variations:
                    extended_texts.append(var)
                    extended_labels.append(label)
        
        return extended_texts, extended_labels
    
    def _train_initial_model(self):
        """Train the model with sample data."""
        # Generate sample data
        texts, labels = self._generate_sample_data()
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Create TF-IDF features
        X = self.vectorizer.fit_transform(processed_texts)
        y = np.array(labels)
        
        # Train the primary model (Logistic Regression)
        self.ensemble_model = self.models['logistic']
        self.ensemble_model.fit(X, y)
        
        # Store feature names for explainability
        self.feature_names = self.vectorizer.get_feature_names_out()
    
    def predict(self, text):
        """Predict whether news is real or fake."""
        if not text or len(text.strip()) == 0:
            return "Real", 0.5, {}
        
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        # Transform to TF-IDF
        X = self.vectorizer.transform([processed_text])
        
        # Get prediction and probability
        prediction = self.ensemble_model.predict(X)[0]
        probabilities = self.ensemble_model.predict_proba(X)[0]
        
        # Calculate confidence (distance from 0.5)
        fake_prob = probabilities[0] if self.ensemble_model.classes_[0] == 'Fake' else probabilities[1]
        real_prob = probabilities[1] if self.ensemble_model.classes_[1] == 'Real' else probabilities[0]
        
        confidence = max(fake_prob, real_prob)
        
        # Get feature importance for explainability
        feature_importance = self._get_feature_importance(X)
        
        return prediction, confidence, feature_importance
    
    def _get_feature_importance(self, X, top_k=20):
        """Get the most important features for the prediction."""
        try:
            # Get coefficients from the logistic regression model
            if hasattr(self.ensemble_model, 'coef_'):
                coefficients = self.ensemble_model.coef_[0]
                
                # Get the non-zero features in the input
                feature_indices = X.nonzero()[1]
                
                # Create importance dictionary
                importance_dict = {}
                for idx in feature_indices:
                    feature_name = self.feature_names[idx]
                    importance = coefficients[idx]
                    importance_dict[feature_name] = importance
                
                # Sort by absolute importance and return top k
                sorted_features = sorted(
                    importance_dict.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:top_k]
                
                return dict(sorted_features)
            else:
                return {}
        except Exception as e:
            return {}
    
    def retrain_with_feedback(self, text, true_label):
        """Retrain the model with user feedback."""
        try:
            # This would be implemented in a production system
            # For now, we'll just acknowledge the feedback
            processed_text = self.preprocess_text(text)
            
            # In a real system, you would:
            # 1. Add this to your training data
            # 2. Retrain the model periodically
            # 3. Validate the new model
            
            return True
        except Exception as e:
            return False
    
    def get_model_info(self):
        """Get information about the current model."""
        return {
            'model_type': 'Logistic Regression with TF-IDF',
            'features': len(self.feature_names) if hasattr(self, 'feature_names') else 0,
            'vectorizer_params': self.vectorizer.get_params(),
            'model_params': self.ensemble_model.get_params() if self.ensemble_model else {}
        }
