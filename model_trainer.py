import os
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from pathlib import Path
import streamlit as st
import re

class FakeNewsTrainer:
    """Train and save fake news detection models."""
    
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.vectorizer = None
        self.model = None
        self.pipeline = None
        self.training_history = []
        
    def preprocess_text(self, text):
        """Clean and preprocess text data."""
        if not text or pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def prepare_features(self, texts):
        """Prepare TF-IDF features from texts."""
        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Initialize vectorizer if not exists
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                lowercase=True
            )
            
        return processed_texts
    
    def train_model(self, training_data, model_type='logistic', test_size=0.2):
        """Train the fake news detection model."""
        
        st.info(f"Training {model_type} model with {len(training_data)} samples...")
        
        # Prepare data
        X_text = training_data['text'].tolist()
        y = training_data['label'].tolist()
        
        # Preprocess texts
        X_processed = self.prepare_features(X_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Create and fit vectorizer
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Select model
        if model_type == 'logistic':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'naive_bayes':
            self.model = MultinomialNB(alpha=0.1)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        st.write("Training model...")
        self.model.fit(X_train_tfidf, y_train)
        
        # Evaluate model
        train_pred = self.model.predict(X_train_tfidf)
        test_pred = self.model.predict(X_test_tfidf)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        # Store training info
        training_info = {
            'model_type': model_type,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'feature_count': X_train_tfidf.shape[1]
        }
        
        self.training_history.append(training_info)
        
        # Create pipeline for easy prediction
        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('classifier', self.model)
        ])
        
        st.success(f"Model trained successfully!")
        st.write(f"Training Accuracy: {train_accuracy:.4f}")
        st.write(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Show classification report
        st.text("Classification Report:")
        report = classification_report(y_test, test_pred, target_names=['Fake', 'Real'])
        st.text(report)
        
        return training_info
    
    def save_model(self, model_name="fake_news_model"):
        """Save the trained model and vectorizer."""
        if self.pipeline is None:
            st.error("No trained model to save!")
            return False
        
        model_path = self.model_dir / f"{model_name}.pkl"
        vectorizer_path = self.model_dir / f"{model_name}_vectorizer.pkl"
        history_path = self.model_dir / f"{model_name}_history.pkl"
        
        try:
            # Save the complete pipeline
            joblib.dump(self.pipeline, model_path)
            
            # Save training history
            with open(history_path, 'wb') as f:
                pickle.dump(self.training_history, f)
            
            st.success(f"Model saved successfully!")
            st.write(f"Model file: {model_path}")
            st.write(f"History file: {history_path}")
            
            return True
            
        except Exception as e:
            st.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_name="fake_news_model"):
        """Load a saved model."""
        model_path = self.model_dir / f"{model_name}.pkl"
        history_path = self.model_dir / f"{model_name}_history.pkl"
        
        try:
            if not model_path.exists():
                st.error(f"Model file not found: {model_path}")
                return False
            
            # Load pipeline
            self.pipeline = joblib.load(model_path)
            
            # Extract components
            self.vectorizer = self.pipeline.named_steps['tfidf']
            self.model = self.pipeline.named_steps['classifier']
            
            # Load training history if available
            if history_path.exists():
                with open(history_path, 'rb') as f:
                    self.training_history = pickle.load(f)
            
            st.success(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, text):
        """Predict if a news article is fake or real."""
        if self.pipeline is None:
            st.error("No model loaded for prediction!")
            return None, None
        
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Get prediction
            prediction = self.pipeline.predict([processed_text])[0]
            probabilities = self.pipeline.predict_proba([processed_text])[0]
            
            # Get confidence (max probability)
            confidence = max(probabilities)
            
            # Convert prediction to label
            label = "Real" if prediction == 1 else "Fake"
            
            return label, confidence
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None, None
    
    def get_feature_importance(self, text, top_k=10):
        """Get important features for a prediction."""
        if self.pipeline is None or not hasattr(self.model, 'coef_'):
            return {}
        
        try:
            # Process text and get features
            processed_text = self.preprocess_text(text)
            tfidf_vector = self.vectorizer.transform([processed_text])
            
            # Get feature names and coefficients
            feature_names = self.vectorizer.get_feature_names_out()
            coefficients = self.model.coef_[0]
            
            # Get non-zero features
            feature_indices = tfidf_vector.nonzero()[1]
            
            # Create importance dictionary
            importance_dict = {}
            for idx in feature_indices:
                feature_name = feature_names[idx]
                importance = coefficients[idx]
                importance_dict[feature_name] = importance
            
            # Sort by absolute importance
            sorted_features = sorted(
                importance_dict.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:top_k]
            
            return dict(sorted_features)
            
        except Exception as e:
            st.error(f"Error getting feature importance: {str(e)}")
            return {}
    
    def list_saved_models(self):
        """List all saved models."""
        model_files = list(self.model_dir.glob("*.pkl"))
        model_names = []
        
        for file_path in model_files:
            if not file_path.name.endswith(('_vectorizer.pkl', '_history.pkl')):
                model_name = file_path.stem
                model_names.append(model_name)
        
        return model_names
    
    def get_model_info(self):
        """Get information about the current model."""
        if not self.training_history:
            return "No training history available"
        
        latest_info = self.training_history[-1]
        return {
            'Model Type': latest_info.get('model_type', 'Unknown'),
            'Training Accuracy': f"{latest_info.get('train_accuracy', 0):.4f}",
            'Test Accuracy': f"{latest_info.get('test_accuracy', 0):.4f}",
            'Training Samples': latest_info.get('train_size', 0),
            'Test Samples': latest_info.get('test_size', 0),
            'Features': latest_info.get('feature_count', 0)
        }