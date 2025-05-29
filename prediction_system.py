import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
from model_trainer import FakeNewsTrainer
from data_processor import analyze_sentiment, analyze_bias
from pathlib import Path
import json

class PredictionSystem:
    """System for making predictions and managing prediction history."""
    
    def __init__(self, model_dir="models", predictions_dir="predictions"):
        self.trainer = FakeNewsTrainer(model_dir)
        self.predictions_dir = Path(predictions_dir)
        self.predictions_dir.mkdir(exist_ok=True)
        
        self.current_model = None
        self.prediction_history = []
        self.load_prediction_history()
    
    def load_model(self, model_name):
        """Load a trained model for predictions."""
        success = self.trainer.load_model(model_name)
        if success:
            self.current_model = model_name
            st.success(f"Model '{model_name}' loaded successfully")
            return True
        return False
    
    def predict_news(self, text, save_prediction=True):
        """Predict if news is fake or real and return detailed results."""
        if self.trainer.pipeline is None:
            st.error("No model loaded. Please load a model first.")
            return None
        
        # Get basic prediction
        label, confidence = self.trainer.predict(text)
        
        if label is None:
            return None
        
        # Get additional analysis
        sentiment_data = analyze_sentiment(text)
        bias_data = analyze_bias(text)
        feature_importance = self.trainer.get_feature_importance(text)
        
        # Create prediction result
        prediction_result = {
            'id': len(self.prediction_history) + 1,
            'timestamp': datetime.now().isoformat(),
            'text': text[:500] + "..." if len(text) > 500 else text,  # Truncate for storage
            'full_text_length': len(text),
            'prediction': label,
            'confidence': float(confidence),
            'model_used': self.current_model,
            'sentiment': {
                'polarity': float(sentiment_data['polarity']),
                'subjectivity': float(sentiment_data['subjectivity']),
                'label': sentiment_data['sentiment_label']
            },
            'bias': {
                'political_bias': float(bias_data['political_bias']),
                'emotional_intensity': float(bias_data['emotional_intensity']),
                'tone': bias_data['tone']
            },
            'feature_importance': feature_importance
        }
        
        # Save prediction if requested
        if save_prediction:
            self.prediction_history.append(prediction_result)
            self.save_prediction_history()
        
        return prediction_result
    
    def bulk_predict(self, texts, progress_bar=None):
        """Make predictions on multiple texts."""
        if self.trainer.pipeline is None:
            st.error("No model loaded. Please load a model first.")
            return []
        
        results = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            if progress_bar:
                progress_bar.progress((i + 1) / total)
            
            result = self.predict_news(text, save_prediction=False)
            if result:
                results.append(result)
        
        # Save all predictions at once
        self.prediction_history.extend(results)
        self.save_prediction_history()
        
        return results
    
    def save_prediction_history(self):
        """Save prediction history to file."""
        history_file = self.predictions_dir / "prediction_history.json"
        try:
            with open(history_file, 'w') as f:
                json.dump(self.prediction_history, f, indent=2, default=str)
        except Exception as e:
            st.error(f"Error saving prediction history: {str(e)}")
    
    def load_prediction_history(self):
        """Load prediction history from file."""
        history_file = self.predictions_dir / "prediction_history.json"
        try:
            if history_file.exists():
                with open(history_file, 'r') as f:
                    self.prediction_history = json.load(f)
        except Exception as e:
            st.warning(f"Could not load prediction history: {str(e)}")
            self.prediction_history = []
    
    def get_prediction_stats(self):
        """Get statistics about predictions."""
        if not self.prediction_history:
            return {
                'total_predictions': 0,
                'fake_count': 0,
                'real_count': 0,
                'avg_confidence': 0,
                'models_used': []
            }
        
        total = len(self.prediction_history)
        fake_count = sum(1 for p in self.prediction_history if p['prediction'] == 'Fake')
        real_count = total - fake_count
        avg_confidence = np.mean([p['confidence'] for p in self.prediction_history])
        models_used = list(set(p.get('model_used', 'Unknown') for p in self.prediction_history))
        
        return {
            'total_predictions': total,
            'fake_count': fake_count,
            'real_count': real_count,
            'fake_percentage': (fake_count / total) * 100 if total > 0 else 0,
            'real_percentage': (real_count / total) * 100 if total > 0 else 0,
            'avg_confidence': avg_confidence,
            'models_used': models_used
        }
    
    def get_recent_predictions(self, limit=10):
        """Get recent predictions."""
        return self.prediction_history[-limit:] if self.prediction_history else []
    
    def export_predictions(self, format='csv'):
        """Export predictions to file."""
        if not self.prediction_history:
            st.warning("No predictions to export")
            return None
        
        # Convert to DataFrame
        df_data = []
        for pred in self.prediction_history:
            row = {
                'id': pred['id'],
                'timestamp': pred['timestamp'],
                'text_preview': pred['text'],
                'prediction': pred['prediction'],
                'confidence': pred['confidence'],
                'model_used': pred.get('model_used', 'Unknown'),
                'sentiment_polarity': pred['sentiment']['polarity'],
                'sentiment_subjectivity': pred['sentiment']['subjectivity'],
                'sentiment_label': pred['sentiment']['label'],
                'political_bias': pred['bias']['political_bias'],
                'emotional_intensity': pred['bias']['emotional_intensity'],
                'tone': pred['bias']['tone']
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        if format == 'csv':
            export_file = self.predictions_dir / f"predictions_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(export_file, index=False)
            return export_file
        elif format == 'json':
            export_file = self.predictions_dir / f"predictions_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            df.to_json(export_file, indent=2)
            return export_file
        
        return None
    
    def clear_history(self):
        """Clear prediction history."""
        self.prediction_history = []
        self.save_prediction_history()
        st.success("Prediction history cleared")
    
    def delete_prediction(self, prediction_id):
        """Delete a specific prediction."""
        self.prediction_history = [p for p in self.prediction_history if p['id'] != prediction_id]
        self.save_prediction_history()
    
    def search_predictions(self, query, field='text'):
        """Search predictions by text content or other fields."""
        if not query:
            return self.prediction_history
        
        query = query.lower()
        results = []
        
        for pred in self.prediction_history:
            if field == 'text' and query in pred['text'].lower():
                results.append(pred)
            elif field == 'prediction' and query in pred['prediction'].lower():
                results.append(pred)
            elif field == 'model' and query in pred.get('model_used', '').lower():
                results.append(pred)
        
        return results