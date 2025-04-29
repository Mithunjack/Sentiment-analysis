import pickle
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Class for sentiment analysis using a trained model"""
    
    def __init__(self, model_path=None, vectorizer_path=None):
        """
        Initialize the sentiment analyzer
        
        Args:
            model_path: Path to the saved model file
            vectorizer_path: Path to the saved vectorizer file
        """
        # Set default paths if not provided
        self.model_path = model_path or os.environ.get(
            'MODEL_PATH', 
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model/sentiment_model.pkl')
        )
        
        self.vectorizer_path = vectorizer_path or os.environ.get(
            'VECTORIZER_PATH',
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model/vectorizer.pkl')
        )
        
        # Load model and vectorizer
        self.model = None
        self.vectorizer = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and vectorizer"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
                
            logger.info("Model and vectorizer loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Load fallback model or raise error in production
            return False
    
    def is_model_loaded(self):
        """Check if model is loaded"""
        return self.model is not None and self.vectorizer is not None
    
    def preprocess_text(self, text):
        """Preprocess text for prediction"""
        # Basic preprocessing - add more steps as needed
        return text.lower()
    
    def predict(self, text):
        """
        Predict sentiment of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (sentiment, confidence)
        """
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Vectorize text
        text_vec = self.vectorizer.transform([processed_text])
        
        # Get prediction probabilities
        proba = self.model.predict_proba(text_vec)[0]
        
        # Get the predicted class
        pred_class = self.model.predict(text_vec)[0]
        
        # Get confidence (probability of predicted class)
        confidence = float(proba[list(self.model.classes_).index(pred_class)])
        
        return pred_class, confidence
    
    def batch_predict(self, texts):
        """
        Predict sentiment for multiple texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of (sentiment, confidence) tuples
        """
        return [self.predict(text) for text in texts]