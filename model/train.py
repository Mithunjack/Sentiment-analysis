import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(filepath):
    """Load review data from CSV file"""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    
    # Expected columns: 'review_text', 'rating'
    # Convert ratings to sentiment labels
    df['sentiment'] = df['rating'].apply(
        lambda x: 'positive' if x > 3 else 'negative' if x < 3 else 'neutral'
    )
    
    return df

def preprocess_text(df):
    """Basic text preprocessing"""
    logger.info("Preprocessing text data")
    # You can add more preprocessing steps here
    df['clean_text'] = df['review_text'].str.lower()
    return df

def train_model(X_train, y_train):
    """Train a simple TF-IDF + Logistic Regression model"""
    logger.info("Training model")
    
    # Create and fit the vectorizer
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_vec = vectorizer.fit_transform(X_train)
    
    # Train the classifier
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train_vec, y_train)
    
    return model, vectorizer

def evaluate_model(model, vectorizer, X_test, y_test):
    """Evaluate model performance"""
    logger.info("Evaluating model")
    
    # Transform test data using the same vectorizer
    X_test_vec = vectorizer.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_vec)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    logger.info(f"Model accuracy: {accuracy:.4f}")
    logger.info(f"Classification report:\n{report}")
    
    return accuracy, report

def save_model(model, vectorizer, model_dir='../model'):
    """Save the trained model and vectorizer"""
    import os
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model and vectorizer
    with open(f"{model_dir}/sentiment_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    with open(f"{model_dir}/vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)
    
    logger.info(f"Model and vectorizer saved to {model_dir}")

def main():
    """Main training function"""
    # You can adjust the filepath as needed
    filepath = "../data/sample_reviews.csv"
    
    # Load and preprocess data
    df = load_data(filepath)
    df = preprocess_text(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], 
        df['sentiment'],
        test_size=0.2, 
        random_state=42
    )
    
    # Train model
    model, vectorizer = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, vectorizer, X_test, y_test)
    
    # Save model
    save_model(model, vectorizer)

if __name__ == "__main__":
    main()