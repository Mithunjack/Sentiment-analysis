import pytest
from fastapi.testclient import TestClient
import os
import sys
import pickle
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app
from app.sentiment import SentimentAnalyzer

# Create a test client
client = TestClient(app)

# Mock sentiment analyzer for testing
@pytest.fixture
def mock_sentiment_analyzer():
    # Create a mock for the SentimentAnalyzer
    with patch('app.sentiment.SentimentAnalyzer') as mock:
        # Configure the mock to return predefined values
        instance = mock.return_value
        instance.is_model_loaded.return_value = True
        instance.predict.return_value = ("positive", 0.95)
        instance.batch_predict.return_value = [("positive", 0.95), ("negative", 0.85)]
        yield mock

# Test the health check endpoint
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

# Test the sentiment analysis endpoint with a mock
def test_analyze_sentiment(mock_sentiment_analyzer):
    response = client.post(
        "/analyze",
        json={"text": "This product is amazing!"}
    )
    assert response.status_code == 200
    assert response.json()["sentiment"] == "positive"
    assert response.json()["confidence"] > 0.9
    assert "processing_time" in response.json()

# Test batch analysis endpoint with a mock
def test_analyze_batch(mock_sentiment_analyzer):
    response = client.post(
        "/analyze/batch",
        json={"reviews": ["This product is amazing!", "This product is terrible!"]}
    )
    assert response.status_code == 200
    assert len(response.json()["results"]) == 2
    assert response.json()["results"][0]["sentiment"] == "positive"
    assert response.json()["results"][1]["sentiment"] == "negative"
    assert "processing_time" in response.json()

# Test error handling
def test_error_handling(mock_sentiment_analyzer):
    # Configure the mock to raise an exception
    instance = mock_sentiment_analyzer.return_value
    instance.predict.side_effect = Exception("Test error")
    
    response = client.post(
        "/analyze",
        json={"text": "This product is amazing!"}
    )
    assert response.status_code == 500
    assert "detail" in response.json()

# Integration test with a real model (optional, commented out by default)
"""
def test_integration_with_real_model():
    # This test requires a trained model to be available
    # Ensure that the model files exist before running this test
    model_path = os.path.join(os.path.dirname(__file__), '../model/sentiment_model.pkl')
    vectorizer_path = os.path.join(os.path.dirname(__file__), '../model/vectorizer.pkl')
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        analyzer = SentimentAnalyzer(model_path, vectorizer_path)
        sentiment, confidence = analyzer.predict("This product is amazing!")
        assert sentiment in ["positive", "negative", "neutral"]
        assert 0 <= confidence <= 1
"""

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])