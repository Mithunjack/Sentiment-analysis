from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from app.sentiment import SentimentAnalyzer
import time
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Product Review Sentiment Analysis API",
    description="API for analyzing sentiment in product reviews",
    version="1.0.0"
)

# Load the sentiment analyzer
sentiment_analyzer = SentimentAnalyzer()

# Define request and response models
class ReviewRequest(BaseModel):
    text: str
    
    class Config:
        schema_extra = {
            "example": {
                "text": "This product exceeded my expectations. Highly recommended!"
            }
        }

class ReviewResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    processing_time: float
    
    class Config:
        schema_extra = {
            "example": {
                "text": "This product exceeded my expectations. Highly recommended!",
                "sentiment": "positive",
                "confidence": 0.92,
                "processing_time": 0.023
            }
        }

class BatchReviewRequest(BaseModel):
    reviews: list[str]
    
    class Config:
        schema_extra = {
            "example": {
                "reviews": [
                    "This product exceeded my expectations. Highly recommended!",
                    "The quality is poor and it broke after a week.",
                    "It's okay, but not worth the price."
                ]
            }
        }

class BatchReviewResponse(BaseModel):
    results: list[dict]
    processing_time: float

@app.post("/analyze", response_model=ReviewResponse)
async def analyze_sentiment(review: ReviewRequest):
    """Analyze sentiment of a single product review"""
    start_time = time.time()
    
    try:
        sentiment, confidence = sentiment_analyzer.predict(review.text)
        processing_time = time.time() - start_time
        
        return {
            "text": review.text,
            "sentiment": sentiment,
            "confidence": confidence,
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/batch", response_model=BatchReviewResponse)
async def analyze_batch(batch: BatchReviewRequest):
    """Analyze sentiment of multiple product reviews"""
    start_time = time.time()
    
    try:
        results = []
        for text in batch.reviews:
            sentiment, confidence = sentiment_analyzer.predict(text)
            results.append({
                "text": text,
                "sentiment": sentiment,
                "confidence": confidence
            })
        
        processing_time = time.time() - start_time
        
        return {
            "results": results,
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Error analyzing batch sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": sentiment_analyzer.is_model_loaded()}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)