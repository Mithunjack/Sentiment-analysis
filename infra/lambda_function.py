"""
lambda_function.py - AWS Lambda handler for sentiment analysis
"""
import json
import os
import pickle
import boto3
import logging
from io import BytesIO

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# S3 client
s3 = boto3.client('s3')

# Get environment variables
MODEL_BUCKET = os.environ.get('MODEL_BUCKET')
MODEL_KEY = os.environ.get('MODEL_KEY', 'sentiment_model.pkl')
VECTORIZER_KEY = os.environ.get('VECTORIZER_KEY', 'vectorizer.pkl')

# Global variables to cache the model between invocations
model = None
vectorizer = None

def load_model_from_s3():
    """Load model and vectorizer from S3"""
    global model, vectorizer
    
    try:
        logger.info(f"Loading model from s3://{MODEL_BUCKET}/{MODEL_KEY}")
        
        # Download model from S3
        model_obj = s3.get_object(Bucket=MODEL_BUCKET, Key=MODEL_KEY)
        model_bytes = model_obj['Body'].read()
        model = pickle.loads(model_bytes)
        
        # Download vectorizer from S3
        vectorizer_obj = s3.get_object(Bucket=MODEL_BUCKET, Key=VECTORIZER_KEY)
        vectorizer_bytes = vectorizer_obj['Body'].read()
        vectorizer = pickle.loads(vectorizer_bytes)
        
        logger.info("Model and vectorizer loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def predict_sentiment(text):
    """Predict sentiment of text"""
    global model, vectorizer
    
    # Load model if not already loaded
    if model is None or vectorizer is None:
        success = load_model_from_s3()
        if not success:
            return {"error": "Failed to load model"}, 500
    
    try:
        # Preprocess text (simple for demo)
        processed_text = text.lower()
        
        # Vectorize text
        text_vec = vectorizer.transform([processed_text])
        
        # Get prediction probabilities
        proba = model.predict_proba(text_vec)[0]
        
        # Get the predicted class
        pred_class = model.predict(text_vec)[0]
        
        # Get confidence (probability of predicted class)
        confidence = float(proba[list(model.classes_).index(pred_class)])
        
        return {"sentiment": pred_class, "confidence": confidence}, 200
    except Exception as e:
        logger.error(f"Error predicting sentiment: {str(e)}")
        return {"error": str(e)}, 500

def lambda_handler(event, context):
    """AWS Lambda handler"""
    logger.info(f"Received event: {json.dumps(event)}")
    
    # Parse the incoming request
    try:
        if 'body' in event:
            # API Gateway request
            body = json.loads(event['body'])
            text = body.get('text', '')
            
            if 'reviews' in body:
                # Batch processing
                reviews = body['reviews']
                results = []
                for review in reviews:
                    result, _ = predict_sentiment(review)
                    if 'error' in result:
                        return {
                            'statusCode': 500,
                            'body': json.dumps({"error": "Error processing batch"})
                        }
                    results.append({
                        'text': review,
                        'sentiment': result['sentiment'],
                        'confidence': result['confidence']
                    })
                
                return {
                    'statusCode': 200,
                    'body': json.dumps({"results": results})
                }
            else:
                # Single text processing
                result, status_code = predict_sentiment(text)
                
                if status_code != 200:
                    return {
                        'statusCode': status_code,
                        'body': json.dumps(result)
                    }
                
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'text': text,
                        'sentiment': result['sentiment'],
                        'confidence': result['confidence']
                    })
                }
        else:
            # Direct Lambda invocation
            text = event.get('text', '')
            result, status_code = predict_sentiment(text)
            
            if status_code != 200:
                return result
            
            return {
                'text': text,
                'sentiment': result['sentiment'],
                'confidence': result['confidence']
            }
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({"error": str(e)})
        }