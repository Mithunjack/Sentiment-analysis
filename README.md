# Sentiment Analysis API for AWS

A complete ML project demonstrating sentiment analysis for product reviews with AWS deployment.

## Project Overview

This project builds a sentiment analysis API that:

- Analyzes product reviews and classifies them as positive, negative, or neutral
- Provides confidence scores with each prediction
- Supports both single and batch analysis via REST API
- Deploys to AWS using best practices for ML applications

## Quick Start

### Local Development

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-aws.git
   cd sentiment-analysis-aws
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Train the model:

   ```bash
   python model/train.py
   ```

5. Run the FastAPI application:

   ```bash
   uvicorn app.main:app --reload
   ```

6. Open in your browser: http://127.0.0.1:8000/docs

### Docker

1. Build the Docker image:

   ```bash
   docker build -t sentiment-analysis-api .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 sentiment-analysis-api
   ```

## AWS Deployment

### Prerequisites

1. Install AWS CLI and configure credentials:

   ```bash
   aws configure
   ```

2. Create S3 bucket for model storage (or use CloudFormation)

### Deployment Steps

1. Create AWS resources with CloudFormation:

   ```bash
   aws cloudformation create-stack \
     --stack-name sentiment-analysis-api-dev \
     --template-body file://infra/cloudformation.yaml \
     --capabilities CAPABILITY_IAM
   ```

2. Deploy using GitHub Actions:
   - Add AWS credentials to your GitHub repository secrets
   - Push to main branch to trigger deployment

## Project Structure

- **app/**: FastAPI application
- **data/**: Sample data for testing
- **model/**: Model training and evaluation scripts
- **infra/**: AWS infrastructure as code
- **tests/**: Unit and integration tests
- **Dockerfile**: Container definition
- **.github/workflows/**: CI/CD pipelines

## API Endpoints

### Health Check

```
GET /health
```

### Analyze Single Review

```
POST /analyze
{
  "text": "This product is amazing!"
}
```

### Analyze Multiple Reviews

```
POST /analyze/batch
{
  "reviews": [
    "This product is amazing!",
    "The quality is terrible."
  ]
}
```

## AWS Architecture

The application is deployed using the following AWS services:

- **ECR**: Container registry for Docker images
- **ECS Fargate**: Serverless container runtime
- **Application Load Balancer**: For distribution of traffic
- **S3**: Model storage
- **CloudWatch**: Logging and monitoring
- **CloudFormation**: Infrastructure as code

## Next Steps

- Add authentication to the API
- Implement model versioning and A/B testing
- Create a monitoring dashboard for model metrics
- Implement model retraining pipeline

## License

MIT
