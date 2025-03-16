from fastapi import FastAPI
from pydantic import BaseModel
import os
import openai

# Initialize FastAPI
app = FastAPI()

# OpenAI API Setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API Key. Ensure OPENAI_API_KEY is set as an environment variable.")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Define Request Model
class ClassificationRequest(BaseModel):
    ticker: str
    filing_date: str

# API Endpoint for Classification
@app.post("/classify")
def classify_10k(request: ClassificationRequest):
    """
    API endpoint to classify 10-K filings based on thematic categories.
    Users provide a ticker and filing date, and it returns classified results.
    """

    # Simulate classification logic (Replace this with actual LLM processing)
    prompt = f"Classify the 10-K filing for {request.ticker} on {request.filing_date} based on thematic categories."
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        classification_results = response.choices[0].message.content.strip()

        return {
            "ticker": request.ticker,
            "filing_date": request.filing_date,
            "classification": classification_results,
        }
    
    except Exception as e:
        return {"error": str(e)}

# Root Endpoint to Confirm API is Running
@app.get("/")
def home():
    return {"message": "Thematic Classification API is running!"}
