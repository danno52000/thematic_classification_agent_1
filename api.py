from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from pydantic import BaseModel
import os
import openai

# Initialize FastAPI with Branding
app = FastAPI(
    title="IDX Thematic Classification API",
    description="An agentic API for company thematic classification",
    version="1.0.0",
    terms_of_service="https://www.idxinsights.com/terms",
    contact={
        "name": "IDX Insights, LLC",
        "url": "https://www.idxinsights.com",
        "email": "support@idxinsights.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

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

# Custom Swagger UI
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>IDX Thematic Classification API</title>
        <link rel="stylesheet" type="text/css" href="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.15.5/swagger-ui.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.15.5/swagger-ui-bundle.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.15.5/swagger-ui-standalone-preset.js"></script>
        <style>
            .topbar-wrapper img {{
                content: url("https://idxinsights.com/wp-content/uploads/2025/03/IDX_thematic_agent_logo.png");
                width: 180px;
                height: auto;
            }}
        </style>
    </head>
    <body> 
        <div id="swagger-ui"></div>
        <script>
            const ui = SwaggerUIBundle({{
                url: "/openapi.json",
                dom_id: '#swagger-ui',
                presets: [SwaggerUIBundle.presets.apis, SwaggerUIStandalonePreset],
                layout: "BaseLayout",
                deepLinking: true,
                syntaxHighlight: {{ theme: "monokai" }},
                showExtensions: true,
                showCommonExtensions: true
            }});
            window.ui = ui;
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Custom ReDoc UI
@app.get("/redoc", include_in_schema=False)
async def redoc_ui():
    return get_redoc_html(
        title="IDX Thematic Classification API Documentation",
        openapi_url="/openapi.json",
        redoc_favicon_url="https://idxinsights.com/wp-content/uploads/2025/03/IDX_thematic_agent_logo.png",
    )
