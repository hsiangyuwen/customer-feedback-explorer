from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
import shutil
import os
from contextlib import asynccontextmanager

from . import services, models, database

# --- Import Google API specific exceptions ---
# The google.genai SDK provides google.genai.errors.APIError for handling API-specific errors.
try:
    from google.genai import errors as google_genai_errors
except ImportError:
    google_genai_errors = None
    print(
        "Warning: google-genai not installed. Specific Google API error handling might be limited."
    )


# Define an async context manager for application lifespan events
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup: Initializing database...")
    await database.init_db()
    print("Database initialization complete.")
    yield
    print("Application shutdown.")


app = FastAPI(
    title="AI-Enhanced Customer Feedback Explorer API",
    description="API for ingesting, vectorizing, and querying customer feedback.",
    version="0.1.0",
    lifespan=lifespan,
)

origins = [
    "http://localhost",  # Keep if you have other services or direct access on this port
    "http://localhost:3000",  # Common port for React development
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)


@app.post("/ingest/", status_code=202)
async def ingest_feedback_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(database.get_session),
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only CSV files are accepted."
        )
    temp_file_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        background_tasks.add_task(services.process_csv_and_store, db, temp_file_path)
        return {
            "message": "CSV file received and processing started in the background."
        }
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
    finally:
        await file.close()


@app.post("/query/", response_model=models.QueryResponse)
async def query_feedback(
    query_request: models.QueryRequest,
    db: AsyncSession = Depends(database.get_session),
):
    print(f"Received query: '{query_request.query}' with top_k={query_request.top_k}")

    if google_genai_errors is None:
        print(
            "Error: google_genai_errors module not imported. Cannot proceed with Google API calls."
        )
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: Google AI SDK components missing or failed to load.",
        )

    if not query_request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # 1. Find similar feedback (uses Gemini embeddings via services.py)
        similar_feedback_records = await services.find_similar_feedback(
            db, query_request.query, query_request.top_k
        )

        if not similar_feedback_records:
            return models.QueryResponse(
                relevant_feedback=[],
                summary="No relevant feedback found for your query.",
            )

        # 2. Prepare texts for LLM summary
        feedback_texts_for_summary = [
            record.text for record in similar_feedback_records
        ]

        # 3. Get LLM summary (uses Gemini LLM via services.py)
        summary = await services.get_llm_summary(
            query_request.query, feedback_texts_for_summary
        )

        # Check if the summary indicates an error from the LLM service itself
        if (
            "LLM prompt was blocked" in summary
            or "LLM content generation was stopped or blocked" in summary
            or "Error generating summary from Gemini LLM" in summary
        ):
            # You might want to return a different HTTP status or a more structured error
            print(f"Warning: LLM summary indicated an issue: {summary}")
            # For now, we pass the error message as the summary, but you could make this a 503 or similar
            # raise HTTPException(status_code=503, detail=f"LLM service issue: {summary}")

        return models.QueryResponse(
            relevant_feedback=similar_feedback_records, summary=summary
        )

    # --- Gemini Specific Error Handling ---
    except google_genai_errors.APIError as e:
        print(
            f"Google GenAI APIError (Code: {e.code if hasattr(e, 'code') else 'N/A'}): {e.message if hasattr(e, 'message') else str(e)}"
        )
        status_code = 503  # Default to Service Unavailable
        detail_message = f"An error occurred with the Google AI service: {e.message if hasattr(e, 'message') else str(e)}"
        if hasattr(e, "code"):
            if e.code == 401 or e.code == 403:  # Unauthorized / Forbidden
                status_code = 401
                detail_message = f"Google API authentication/authorization failed. Please check API key and permissions. (Code: {e.code})"
            elif e.code == 429:  # Too Many Requests / Resource Exhausted
                status_code = 429
                detail_message = f"Google API rate limit exceeded. Please try again later. (Code: {e.code})"
            elif e.code == 400:  # Bad Request / Invalid Argument
                status_code = 400
                detail_message = (
                    f"Invalid argument provided to Google API. (Code: {e.code})"
                )
        raise HTTPException(status_code=status_code, detail=detail_message)
    # --- End Gemini Specific Error Handling ---

    except Exception as e:  # Generic catch-all for other unexpected errors
        print(f"Unexpected error in /query endpoint (Type: {type(e)}): {e}")
        import traceback

        traceback.print_exc()
        # Check if the error message itself is from a failed LLM call, if services.get_llm_summary returns error strings
        if isinstance(e, HTTPException):  # If it's already an HTTPException, re-raise
            raise
        raise HTTPException(
            status_code=500, detail=f"An internal server error occurred: {str(e)}"
        )


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to the AI-Enhanced Customer Feedback Explorer API!"}
