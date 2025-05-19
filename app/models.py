from pydantic import BaseModel
from typing import List, Optional, Any


class FeedbackRecord(BaseModel):
    id: Optional[int] = None
    original_id: Optional[str] = None  # To store original ID from CSV if any
    text: str
    # You can add more fields from your CSV here if needed
    # e.g., overall_rating: Optional[float] = None


class FeedbackCreate(FeedbackRecord):
    pass


class FeedbackInDB(FeedbackRecord):
    embedding: Optional[List[float]] = None  # Stored as vector in pgvector

    class Config:
        from_attributes = True


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5  # Number of similar documents to retrieve


class QueryResponse(BaseModel):
    relevant_feedback: List[FeedbackInDB]
    summary: str
