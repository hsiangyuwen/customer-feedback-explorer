import os
from google import genai
from google.genai import types
import pandas as pd
from typing import List, Union, cast
from tenacity import retry, stop_after_attempt, wait_random_exponential
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text as sql_text
from .models import FeedbackInDB
from .database import FeedbackDB
import traceback
import asyncio

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# --- Google AI Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError(
        "CRITICAL: GOOGLE_API_KEY environment variable not set or is empty."
    )
client = genai.Client(api_key=GOOGLE_API_KEY)
# --- End Google AI Configuration ---

FEEDBACK_COLUMN_NAME = os.getenv("FEEDBACK_COLUMN_NAME", "reviewText")

# --- Model Configuration ---
# Gemini Models
GEMINI_EMBEDDING_MODEL_NAME = os.getenv(
    "GEMINI_EMBEDDING_MODEL", "models/embedding-001"
)

# The output dimensionality for "models/embedding-001" is 768.
# If you use "models/text-embedding-004", it's also 768.
# This is CRITICAL for pgvector and your DB schema.
# You MUST update your FeedbackDB embedding column and any dimension constants.
EMBEDDING_DIMENSIONS_GEMINI = 768  # For embedding-001 and text-embedding-004

GEMINI_CHAT_MODEL_NAME = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash-latest")
# --- End Model Configuration ---


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
async def embed_single_text(
    text: str,
    model_name: str = GEMINI_EMBEDDING_MODEL_NAME,
    task_type: str = "RETRIEVAL_QUERY",
) -> List[float]:
    """
    Embeds a single text string into a vector using the specified model and task type.
    This is for finding similar feedback.

    Args:
        text (str): The text to embed.
        model_name (str, optional): The model to use for embedding. Defaults to GEMINI_EMBEDDING_MODEL_NAME.
        task_type (str, optional): The task type for embedding. Defaults to "RETRIEVAL_DOCUMENT".

    Returns:
        List[float]: The embedding vector.
    """

    if not text or text.isspace():
        print(
            f"Warning: Empty or whitespace-only text provided to embed_single_text. Returning zero vector."
        )
        return [0.0] * EMBEDDING_DIMENSIONS_GEMINI

    print(
        f"DEBUG Gemini embed: Attempting single text embedding, model '{model_name}', task '{task_type}'."
    )

    try:
        response = await asyncio.to_thread(
            client.models.embed_content,
            model=model_name,
            contents=[text],  # API expects a list, even for a single item
            config=types.EmbedContentConfig(task_type=task_type),
        )
        # response.embeddings is a list of ContentEmbedding objects
        if not response.embeddings or not response.embeddings[0]:
            print(
                f"Warning: Gemini embedding returned no embedding for single text input. Returning zero vector."
            )
            return [0.0] * EMBEDDING_DIMENSIONS_GEMINI

        # Extract the 'values' attribute from the ContentEmbedding object
        embedding_obj = response.embeddings[0]
        if not hasattr(embedding_obj, "values"):
            raise ValueError("ContentEmbedding object missing 'values' attribute.")

        embedding: List[float] = embedding_obj.values

        if (
            not isinstance(embedding, list)
            or len(embedding) != EMBEDDING_DIMENSIONS_GEMINI
        ):
            raise ValueError(
                f"Unexpected structure or dimension for Gemini embedding. Expected List[float_dim_{EMBEDDING_DIMENSIONS_GEMINI}]."
            )
        return embedding
    except Exception as e:
        print(f"Error during Gemini single text embedding (Type: {type(e)}): {e}")
        traceback.print_exc()
        return [0.0] * EMBEDDING_DIMENSIONS_GEMINI


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
async def embed_texts_batch(
    texts: List[str],
    model_name: str = GEMINI_EMBEDDING_MODEL_NAME,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> List[List[float]]:
    """
    Embeds a list of text strings into vectors using the specified model and task type.
    This is for processing the input csv file and saving embeddings to the database.

    Args:
        texts (List[str]): The list of texts to embed.
        model_name (str, optional): The model to use for embedding. Defaults to GEMINI_EMBEDDING_MODEL_NAME.
        task_type (str, optional): The task type for embedding. Defaults to "RETRIEVAL_DOCUMENT".

    Returns:
        List[List[float]]: The list of embedding vectors.
    """
    processed_texts = [t for t in texts if t and not t.isspace()]
    if not processed_texts:
        print(
            f"Warning: Empty list or list with only empty/whitespace texts provided to embed_texts_batch. Returning empty list."
        )
        return []

    print(
        f"DEBUG Gemini embed: Attempting batch embeddings for {len(processed_texts)} text(s), model '{model_name}', task '{task_type}'."
    )

    try:
        response = await asyncio.to_thread(
            client.models.embed_content,
            model=model_name,
            contents=processed_texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        # response.embeddings is a list of ContentEmbedding objects
        embedding_objects: List[types.ContentEmbedding] = response.embeddings

        if not embedding_objects:
            print(
                f"Warning: Gemini embedding returned no embeddings for batch input. Returning list of zero vectors."
            )
            return [[0.0] * EMBEDDING_DIMENSIONS_GEMINI for _ in processed_texts]

        # Extract the 'values' attribute from each ContentEmbedding object
        all_embeddings: List[List[float]] = []
        for emb_obj in embedding_objects:
            if not hasattr(emb_obj, "values"):
                # Handle case where an individual embedding object might be malformed
                print(
                    f"Warning: ContentEmbedding object missing 'values' attribute in batch. Using zero vector for this item."
                )
                all_embeddings.append([0.0] * EMBEDDING_DIMENSIONS_GEMINI)
                continue
            all_embeddings.append(emb_obj.values)

        if (
            len(all_embeddings)
            != len(processed_texts)  # Check if number of embeddings matches input texts
            or not all(
                isinstance(emb, list) and len(emb) == EMBEDDING_DIMENSIONS_GEMINI
                for emb in all_embeddings
            )
        ):
            # More detailed error for debugging structure
            first_emb_details = "N/A"
            if all_embeddings:
                first_emb = all_embeddings[0]
                first_emb_details = f"type: {type(first_emb)}, len: {len(first_emb) if isinstance(first_emb, list) else 'N/A'}"

            raise ValueError(
                f"Unexpected structure, dimension, or count for Gemini batch embeddings. Expected {len(processed_texts)} embeddings of List[float_dim_{EMBEDDING_DIMENSIONS_GEMINI}]. Got {len(all_embeddings)} embeddings. First embedding details: {first_emb_details}."
            )
        return all_embeddings
    except Exception as e:
        print(f"Error during Gemini batch embedding (Type: {type(e)}): {e}")
        traceback.print_exc()
        return [
            [0.0] * EMBEDDING_DIMENSIONS_GEMINI for _ in processed_texts
        ]  # Fallback to zero vectors for the batch size


async def find_similar_feedback(
    db: AsyncSession, query_text: str, top_k: int = 5
) -> List[FeedbackInDB]:
    """
    Finds similar feedback based on a query text.

    Args:
        db (AsyncSession): The database session.
        query_text (str): The text to search for similar feedback.
        top_k (int, optional): The number of similar feedback to return. Defaults to 5.

    Returns:
        List[FeedbackInDB]: A list of similar feedback.
    """
    # Get query embedding
    query_embedding_list_float: List[float] = await embed_single_text(
        query_text,
        task_type="RETRIEVAL_QUERY",
    )
    # Validate the embedding
    if (
        not query_embedding_list_float
        or not isinstance(query_embedding_list_float, list)
        or len(query_embedding_list_float) != EMBEDDING_DIMENSIONS_GEMINI
        or all(v == 0.0 for v in query_embedding_list_float)
    ):
        print(
            "Error: Could not generate a valid Gemini embedding for the query or got zero vector."
        )
        return []
    # --- SOLUTION: Convert the list of floats to pgvector string format ---
    # pgvector expects a string like '[0.1,0.2,0.3]'
    query_embedding_str = str(query_embedding_list_float)

    print(
        f"DEBUG find_similar_feedback: Query embedding string for DB: {query_embedding_str[:100]}..."
    )  # Log part of it
    # The SQL query now explicitly casts the string parameter to a vector
    stmt = sql_text(f"""
        SELECT id, original_id, text, embedding
        FROM feedback
        ORDER BY embedding <=> CAST(:query_embedding_param AS vector)
        LIMIT :limit_param
    """)

    try:
        result = await db.execute(
            stmt,
            {
                "query_embedding_param": query_embedding_str,  # Pass the string representation
                "limit_param": top_k,
            },
        )
        similar_items_db = result.fetchall()
    except Exception as e:
        print(f"Database error during similarity search: {e}")
        traceback.print_exc()
        return []

    similar_feedbacks = []
    for row in similar_items_db:
        emb = (
            row.embedding
        )  # pgvector.sqlalchemy should ideally convert this back to a list/array
        if isinstance(emb, str):  # Fallback if it's still a string from DB
            try:
                emb = [float(f) for f in emb.strip("[]").split(",")]
            except ValueError:
                print(
                    f"Warning: Could not parse embedding string from DB for row ID {row.id}. Using zero vector."
                )
                emb = [0.0] * EMBEDDING_DIMENSIONS_GEMINI
        elif emb is None:  # Handle if embedding is NULL in DB
            print(
                f"Warning: Embedding is NULL in DB for row ID {row.id}. Using zero vector."
            )
            emb = [0.0] * EMBEDDING_DIMENSIONS_GEMINI

        similar_feedbacks.append(
            FeedbackInDB(
                id=row.id,
                original_id=row.original_id,
                text=row.text,
                # Ensure embedding is a list of floats for Pydantic model
                embedding=list(emb)
                if isinstance(emb, (list, tuple))
                else ([0.0] * EMBEDDING_DIMENSIONS_GEMINI),
            )
        )
    return similar_feedbacks


async def process_csv_and_store(db: AsyncSession, file_path: str):
    """
    Processes a CSV file containing customer feedback and stores it in the database.

    Args:
        db (AsyncSession): The database session.
        file_path (str): The path to the CSV file.

    Returns:
        dict: A dictionary containing the result of the operation.
    """

    # Read the CSV file
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV {file_path}: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return {"error": f"Error reading CSV: {e}"}
    if FEEDBACK_COLUMN_NAME not in data.columns:
        msg = f"Error: Column '{FEEDBACK_COLUMN_NAME}' not found in CSV."
        print(msg)
        os.remove(file_path)
        return {"error": msg, "available_columns": list(data.columns)}
    # Drop rows with missing feedback text
    data.dropna(subset=[FEEDBACK_COLUMN_NAME], inplace=True)
    # Remove empty strings
    data = data[data[FEEDBACK_COLUMN_NAME].astype(str).str.strip() != ""]
    if data.empty:
        print("No valid feedback text in CSV after cleaning.")
        os.remove(file_path)
        return {"message": "No valid feedback text to process."}
    feedback_texts = data[FEEDBACK_COLUMN_NAME].astype(str).tolist()
    all_embeddings_list = []

    # Gemini `embed_content` can handle lists. Max 100 items per call for `batch_embed_contents`.
    # For `models/embedding-001`, the limit is often around requests per minute, not just batch size.
    batch_size_gemini = 100
    for i in range(0, len(feedback_texts), batch_size_gemini):
        batch_texts = feedback_texts[i : i + batch_size_gemini]
        if not batch_texts:
            continue
        print(
            f"DEBUG process_csv: Processing batch {i // batch_size_gemini + 1}, size: {len(batch_texts)}"
        )
        # get_embedding returns List[List[float]] for list input
        batch_embeddings_result = await embed_texts_batch(batch_texts)
        if (
            batch_embeddings_result
            and isinstance(batch_embeddings_result, list)
            and len(batch_embeddings_result) == len(batch_texts)
            and all(
                isinstance(emb, list) and len(emb) == EMBEDDING_DIMENSIONS_GEMINI
                for emb in batch_embeddings_result
            )
        ):
            all_embeddings_list.extend(batch_embeddings_result)
        else:
            print(
                f"Warning: Embedding for batch starting at index {i} failed or returned malformed/partial results. Using zero vectors."
            )
            all_embeddings_list.extend(
                [[0.0] * EMBEDDING_DIMENSIONS_GEMINI for _ in batch_texts]
            )

        # Add a small delay to respect potential RPM limits for free tier
        await asyncio.sleep(0.5)  # 0.5 second delay between batches. Adjust as needed.
    if len(all_embeddings_list) != len(data):
        msg = f"Critical Error: Mismatch after batch processing. Feedback texts: {len(data)}, Embeddings: {len(all_embeddings_list)}."
        print(msg)
        os.remove(file_path)
        return {"error": "Internal error during embedding generation: count mismatch."}
    records_to_insert = []
    for i in range(len(data)):
        row = data.iloc[i]
        original_id_val = (
            str(row["id"]) if "id" in row and pd.notna(row["id"]) else None
        )
        embedding_vector = all_embeddings_list[i]
        if (
            all(v == 0.0 for v in embedding_vector)
            and len(embedding_vector) == EMBEDDING_DIMENSIONS_GEMINI
        ):
            print(
                f"Warning: Storing zero vector for feedback ID {original_id_val} (Text: '{str(row[FEEDBACK_COLUMN_NAME])[:30]}...')."
            )
        records_to_insert.append(
            FeedbackDB(
                original_id=original_id_val,
                text=str(row[FEEDBACK_COLUMN_NAME]),
                embedding=embedding_vector,  # Store Gemini embedding
            )
        )

    result_message = {"message": "No records to insert."}
    if records_to_insert:
        db.add_all(records_to_insert)
        await db.commit()
        print(
            f"Successfully processed and stored {len(records_to_insert)} feedback records using Gemini embeddings."
        )
        result_message = {
            "message": f"Successfully processed and stored {len(records_to_insert)} feedback records.",
            "count": len(records_to_insert),
        }
    else:
        print("No records were prepared for insertion.")
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"Temp file {file_path} removed.")
        except Exception as e:
            print(f"Error removing temp file {file_path}: {e}")
    return result_message


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
async def get_llm_summary(query: str, relevant_feedback_texts: List[str]) -> str:
    if not relevant_feedback_texts:
        return "No relevant feedback found to summarize."

    context = "\n\n".join(relevant_feedback_texts)
    # Gemini prompt structure can be simpler
    prompt = f"""Please analyze the following customer feedback snippets in relation to the user's query.
Provide a concise and insightful summary or answer.

User Query: "{query}"

Relevant Feedback Snippets:
---
{context}
---

Analysis and Summary:
"""
    print(
        f"DEBUG Gemini LLM: Generating summary with model '{GEMINI_CHAT_MODEL_NAME}'."
    )
    try:
        response = await client.aio.models.generate_content(
            model=GEMINI_CHAT_MODEL_NAME,  # Direct model kwarg
            contents=prompt,  # Direct contents kwarg
        )

        # Accessing the text:
        # Try response.text first as it's often populated by the SDK for simple text responses.
        if response.text:
            return response.text.strip()

        # Fallback to checking candidates if response.text is empty or not available.
        if response.candidates:
            candidate = response.candidates[0]
            # Use the FinishReason enum for comparison for better readability and robustness
            if candidate.finish_reason == types.Candidate.FinishReason.STOP:
                if candidate.content and candidate.content.parts:
                    # Concatenate text from all parts.
                    summary = "".join(
                        part.text
                        for part in candidate.content.parts
                        if hasattr(part, "text")
                    )
                    return summary.strip()
                else:
                    # This case means STOP reason but no content, which is unusual.
                    print(
                        "ERROR Gemini LLM: No content parts in candidate despite STOP reason."
                    )
                    return "LLM generated no content."  # Or handle as an error
            else:  # Other finish reasons (BLOCKED, MAX_TOKENS, SAFETY, RECITATION, OTHER)
                reason_name = (
                    candidate.finish_reason.name
                    if hasattr(candidate.finish_reason, "name")
                    else str(candidate.finish_reason)
                )
                print(
                    f"ERROR Gemini LLM: Generation finished with reason: {reason_name}. Candidate: {candidate}"
                )

                # Construct a more informative error message
                error_message = (
                    f"LLM content generation not completed. Reason: {reason_name}"
                )
                if candidate.safety_ratings:  # Add safety ratings if present
                    ratings_str = ", ".join(
                        [
                            f"{sr.category.name}: {sr.probability.name}"
                            for sr in candidate.safety_ratings
                        ]
                    )
                    error_message += f". Safety Ratings: [{ratings_str}]"
                return error_message
        else:  # No candidates, could be due to prompt blocking or other issues.
            print(
                f"ERROR Gemini LLM: No candidates in response. Full response: {response}"
            )
            # Check response.prompt_feedback for why the prompt might have been blocked.
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason_name = (
                    response.prompt_feedback.block_reason.name
                    if hasattr(response.prompt_feedback.block_reason, "name")
                    else str(response.prompt_feedback.block_reason)
                )
                # block_reason_message might provide more details
                details = (
                    response.prompt_feedback.block_reason_message or block_reason_name
                )
                return f"LLM prompt was blocked. Reason: {details}"
            return "LLM generation failed: No candidates or parsable text returned."

    except Exception as e:
        print(f"Error during Gemini LLM call (Type: {type(e)}): {e}")
        traceback.print_exc()
        return "Error generating summary from Gemini LLM."
