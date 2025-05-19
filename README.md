# customer-feedback-explorer

This project provides a full-stack solution for ingesting customer feedback (from CSV files), performing semantic search using vector embeddings, and generating AI-powered summaries and insights based on user queries.

## Features

- **Data Ingestion:** Upload CSV files containing customer feedback. The text is processed, and embeddings are generated.
- **Vector Search:** User queries are converted to embeddings to find the most semantically similar feedback entries from the database.
- **AI Summarization:** Relevant feedback is fed to a Large Language Model (LLM) to generate concise summaries and answers to user queries.
- **Web Interface:** A React-based frontend for data upload and an interactive chat interface for querying.

## Architectural Overview

The application consists of three main services, orchestrated using Docker Compose:

1.  **Database Service (`db`):**

    - **Engine:** PostgreSQL 16 with the `pgvector` extension.
    - **Functionality:** Stores customer feedback text and their corresponding vector embeddings. Enables efficient similarity searches using Hierarchical Navigable Small World (HNSW) indexing.

2.  **Backend Service (`backend_service`):**

    - **Framework:** FastAPI (Python 3.9.6)
    - **Functionality:**
      - Provides API endpoints for `/ingest` (data processing) and `/query` (similarity search and LLM summary).
        ```bash
        curl -X POST \
        -F "file=@./dir-path-to-csv-file/amazon_review.csv" \
        http://localhost:8000/ingest/
        ```
        ```bash
        curl -X POST \
        -H "Content-Type: application/json" \
        -d '{"query": "What are common complaints about battery life?", "top_k": 3}' \
        http://localhost:8000/query/
        ```
      - Handles CSV parsing, text cleaning.
      - Integrates with Google Gemini API for:
        - Generating text embeddings (`models/embedding-001`).
        - Generating chat completions/summaries (`gemini-1.5-flash-latest`).
      - Communicates with the PostgreSQL/pgvector database.
    - **Database Interaction:** Uses SQLAlchemy for ORM and raw SQL for pgvector-specific queries.

3.  **Frontend Service (`frontend_react_service`):**
    - **Framework:** React 18 written in TypeScript with Vite for local development and docker build.
    - **Styling:** Tailwind CSS.
    - **Functionality:**
      - Provides a user interface to upload CSV files (calls `/ingest` API).
      - Offers a chat interface for users to type natural language queries (calls `/query` API) and view AI-generated insights and relevant feedback.
    - **Dockerized Serving:** The production build of the React app is served by Nginx for efficiency and proper SPA routing.

**Diagram (Conceptual Flow):**
![Diagram](https://i.imgur.com/8B1qDwt.png)

## Setup Instructions

**Prerequisites:**

- Docker and Docker Compose installed.
- A Google AI API Key (enable "Generative Language API" in your Google Cloud Project).

**Steps:**

1.  **Clone the Repository:**

    ```bash
    git clone <your-repository-url>
    cd customer-feedback-explorer
    ```

2.  **Configure Environment Variables:**

    - Navigate to the `app/` directory.
    - Copy `.env.example` to `.env`: `cp .env.example .env`
    - Edit `app/.env` and add your `GOOGLE_API_KEY`:
      ```env
      GOOGLE_API_KEY="YOUR_GOOGLE_AI_API_KEY"
      ```
    - The frontend (`frontend-react`) uses a runtime configuration for its `VITE_API_BASE_URL` which is set in `docker-compose.yml`. For local Vite development (outside Docker), you can create `frontend-react/.env` from `frontend-react/.env.example` with `VITE_API_BASE_URL=http://localhost:8000` (fastapi default port).

3.  **Build and Run with Docker Compose:**
    From the project root directory (`customer-feedback-explorer/`):

    ```bash
    docker-compose up --build -d
    ```

    - `--build`: Forces a rebuild of the images if Dockerfiles or contexts have changed.
    - `-d`: Runs containers in detached mode.

4.  **Access the Application:**

    - **Frontend:** Open your browser and navigate to `http://localhost:3000`.
    - **Backend API (Directly, if needed):** Accessible at `http://localhost:8000`.

5.  **Stopping the Application:**
    ```bash
    docker-compose down
    ```
    To remove volumes (like database data):
    ```bash
    docker-compose down -v
    ```

**Local Development (without Docker involved):**

- **Backend (FastAPI using uvicorn):**
  ```bash
  # From project root, assuming you have a Python (virtual) environment set up
  pip install -r app/requirements.txt
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
  ```
- **Frontend (Vite dev server):**
  ```bash
  # From project root
  # Ensure `frontend-react/.env` has `VITE_API_BASE_URL=http://localhost:8000` for this mode.
  cd frontend-react
  npm install
  npm run dev # Frontend will be available at http://localhost:3000
  ```

## Key Design Decisions & Assumptions

1.  **Technology Stack:**

    - **Backend:** Python/FastAPI chosen for rapid API development and asynchronous capabilities.
    - **Frontend:** React/TailwindCSS/Vite chosen for a modern, efficient UI development experience.
    - **Database:** PostgreSQL with `pgvector` for robust relational storage combined with powerful vector similarity search. HNSW indexing is used for performance.
    - **Embeddings & LLM:** Google Gemini API (free tier models like `embedding-001` and `gemini-1.5-flash`) selected for cost-effectiveness and ease of integration.
    - **Containerization:** Docker and Docker Compose for consistent development, deployment, and service orchestration.

2.  **API URL Configuration for Frontend:**

    - The Dockerized frontend (served by Nginx) receives its backend API URL (`VITE_API_BASE_URL`) at runtime. This is injected into `index.html` by a shell script (`substitute-env.sh`) when the Nginx container starts, using an environment variable set in `docker-compose.yml`. This allows flexibility in deploying backend and frontend services with different hostnames/ports.
    - For local Vite development, `import.meta.env.VITE_API_BASE_URL` is used, sourced from `frontend-react/.env`.

3.  **Data Ingestion as Background Task:**

    - The `/ingest` API endpoint accepts the file and immediately returns a `202 Accepted` response. The actual CSV processing, embedding generation, and database insertion happen in a background task managed by FastAPI. This prevents long-running HTTP requests and improves user experience.

4.  **Error Handling:**

    - Basic error handling is implemented in both frontend and backend services, returning appropriate HTTP status codes and messages. More granular error handling would be needed for production.

5.  **Embedding Dimensions:**

    - The system is configured for Gemini `embedding-001` model which outputs 768-dimensional vectors. The database schema (`vector(768)`) reflects this. Changing embedding models might require schema updates.

6.  **Nginx for Serving Frontend Build:**

    - When Dockerizing the frontend for a production-like setup, Nginx is used to serve the static build output from Vite. This is a standard practice because:
      - Nginx is highly efficient for serving static files.
      - It correctly handles client-side routing for Single Page Applications (SPAs) like React apps (by always serving `index.html` for unknown paths).
      - It provides a robust, production-ready web server, unlike the Vite development server.
      - It offers capabilities like reverse proxying, load balancing, SSL termination, and security hardening if needed in more advanced setups.

7.  **Assumptions:**
    - Input CSV files have a clearly identifiable column for feedback text (configurable via `FEEDBACK_COLUMN_NAME` in `app/.env`).
    - The user has a valid Google AI API key with necessary permissions.
    - Rate limits for the free tier of Google Gemini API are respected (some basic delays are added in batch processing).

## Suggestions for Production Readiness

To evolve this solution into a production-ready application:

1.  **Robust Error Handling & Centralized Logging:**
    - Implement comprehensive error management and structured logging across all services.
2.  **Security Hardening:**
    - Enforce HTTPS.
    - Implement API authentication/authorization.
    - Add input validation.
3.  **Scalability & Performance Optimization:**
    - Prepare backend for horizontal scaling.
    - Monitor and manage LLM/Embedding API quotas and costs; implement caching where feasible.
    - Utilize a CDN for frontend static assets if needed.
4.  **Configuration & Secret Management:**
    - Employ a secure system for managing environment configurations and secrets (e.g., Vault, Kubernetes Secrets).
5.  **CI/CD Pipeline:**
    - Automate testing, image building, and deployments.
6.  **Comprehensive Testing:**
    - Implement thorough unit, integration, and end-to-end tests.
7.  **Advanced API Gateway/Proxying (with Nginx or dedicated gateway):**
    - Consider proxying all API traffic through a single gateway (like Nginx in front of the backend or a dedicated API gateway) for centralized control.
