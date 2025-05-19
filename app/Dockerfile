# Official Python runtime as a parent image
FROM python:3.9.6-slim-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR .

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY ./app/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application code into the container
COPY ./app /app/

# Expose port 8000 (where Uvicorn will run)
EXPOSE 8000

# Command to run the application using Uvicorn
# The init_db function will be called by FastAPI's lifespan event.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]