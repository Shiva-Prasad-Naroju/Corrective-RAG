# Use official lightweight Python image
FROM python:3.12.8-slim


# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (FAISS, build tools, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (to leverage Docker cache)
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project files
COPY . /app/

# Expose port (for FastAPI, Flask, or Streamlit UI if used)
EXPOSE 8000

# Default command to run the app
CMD ["python", "main.py"]
