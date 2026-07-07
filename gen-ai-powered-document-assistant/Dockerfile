# Use Python 3.11 as the base image
FROM python:3.11-slim

# Create a working directory inside the container
WORKDIR /app

# Copy the backend requirements file
COPY backend/requirements.txt .

# Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Open port 8000
EXPOSE 8000

# Start the FastAPI server
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]