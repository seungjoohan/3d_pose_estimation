FROM python:3.10

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create directories for uploads and temp files
RUN mkdir -p /app/uploads /app/temp

# Environment variables
ENV PORT=8080
ENV FLASK_ENV=production
ENV GOOGLE_CLOUD_PROJECT=pose-app

# Make port 8080 available
EXPOSE 8080

# Run the application with Gunicorn (production WSGI server)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app