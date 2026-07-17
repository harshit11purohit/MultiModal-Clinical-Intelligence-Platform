# Use a slim Python 3.11 image
FROM python:3.11-slim

# Install Tesseract and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install (Your txt file already handles the CPU flag!)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose the port
EXPOSE 8080

# Command to run the app (Targeting the src folder)
CMD ["python", "app.py"]