# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set working directory in the container
WORKDIR /app

# Install system dependencies for OpenCV and other libraries
RUN apt update && \
    apt upgrade && apt-get install ffmpeg libsm6 libxext6 -y

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure directories exist
RUN mkdir -p /app/files /app/photos

# Copy the application code and templates
COPY server.py .
COPY cv_detect.py .
COPY templates/ ./templates/
COPY files/ ./files/


# Expose the port the Flask app runs on
EXPOSE 5000

# Set environment variable to ensure Flask runs in production mode
ENV FLASK_ENV=production

# Command to run the Flask application
CMD ["python", "server.py"]