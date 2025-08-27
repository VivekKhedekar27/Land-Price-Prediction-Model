# Use an official Python runtime as the base image
FROM python:3.9-slim

# Install necessary system dependencies including libgomp
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && apt-get clean

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model files and application code into the container
COPY property_price_lgb.pkl training_references_2.pkl /app/
COPY app.py /app/

# Expose port 8000
EXPOSE 8000

# Start the Flask app
CMD ["python", "app.py"]
