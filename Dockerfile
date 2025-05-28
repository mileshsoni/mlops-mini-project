# Stage 1: Build stage
FROM python:3.11 AS build

WORKDIR /app

# Copy and install requirements
COPY flask_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and model
COPY flask_app/ /app/
COPY models/vectorizer.pkl /app/models/vectorizer.pkl

# Download NLTK data
RUN python -m nltk.downloader stopwords wordnet


# Stage 2: Final minimal image
FROM python:3.11-slim AS final

WORKDIR /app

# Install dependencies again in final image
COPY flask_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and models
COPY flask_app/ /app/
COPY models/vectorizer.pkl /app/models/vectorizer.pkl

# Download NLTK data again
RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

# Start app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
