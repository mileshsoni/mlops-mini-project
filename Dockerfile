# stage-1 : Build Stage
FROM python:3.11 AS build

WORKDIR /app

# copy the requirements.txt file from the flask_app folder
COPY flask_app/requirements.txt /app/

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy the application code and model files
COPY flask_app/ /app/
COPY models/vectorizer.pkl /app/models/vectorizer.pkl

# download only the necessary NLTK data
RUN python -m nltk.downloader stopwords wordnet

# stage - 2: Final Stage
FROM python:3.11-slim AS final

WORKDIR /application

# copy only the necessary files from the build stage
COPY --from=build /app /app

# expose the application port
EXPOSE 5000

# set the command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
