# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY ./src /app/src
COPY ./app /app/app
COPY ./models /app/models
COPY ./data /app/data
COPY ./config.yaml /app/config.yaml

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable to tell Streamlit not to open a browser
ENV STREAMLIT_SERVER_HEADLESS=true

# Run streamlit_app.py when the container launches
# Use the healthcheck to ensure the app is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]