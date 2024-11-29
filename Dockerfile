# Use the official Python 3.12 base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --upgrade setuptools
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fastapi[standard]

# Copy the rest of the application code into the container
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Define environment variables
ENV TMP_DIR=/app/tmp

# Create the temporary directory
RUN mkdir -p $TMP_DIR

# Set the command to run the FastAPI server
CMD ["fastapi", "run", "--port", "8000"]