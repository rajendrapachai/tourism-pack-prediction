
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- FIX: Create a non-root user and set permissions for caching ---
RUN useradd -m appuser
RUN chown -R appuser:appuser /app
# Set HF_HOME environment variable to a writable directory within the app's WORKDIR
ENV HF_HOME /app/.hf_cache
RUN mkdir -p ${HF_HOME}
RUN chown -R appuser:appuser ${HF_HOME}
USER appuser
# -----------------------------------------------------------------

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run app.py when the container launches
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501"]
