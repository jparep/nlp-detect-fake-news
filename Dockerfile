# Use the official Miniconda image
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Install system dependencies and clean up to reduce image size
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the environment.yml file to the container
COPY environment.yml /app/

# Create the Conda environment with retry logic
RUN conda config --set channel_priority strict && \
    for i in {1..5}; do conda env create -f environment.yml && break || sleep 5; done

# Activate the environment and set the PATH
SHELL ["conda", "run", "-n", "nlp-env", "/bin/bash", "-c"]

# Ensure the conda environment is activated by default
RUN echo "conda activate nlp-env" >> ~/.bashrc

# Copy the current directory contents into the container at /app
COPY . /app

# Expose port 8000
EXPOSE 8000

# Run the application
CMD ["conda", "run", "--no-capture-output", "-n", "nlp-env", "gunicorn", "--chdir", "core", "--bind", "0.0.0.0:8000", "core.wsgi:application"]
