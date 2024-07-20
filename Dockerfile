# Use an official Python runtime as a parent image
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Copy the environment.yml file
COPY environment.yml /app/

# Create the environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "nlp-env", "/bin/bash", "-c"]

# Copy the current directory contents into the container at /app
COPY . /app

# Expose the port the app runs on
EXPOSE 8000

# Run the Django server
CMD ["conda", "run", "--no-capture-output", "-n", "nlp-env", "gunicorn", "--chdir", "core", "--bind", ":8000", "core.wsgi:application"]
