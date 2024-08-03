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

# Ensure the environment is activated:
RUN echo "conda activate nlp-env" >> ~/.bashrc
ENV PATH /opt/conda/envs/nlp-env/bin:$PATH

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the Django server
CMD ["gunicorn", "--chdir", "core", "--bind", "0.0.0.0:8000", "core.wsgi:application"]
