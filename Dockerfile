# Use the official Python image from the Docker Hub
FROM python:3.10

# # Update pip to the latest version
RUN pip install --upgrade pip

# Set the working directory in the container
WORKDIR /SaVer_Toolbox

# Copy the current directory contents into the container at /app
COPY ./ /SaVer_Toolbox

RUN pip3 install -e .