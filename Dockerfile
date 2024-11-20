# Use the official Ubuntu base image
FROM ubuntu:22.04

# Set the container name
LABEL name="SaVer_Toolbox"

# Set the environment variables for some packages: 
ENV TZ=America/New_York \
    DEBIAN_FRONTEND=noninteractive


# Update the package list and install Python3 and pip
RUN apt-get update && apt-get install -y --no-install-recommends build-essential python3 python3-pip python3-dev libblas-dev liblapack-dev pkg-config libopenblas-dev libatlas-base-dev libhdf5-dev
RUN pip3 install --upgrade meson ninja

# Set the working directory in the container
WORKDIR /SaVer_Toolbox

# Copy the current directory contents into the container at /app
COPY ./ ./

# Install the SaVer_Toolbox: 
RUN pip3 install .

WORKDIR /