# Use the NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHON_VERSION=3.8 

# Add the deadsnakes PPA to get Python 3.8
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-dev \
    python3.8-venv \
    python3-distutils \
    wget \
    curl \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install CUDA and cuDNN runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8=8.6.* \
    libcudnn8-dev=8.6.* \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure Python 3.8 is the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --set python /usr/bin/python3.8

# Install pip for Python 3.8
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

# Install the specified Python packages
RUN pip install --ignore-installed nvflare~=2.5.0rc tensorflow[and-cuda] jupyterlab

# Create a workspace directory
WORKDIR /workspace

# Set up JupyterLab as the default command
CMD ["jupyter-lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
