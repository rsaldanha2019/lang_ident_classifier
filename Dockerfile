FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    openmpi-bin \
    curl \
    ca-certificates \
    gnupg \
    python3.10 \
    python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Create symlinks for python and pip
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/local/bin/pip /usr/bin/pip

# Add NVIDIA package repositories (to get nvidia-utils)
RUN curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | apt-key add - && \
    curl -s -L https://nvidia.github.io/libnvidia-container/ubuntu22.04/libnvidia-container.list \
        -o /etc/apt/sources.list.d/nvidia-container.list && \
    apt-get update && apt-get install -y nvidia-utils-535 && \
    rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"

# Set NVIDIA runtime environment
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Avoid OpenBLAS and other multi-threading issues
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install your wheel and remove it after install
# Copy wheel file
COPY dist/ ./dist/

# Install wheel and delete it
RUN pip install ./dist/lang_ident_classifier-*.whl && rm -rf ./dist

# Final sanity checks
RUN python --version && pip --version && mpiexec --version
