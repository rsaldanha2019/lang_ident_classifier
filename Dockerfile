# Dockerfile
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and python3.10, curl, gosu for switching users
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    openmpi-bin \
    curl \
    ca-certificates \
    gnupg \
    python3.10 \
    python3.10-distutils \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# Install pip for python3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Setup python/pip aliases
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/local/bin/pip /usr/bin/pip

# CUDA environment variables (explicit)
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"

# Avoid multithreading issues
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# Create an app directory for your code
WORKDIR /app

# Copy your requirements.txt and install dependencies
COPY requirements.txt .
# For ensuring bitsandbyte work
ENV PIP_PREFER_BINARY=1
RUN pip install --no-cache-dir -r requirements.txt

# Copy your package wheel and install it
COPY dist/ ./dist/
RUN pip install ./dist/lang_ident_classifier-*.whl && rm -rf ./dist

# # Copy the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Entrypoint will switch user dynamically, default command runs bash (can be overridden)
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
