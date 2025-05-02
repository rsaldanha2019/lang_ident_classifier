FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

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

# Set python/pip aliases
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/local/bin/pip /usr/bin/pip

# Set CUDA environment vars (inherited from base but explicit for clarity)
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"

# Avoid OpenBLAS and other multi-threading issues
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# Workdir
WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install your package wheel
COPY dist/ ./dist/
RUN pip install ./dist/lang_ident_classifier-*.whl && rm -rf ./dist

# Final check
RUN python --version && pip --version && mpiexec --version
