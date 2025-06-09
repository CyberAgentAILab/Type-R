FROM ubuntu:24.04

RUN apt update && apt -y install wget

# When using an NVIDIA Docker image like nvidia/cuda:12.6.0-runtime-ubuntu24.04, unhold the CUDA packages to prevent them from being updated, instead of installing cuda-keyring, by running the following command:
# apt-mark unhold $(apt-mark showhold)

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb


# Install python
RUN DEBIAN_FRONTEND=noninteractive && \
    apt update && apt -y install \
    git \
    g++ \
    curl \
    libfontconfig1 \
    libglib2.0-0 \
    cuda-nvcc-12-6 \
    cuda-profiler-api-12-6 \
    libcusparse-dev-12-6 \
    libcublas-dev-12-6 \
    libcusolver-dev-12-6 \
    python3-dev \
    libgl1 \
    libgl1-mesa-dev

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin/:$PATH"

# Set the working directory
WORKDIR /workdir
COPY . /workdir

# Build the runtime, with CUDA target to A100, L4, H100/H200.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"
ENV CC="gcc"
ENV CXX="g++"

RUN ./script/apply_patch.sh && uv sync

CMD ["/bin/sh", "-c", "sleep INF"]