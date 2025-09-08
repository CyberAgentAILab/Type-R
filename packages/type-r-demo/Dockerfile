FROM ghcr.io/astral-sh/uv:bookworm-slim

ENV STREAMLIT_SERVER_PORT=8501

# Install system dependencies
RUN DEBIAN_FRONTEND=noninteractive apt update \
    && apt -y install libfontconfig1 libglib2.0-0 libgomp1 libgl1 libegl1 \
    && rm -rf /var/lib/apt/lists/*

# Install application
ADD . /app
WORKDIR /app/packages/type-r-demo
RUN uv sync

# Start the Streamlit server
CMD ["uv", "run", "streamlit", "run", "type_r/demo/__main__.py"]