# Type-R demo application

This project implements a demo application for the [Type-R project](https://github.com/CyberAgentAILab/Type-R).

The frontend and backend is built using [streamlit](https://docs.streamlit.io/).

## Pre-requisites

This demo assumes the Hugging Face token is properly set beforehand.
Also, the demo requires a GPU to be available on the system.

To launch a demo app locally, run the following.

```bash
uv run streamlit run type_r/demo/__main__.py
```

Or, using docker, run the following to build and launch a sever.

```bash
cd ../..  # Type-R root dir.
docker build \
    --platform linux/amd64 \
    -f packages/type-r-demo/Dockerfile \
    -t app:latest \
    .
docker run -it --rm --platform linux/amd64 -p 8501:8501 -e HF_TOKEN=xxx app:latest
```