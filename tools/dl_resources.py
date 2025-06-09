from huggingface_hub import snapshot_download

local_path = snapshot_download(
    repo_id="cyberagent/type-r", repo_type="model", local_dir="resources"
)
