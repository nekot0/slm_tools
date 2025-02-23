from huggingface_hub import snapshot_download

model_name = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
local_model_dir = "/data/models/Llama-3.1-Swallow-8B-Instruct-v0.3"

snapshot_download(repo_id=model_name, local_dir=local_model_dir)