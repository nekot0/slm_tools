# Need to execute "huggingface-cli login" and enter Access Token to get the gemma family.

from huggingface_hub import snapshot_download

model_name = "google/gemma-2-9b-it"
local_model_dir = "/data/models/gemma-2-9b-it"

snapshot_download(repo_id=model_name, local_dir=local_model_dir)