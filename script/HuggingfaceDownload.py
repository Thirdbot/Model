from huggingface_hub import snapshot_download
from slugify import slugify

from configs import load_config

folders_dict = load_config("Paths")


def download_hf_model(model_repo_id,local_dir):
    model_specific_name = slugify(model_repo_id,replacements=[["/","-"]]) # turn / into -
    model_specific_path = local_dir.joinpath(model_specific_name)
    model_specific_path.mkdir(parents=True, exist_ok=True)
    print(f"downloading model {model_repo_id} to {local_dir}")
    return snapshot_download(
        repo_id=model_repo_id,
        local_dir=model_specific_path,

    )