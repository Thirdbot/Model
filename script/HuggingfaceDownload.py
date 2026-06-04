from huggingface_hub import snapshot_download
from slugify import slugify
from transformers import Qwen3VLModel, AutoProcessor, AutoModel

"""
It needs to handle the download of the model and dataset from huggingface hub
"""

def download_setup(model_repo_id,local_dir):
    model_specific_path = local_dir.joinpath(model_repo_id) # use model_repo_id as path
    model_specific_path.mkdir(parents=True, exist_ok=True)
    print(f"downloading model {model_repo_id} to {model_specific_path}")
    return model_specific_path

def download_QwenVL_model(model_repo_id,local_dir,cache_dir):
    model_specific_path = download_setup(model_repo_id, local_dir)

    model = Qwen3VLModel.from_pretrained(model_repo_id,cache_dir=cache_dir)
    processor = AutoProcessor.from_pretrained(model_repo_id,cache_dir=cache_dir)
    model.save_pretrained(model_specific_path)
    processor.save_pretrained(model_specific_path)
    return True

def download_InternVL_model(model_repo_id,local_dir,cache_dir):
    model_specific_path = download_setup(model_repo_id, local_dir)

    model = AutoModel.from_pretrained(model_repo_id,cache_dir=cache_dir,trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_repo_id,cache_dir=cache_dir,trust_remote_code=True)
    model.save_pretrained(model_specific_path)
    processor.save_pretrained(model_specific_path)
    return True

def download_hf_dataset(dataset_repo_id,local_dir):
    dataset_specific_name = slugify(dataset_repo_id,replacements=[["/","-"]])
    dataset_specific_path = local_dir.joinpath(dataset_specific_name)
    dataset_specific_path.mkdir(parents=True, exist_ok=True)
    print(f"downloading dataset {dataset_repo_id} to {local_dir}")
    return snapshot_download(
        repo_id=dataset_repo_id,
        local_dir=dataset_specific_path,
        repo_type="dataset"
    )