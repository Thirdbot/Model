from huggingface_hub import snapshot_download
from configs import load_config
from pathlib import Path
from script.helper.model_solver import ModelSolver

"""
It needs to handle the download of the model and dataset from huggingface hub
"""

def get_hub_cache_dir(cache_dir=None):
    settings = load_config('paths')
    root = Path(settings.root).resolve()
    return cache_dir or root / settings.dirs.hub_cache


def solve_model(model_repo_id_or_path, cache_dir=None,load_in_n_bit=4,unsloth_mode=True):
    cache_dir = get_hub_cache_dir(cache_dir)
    solver = ModelSolver(model_repo_id_or_path, cache_dir=cache_dir,load_in_n_bit=load_in_n_bit,unsloth_mode=unsloth_mode)
    loaded = solver.solve()
    return solver, loaded

def download_hf_dataset(dataset_repo_id, cache_dir=None):

    cache_dir = get_hub_cache_dir(cache_dir)

    print(f"downloading dataset {dataset_repo_id} to Hugging Face cache")
    return snapshot_download(
        repo_id=dataset_repo_id,
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )
