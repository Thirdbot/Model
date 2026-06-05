import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

HF_CACHE = ROOT / ".cache" / "huggingface"
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE / "hub")
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE / "datasets")
os.environ["HF_MODULES_CACHE"] = str(HF_CACHE / "modules")

from script.FolderManager import  manager,load_config
from script.HuggingfaceDownload import solve_dataset, solve_model



if __name__ == "__main__":

    settings = load_config('paths')
    root = Path(settings.root).resolve()

    # create folders
    manager()
    # download model and dataset
    model_solver, loaded_model = solve_model("unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
                                             cache_dir=root / settings.dirs.hub_cache,
                                             load_in_n_bit=16,
                                             unsloth_mode=True )
    dataset_solver, dataset_path = solve_dataset(
        "intro/flickr8k",
        cache_dir=root / settings.dirs.hub_cache,
    )

    print("model snapshot:", model_solver.snapshot_path)
    print("dataset snapshot:", dataset_path)

    model_solver.status_report()
    dataset_solver.status_report()

    if dataset_solver.needs_conversion:
        raise RuntimeError(dataset_solver.conversion_reason)

    model,tokenizer = loaded_model[:2]
    # model, tokenizer = load_unsloth_visiontotext_model(select_model_path.as_posix())
