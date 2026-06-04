from script.FolderManager import  manager,load_config
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent
HF_CACHE = ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE / "datasets"))
os.environ.setdefault("HF_MODULES_CACHE", str(HF_CACHE / "modules"))

from script.HuggingfaceDownload import download_hf_dataset, download_InternVL_model
from script.DatasetLoader import load_hf_dataset
from script.ModelLoader import load_unsloth_visiontotext_model, load_hf_visiontotext_model



if __name__ == "__main__":
    root_path = Path(__file__).parent
    config = load_config("Paths")
    model_dir = root_path.joinpath(config.model_path.name, config.model_path.sub_name)
    dataset_dir = root_path.joinpath(config.data_path.name, config.data_path.sub_name)
    manager(root_path)
    download_InternVL_model("OpenGVLab/InternVL3-1B",
                      local_dir=model_dir,cache_dir=HF_CACHE)
    download_hf_dataset(
        "AdaptLLM/remote-sensing-visual-instructions",
        dataset_dir
    )
    select_dataset_path = list(dataset_dir.iterdir())[0]
    select_model_path = model_dir.joinpath("OpenGVLab/InternVL3-1B")
    print("select path:",select_model_path.as_posix())
    load_hf_dataset(select_dataset_path.as_posix())
    model,tokenizer = load_hf_visiontotext_model(select_model_path.as_posix())
    # model, tokenizer = load_unsloth_visiontotext_model(select_model_path.as_posix())