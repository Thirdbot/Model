from script.FolderManager import  manager,load_config,load_config_to_path
from pathlib import Path
from script.HuggingfaceDownload import download_hf_model

if __name__ == "__main__":
    root_path = Path(__file__).parent
    config = load_config("Paths")
    model_dir = root_path.joinpath(config.model_path.name, config.model_path.sub_name)
    manager(root_path)
    download_hf_model("mjschock/SmolVLM-Instruct",
                      model_dir)