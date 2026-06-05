import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from script.FolderManager import  manager
from script.HuggingfaceDownload import solve_dataset, solve_model


if __name__ == "__main__":

    # create folders
    manager()
    # download model and dataset
    model_solver, loaded_model = solve_model("unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
                                             load_in_n_bit=16,
                                             unsloth_mode=True )
    dataset_solver, dataset = solve_dataset(
        "intro/flickr8k",
    )

    print("model source:", model_solver.source)
    print("dataset:", dataset)

    model_solver.status_report()
    dataset_solver.status_report()

    if dataset_solver.needs_conversion:
        raise RuntimeError(dataset_solver.conversion_reason)

    model,tokenizer = loaded_model[:2]
    # model, tokenizer = load_unsloth_visiontotext_model(select_model_path.as_posix())
