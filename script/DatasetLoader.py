from pathlib import Path

from datasets import load_dataset

"""
1. It needs to load the dataset from generated csv
2. It needs to use prompt-template from config, or manually input
3. It needs to output as a formatted dataset.
"""


def load_hf_dataset(local_path):
    local_path = Path(local_path)
    cache_dir = local_path.parents[2] / ".cache" / "huggingface" / "datasets"

    if local_path.is_file():
        files = [local_path]
    else:
        files = list(local_path.rglob("*"))

    json_files = [str(file) for file in files if file.suffix in {".json", ".jsonl"}]
    if json_files:
        return load_dataset("json", data_files=json_files, cache_dir=str(cache_dir))

    csv_files = [str(file) for file in files if file.suffix == ".csv"]
    if csv_files:
        return load_dataset("csv", data_files=csv_files, cache_dir=str(cache_dir))

    parquet_files = [str(file) for file in files if file.suffix == ".parquet"]
    if parquet_files:
        return load_dataset("parquet", data_files=parquet_files, cache_dir=str(cache_dir))

    raise ValueError(f"Cannot detect a supported dataset file in {local_path}")
