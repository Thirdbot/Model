from pathlib import Path
import csv
import os
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
from datasets import Image, load_dataset

class DataSolver:
    """
    It needs to solve a dataset problem like loading must come from a csv file,
    and it needs to compatible with model-types by applying prompt-template from those models
    All this is pointing to huggingface dataset and it compatibility.
    """

    def __init__(self, repo_id_or_dataset_path):
        self.repo_id_or_dataset_path = str(repo_id_or_dataset_path)
        self.source = self.repo_id_or_dataset_path
        self.dataset = None
        self.is_missing_path = False
        self.come_from_path = self._is_path()
        self.is_repo = not self.come_from_path and not self.is_missing_path

        # status
        self.modality = None  # vision / lang
        self.dataset_format = None
        self.needs_conversion = False
        self.conversion_reason = None
        self.modality_reason = None
        self.csv_files = []
        self.csv_columns = []
        self.detected_file_traits = []
        self.detected_column_traits = []

        if self.come_from_path or self.is_missing_path:
            self._modality_solver()

    def status_report(self):
        print(f"""
        Currently, loading {self.source}
        Loaded with native datasets loader: {self.dataset is not None}
        Dataset format: {self.dataset_format}
        Modality: {self.modality}
        Modality reason: {self.modality_reason}
        Needs conversion: {self.needs_conversion}
        Conversion reason: {self.conversion_reason}
        CSV files: {self.csv_files}
        CSV columns: {self.csv_columns}
        Detected file traits: {self.detected_file_traits}
        Detected column traits: {self.detected_column_traits}
        """)

    def _modality_solver(self):
        files = self._file_names()
        self.csv_files = [file for file in files if file.lower().endswith(".csv")]
        self.detected_file_traits = self._infer_file_traits(files)

        if not self.csv_files:
            self.dataset_format = "unsupported"
            self.modality = "unknown"
            self.modality_reason = "No CSV file found, so modality is not decided."
            self.needs_conversion = True
            self.conversion_reason = "No CSV file found. Convert dataset files to a CSV manifest first."
            return self.modality

        self.dataset_format = "csv"
        self.needs_conversion = False
        self.conversion_reason = None

        self.csv_columns = self._read_csv_columns()
        self.detected_column_traits = self._infer_column_traits(self.csv_columns)
        self.modality = self._infer_modality_from_file_traits()
        return self.modality

    def _infer_modality_from_file_traits(self):
        if any(trait in self.detected_file_traits for trait in ("image_files", "audio_files", "video_files", "archives")):
            self.modality_reason = "CSV plus non-text asset files/archive detected at file level."
            return "vision"

        if self.detected_file_traits == ["csv"]:
            self.modality_reason = "Only CSV files detected at file level."
            return "lang"

        self.modality_reason = "CSV exists, but file-level traits are mixed or unclear. Columns are report-only."
        return "unknown"

    def _infer_file_traits(self, files):
        suffixes = {Path(file).suffix.lower() for file in files}
        traits = []

        if ".csv" in suffixes:
            traits.append("csv")
        if suffixes & {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
            traits.append("image_files")
        if suffixes & {".wav", ".mp3", ".flac", ".ogg"}:
            traits.append("audio_files")
        if suffixes & {".mp4", ".avi", ".mov", ".webm", ".mkv"}:
            traits.append("video_files")
        if suffixes & {".json", ".jsonl"}:
            traits.append("json_files")
        if ".parquet" in suffixes:
            traits.append("parquet_files")
        if suffixes & {".zip", ".tar", ".gz"}:
            traits.append("archives")

        return traits

    def _file_names(self):
        if self.come_from_path:
            path = Path(self.source)
            if path.is_file():
                return [path.name]
            return [file.name for file in path.rglob("*") if file.is_file()]

        return []

    def _read_csv_columns(self):
        if not self.come_from_path:
            return []

        path = Path(self.source)
        csv_path = path if path.is_file() and path.suffix == ".csv" else None
        if csv_path is None:
            csv_paths = list(path.rglob("*.csv"))
            if not csv_paths:
                return []
            csv_path = csv_paths[0]

        with csv_path.open(newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            return next(reader, [])

    def _infer_column_traits(self, columns):
        normalized = {column.strip().lower() for column in columns}

        image_columns = {"image", "image_path", "img", "path", "file_name", "filename"}
        text_columns = {"text", "prompt", "instruction", "question", "caption", "answer", "response", "output"}
        label_columns = {"label", "labels", "class", "category"}

        has_image = bool(normalized & image_columns)
        has_text = bool(normalized & text_columns)
        has_label = bool(normalized & label_columns)

        traits = []
        if has_image:
            traits.append("image_columns")
        if has_text:
            traits.append("text_columns")
        if has_label:
            traits.append("label_columns")
        if {"prompt", "response"} <= normalized or {"instruction", "response"} <= normalized:
            traits.append("instruction_columns")

        return traits

    def solve(self):
        if self.is_repo:
            self.dataset = self._load_repo_dataset()
            self._modality_solver_from_dataset()
            return self._dataset_solver()

        if self.come_from_path:
            self.dataset = self._load_local_dataset()
            if self.dataset_format == "csv":
                self._modality_solver()
            else:
                self._modality_solver_from_dataset()

        return self._dataset_solver()

    def _load_repo_dataset(self):
        try:
            return load_dataset(
                self.repo_id_or_dataset_path,
                token=hf_token,
            )
        except GatedRepoError as error:
            raise PermissionError("The repo_id needs permission before it can be loaded.") from error
        except RepositoryNotFoundError as error:
            raise FileNotFoundError(f"Dataset repo not found: {self.repo_id_or_dataset_path}") from error
        except HfHubHTTPError as error:
            if error.response is not None and error.response.status_code == 429:
                raise RuntimeError(
                    "Hugging Face rate limited this dataset load. Wait a few minutes, then retry; "
                    "the cache will resume already-downloaded files where possible."
                ) from error
            raise

    def _load_local_dataset(self):
        path = Path(self.source)
        if self.dataset_format == "csv":
            data_files = str(path) if path.is_file() else [str(file) for file in path.rglob("*.csv")]
            return load_dataset("csv", data_files=data_files)
        if path.is_dir():
            return load_dataset("imagefolder", data_dir=str(path))
        raise RuntimeError(self.conversion_reason or f"Cannot load local dataset path: {self.source}")

    def _modality_solver_from_dataset(self):
        split = self._first_split()
        if split is None:
            self.dataset_format = "unknown"
            self.modality = "unknown"
            self.modality_reason = "Loaded dataset has no readable splits."
            self.needs_conversion = True
            self.conversion_reason = "Dataset loaded, but no split could be inspected."
            return self.modality

        self.csv_columns = list(getattr(split, "column_names", []) or [])
        self.detected_column_traits = self._infer_column_traits(self.csv_columns)
        self.dataset_format = "native"
        self.needs_conversion = False
        self.conversion_reason = None

        features = getattr(split, "features", {}) or {}
        has_image_feature = any(isinstance(feature, Image) for feature in features.values())
        if has_image_feature:
            self.modality = "vision"
            self.modality_reason = "Native dataset has an Image feature."
        elif "image_columns" in self.detected_column_traits:
            self.modality = "vision"
            self.modality_reason = "Native dataset has image-like columns."
        elif "text_columns" in self.detected_column_traits or "instruction_columns" in self.detected_column_traits:
            self.modality = "lang"
            self.modality_reason = "Native dataset has text/instruction columns."
        else:
            self.modality = "unknown"
            self.modality_reason = "Native dataset loaded, but columns do not clearly show modality."

        return self.modality

    def _first_split(self):
        if self.dataset is None:
            return None
        if hasattr(self.dataset, "keys"):
            for split_name in self.dataset.keys():
                return self.dataset[split_name]
        return self.dataset

    def _dataset_solver(self):
        """
        Return the loaded dataset when available; otherwise return the local source path.
        :return:
        """
        return self.dataset or self.source

    def _is_path(self):
        path = Path(self.source).expanduser()
        if path.exists():
            self.source = str(path)
            return True

        if path.is_absolute() or self.source.startswith(("./", "../", "~")):
            self.is_missing_path = True

        return False

if __name__ == "__main__":
    raise SystemExit("Use DataSolver(repo_id_or_dataset_path).solve()")
