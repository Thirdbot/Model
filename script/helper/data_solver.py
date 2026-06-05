from pathlib import Path
import csv

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
import os
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

class DataSolver:
    """
    It needs to solve a dataset problem like loading must come from a csv file,
    and it needs to compatible with model-types by applying prompt-template from those models
    All this is pointing to huggingface dataset and it compatibility.
    """

    def __init__(self, repo_id_or_dataset_path, cache_dir=None, load_in_n_bit=4, unsloth_mode=True):
        self.repo_id_or_dataset_path = str(repo_id_or_dataset_path)
        self.source = self.repo_id_or_dataset_path
        self.cache_dir = str(cache_dir) if cache_dir else None
        self.snapshot_path = None
        self.is_existed = False
        self.is_missing_path = False
        self.come_from_path = self._is_path()
        self.is_repo = not self.come_from_path and not self.is_missing_path
        self.need_permission = False
        self.need_download = self.is_repo
        self.all_files = None

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
        Currently, loading {self.source} from {self.cache_dir}
        Snapshot path: {self.snapshot_path}
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

        if not self.all_files:
            return []
        return [file.rfilename for file in self.all_files]

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
        if self.need_permission:
            raise PermissionError("The repo_id needs permission before it can be downloaded.")

        if self.is_repo and self.need_download:
            self._download_dataset()
            self.come_from_path = True
            self.all_files = None
            self._modality_solver()

        return self._dataset_solver()

    def _download_dataset(self):
        try:
            self.snapshot_path = snapshot_download(
                repo_id=self.repo_id_or_dataset_path,
                repo_type="dataset",
                cache_dir=self.cache_dir,
                token=hf_token,
            )
        except GatedRepoError as error:
            self.need_permission = True
            raise PermissionError("The repo_id needs permission before it can be downloaded.") from error
        except RepositoryNotFoundError as error:
            raise FileNotFoundError(f"Dataset repo not found: {self.repo_id_or_dataset_path}") from error

        self.source = self.snapshot_path
        return self.snapshot_path

    def _dataset_solver(self):
        """
        Return the usable local dataset path for the next loader/trainer layer.
        :return:
        """
        return self.source

    def _is_path(self):
        path = Path(self.source).expanduser()
        if path.exists():
            self.is_existed = True
            self.source = str(path)
            return True

        if path.is_absolute() or self.source.startswith(("./", "../", "~")):
            self.is_missing_path = True

        self.is_existed = False
        return False

if __name__ == "__main__":
    raise SystemExit("Use DataSolver(repo_id_or_dataset_path).solve()")
