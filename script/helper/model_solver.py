from pathlib import Path

from huggingface_hub import auth_check, snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

CUSTOM = "custom"
CAUSALLM = "causal"
MULTI = "multimodal"
SPECIAL_VL = "special_vision"

SPECIAL_VL_MODELS = ["internvl_chat"]


class ModelSolver:
    """
    Inspect a local path or Hugging Face repo id, download repo snapshots when
    needed, then choose one loader path from detected model traits.
    """

    def __init__(self, repo_id_or_model_path, cache_dir=None):
        self.repo_id_or_model_path = str(repo_id_or_model_path)
        self.source = self.repo_id_or_model_path
        self.cache_dir = str(cache_dir) if cache_dir else None
        self.snapshot_path = None

        self.is_existed = False
        self.is_missing_path = False
        self.come_from_path = self._is_path()
        self.is_repo = False
        self.need_permission = False
        self.need_download = self._need_download()

        self.config = self._get_config()
        self.model_type = None
        self.architectures = []
        self.auto_map = {}
        self.llm_config = {}
        self.vision_config = {}
        self.MODEL_TYPES = self._get_model_types()

        self.model = None
        self.tokenizer = None
        self.processor = None

    def solve(self):
        if self.need_permission:
            raise PermissionError("The repo_id needs permission before it can be downloaded.")

        if self.is_repo and self.need_download:
            self._download_model()
            self.config = self._get_config()
            self.MODEL_TYPES = self._get_model_types()

        return self._model_solver()

    def _get_model_types(self):
        collectible_types = []
        if not self.config:
            return collectible_types

        self.model_type = getattr(self.config, "model_type", None)
        self.architectures = getattr(self.config, "architectures", []) or []
        self.auto_map = getattr(self.config, "auto_map", {}) or {}
        self.llm_config = getattr(self.config, "llm_config", {}) or {}
        self.vision_config = getattr(self.config, "vision_config", {}) or {}

        if self.auto_map:
            collectible_types.append(CUSTOM)
        if any("ForCausalLM" in arch for arch in self.architectures):
            collectible_types.append(CAUSALLM)
        if self.llm_config and self.vision_config:
            collectible_types.append(MULTI)
        if self.model_type in SPECIAL_VL_MODELS:
            collectible_types.append(SPECIAL_VL)

        return collectible_types

    def _download_model(self):
        self.snapshot_path = snapshot_download(
            self.repo_id_or_model_path,
            repo_type="model",
            cache_dir=self.cache_dir,
        )
        self.source = self.snapshot_path
        return self.snapshot_path

    def _model_solver(self):
        if SPECIAL_VL in self.MODEL_TYPES:
            self.model, self.tokenizer = self._load_special_vision(self.source)
            return self.model, self.tokenizer

        if CUSTOM in self.MODEL_TYPES:
            loaded = self._load_custom(self.source)
            self.model = loaded[0]
            self.tokenizer = loaded[1]
            if len(loaded) == 3:
                self.processor = loaded[2]
            return loaded

        if MULTI in self.MODEL_TYPES:
            self.model, self.processor = self._load_multi(self.source)
            return self.model, self.processor

        if CAUSALLM in self.MODEL_TYPES:
            self.model, self.tokenizer = self._load_causal(self.source)
            return self.model, self.tokenizer

        self.model, self.tokenizer = self._load_auto(self.source)
        return self.model, self.tokenizer

    def _load_causal(self, source):
        try:
            model = AutoModelForCausalLM.from_pretrained(source, cache_dir=self.cache_dir)
            tokenizer = AutoTokenizer.from_pretrained(source, cache_dir=self.cache_dir)
            return model, tokenizer
        except Exception:
            model = AutoModel.from_pretrained(source, cache_dir=self.cache_dir)
            tokenizer = AutoTokenizer.from_pretrained(source, cache_dir=self.cache_dir)
            return model, tokenizer

    def _load_auto(self, source):
        try:
            model = AutoModel.from_pretrained(source, cache_dir=self.cache_dir)
            tokenizer = AutoTokenizer.from_pretrained(source, cache_dir=self.cache_dir)
            return model, tokenizer
        except Exception as error:
            raise RuntimeError(f"Could not load model with AutoModel: {error}") from error

    def _load_multi(self, source):
        try:
            model = AutoModel.from_pretrained(source, cache_dir=self.cache_dir)
            processor = AutoProcessor.from_pretrained(source, cache_dir=self.cache_dir)
            return model, processor
        except Exception as error:
            raise RuntimeError(f"Could not load multimodal model: {error}") from error

    def _load_special_vision(self, source):
        try:
            model = AutoModel.from_pretrained(
                source,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                source,
                trust_remote_code=True,
                use_fast=False,
                cache_dir=self.cache_dir,
            )
            return model, tokenizer
        except Exception as error:
            raise RuntimeError(f"Could not load special vision model: {error}") from error

    def _load_custom(self, source):
        if "AutoModelForCausalLM" in self.auto_map:
            model = AutoModelForCausalLM.from_pretrained(
                source,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
            )
        elif "AutoModel" in self.auto_map:
            model = AutoModel.from_pretrained(
                source,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
            )
        else:
            model = AutoModel.from_pretrained(
                source,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
            )

        tokenizer = AutoTokenizer.from_pretrained(
            source,
            trust_remote_code=True,
            use_fast=False,
            cache_dir=self.cache_dir,
        )
        try:
            processor = AutoProcessor.from_pretrained(
                source,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
            )
            return model, tokenizer, processor
        except Exception:
            return model, tokenizer

    def _get_config(self):
        try:
            return AutoConfig.from_pretrained(
                self.source,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
            )
        except Exception:
            return None

    def _is_path(self):
        path = Path(self.repo_id_or_model_path).expanduser()
        if path.exists():
            self.is_existed = True
            self.source = str(path)
            return True

        if path.is_absolute() or self.repo_id_or_model_path.startswith(("./", "../", "~")):
            self.is_missing_path = True

        self.is_existed = False
        return False

    def _need_download(self):
        if self.come_from_path or self.is_missing_path:
            return False

        try:
            auth_check(self.repo_id_or_model_path)
            self.is_repo = True
            return True
        except GatedRepoError:
            self.need_permission = True
            self.is_repo = True
            return True
        except RepositoryNotFoundError:
            self.is_repo = False
            return False


if __name__ == "__main__":
    repo_id = "OpenGVLab/InternVL3-1B"
    model_solver = ModelSolver(repo_id)
    model_solver.solve()
