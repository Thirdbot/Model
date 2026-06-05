import json
from pathlib import Path

from unsloth import FastVisionModel,FastLanguageModel
from huggingface_hub import auth_check, snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig
)

from configs import load_config

model_config_naming = load_config("models")
names = model_config_naming.names

CUSTOM = names.CUSTOM
CAUSALLM = names.CAUSALLM
MULTI = names.MULTI
SPECIAL_VL = names.SPECIAL_VL

SPECIAL_VL_MODELS = model_config_naming.SPECIAL_VL_MODELS


class ModelSolver:
    """
    Inspect a local path or Hugging Face repo id, download repo snapshots when
    needed, then choose one loader path from detected model traits.
    """

    def __init__(self, repo_id_or_model_path, cache_dir=None,load_in_n_bit=4,unsloth_mode=True):
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
        self.raw_config = {}
        self.MODEL_TYPES = self._get_model_types()

        self.model = None
        self.tokenizer = None
        self.processor = None

        self.unsloth_mode = unsloth_mode
        self.max_seq_length = 2048
        # Quantized for unsloth
        self.load_in_n_bit = load_in_n_bit or None
        self.load_4_bit = True if self.load_in_n_bit == 4 else False
        self.load_8_bit = True if self.load_in_n_bit == 8 else False
        self.load_16_bit = True if self.load_in_n_bit == 16 else False

        # Quantized for HF only 4 or 8
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.load_4_bit,
            load_in_8bit=self.load_8_bit,
        )
        # Non-Quantized
        self.full_finetuning = True if self.load_in_n_bit is None else False

        # status
        self.load_with = None # unsloth or HF
        self.load_method = None # custom,causal,...

        # Lora Parameters for unsloth and hf

    def status_report(self):
        print(f"""
        Currently, loading {self.repo_id_or_model_path} from {self.cache_dir}
        Loading model type: {self.model_type}
        Loading with Quantized n bits: {self.load_in_n_bit}
        Loading with Framework: {self.load_with}
        Loading with method: {self.load_method}
        
        """)
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
        self._load_raw_config_fields() # fall-back when config.json is not reliable

        if self.auto_map:
            collectible_types.append(CUSTOM)
        if any("ForCausalLM" in arch for arch in self.architectures):
            collectible_types.append(CAUSALLM)
        if self.llm_config and self.vision_config:
            collectible_types.append(MULTI)
        if self.model_type in SPECIAL_VL_MODELS:
            collectible_types.append(SPECIAL_VL)

        return collectible_types

    def _load_raw_config_fields(self):
        config_path = Path(self.source) / "config.json"
        if not config_path.exists():
            return

        try:
            self.raw_config = json.loads(config_path.read_text())
        except Exception:
            return

        self.model_type = self.model_type or self.raw_config.get("model_type")
        self.architectures = self.architectures or self.raw_config.get("architectures", [])
        self.auto_map = self.auto_map or self.raw_config.get("auto_map", {})
        self.llm_config = self.llm_config or self.raw_config.get("llm_config", {})
        self.vision_config = self.vision_config or self.raw_config.get("vision_config", {})

    def _load_tokenizer(self, source, trust_remote_code=False):
        try:
            return AutoTokenizer.from_pretrained(
                source,
                cache_dir=self.cache_dir,
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            return AutoTokenizer.from_pretrained(
                source,
                cache_dir=self.cache_dir,
                trust_remote_code=trust_remote_code,
                use_fast=False,
            )

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
        # try to load causal model with unsloth or fall back to hf
        # options to not quantize in unsloth
        if self.unsloth_mode and not self.load_in_n_bit:
            try:
                self.load_with = "unsloth"
                self.load_method = "Causal"
                return self._load_llm_with_unsloth(source)
            except Exception as error:
                print(f"Could not load causal model with unsloth: {error}")
        # options to quantize in unsloth
        elif self.unsloth_mode and self.load_in_n_bit:
            try:
                self.load_with = "unsloth"
                self.load_method = "Causal"
                return self._load_llm_with_unsloth_quantized(source)
            except Exception as error:
                print(f"Could not load causal model with unsloth quantized: {error}")

        # fall-back doing quantizing in hf
        try:
            # not quantize in hf model
            if not self.load_in_n_bit:
                model = AutoModelForCausalLM.from_pretrained(source, cache_dir=self.cache_dir)
                tokenizer = self._load_tokenizer(source)
                self.load_with = "huggingface"
                self.load_method = "Causal"
                return model, tokenizer
            # quantize in hf model
            elif self.load_in_n_bit:
                model = AutoModelForCausalLM.from_pretrained(source,
                                                             cache_dir=self.cache_dir,
                                                             quantization_config=self.bnb_config)
                tokenizer = self._load_tokenizer(source)
                self.load_with = "huggingface"
                self.load_method = "Causal"
                return model, tokenizer
        # fall-back not doing quantizing or load in unsloth
        except Exception:
            model = AutoModel.from_pretrained(source, cache_dir=self.cache_dir)
            tokenizer = self._load_tokenizer(source)
            self.load_with = "huggingface"
            self.load_method = "Causal"
            return model, tokenizer

    def _load_auto(self, source):
        # try to load auto model with unsloth or fall back to hf
        if self.unsloth_mode and not self.load_in_n_bit:
            try:
                self.load_with = "unsloth"
                self.load_method = "Auto"
                return self._load_llm_with_unsloth(source)
            except Exception as error:
                print(f"Could not load auto model with unsloth: {error}")
        elif self.unsloth_mode and self.load_in_n_bit:
            try:
                self.load_with = "unsloth"
                self.load_method = "Auto"
                return self._load_llm_with_unsloth_quantized(source)
            except Exception as error:
                print(f"Could not load auto model with unsloth quantized: {error}")

        # fall-back doing quantizing in hf
        try:
            if not self.load_in_n_bit:
                model = AutoModel.from_pretrained(source, cache_dir=self.cache_dir)
            else:
                model = AutoModel.from_pretrained(
                    source,
                    cache_dir=self.cache_dir,
                    quantization_config=self.bnb_config,
                )
            tokenizer = self._load_tokenizer(source)
            self.load_with = "huggingface"
            self.load_method = "Auto"
            return model, tokenizer
        except Exception as error:
            raise RuntimeError(f"Could not load model with AutoModel: {error}") from error

    def _load_multi(self, source):
        # try to load multimodal model with unsloth or fall back to hf
        if self.unsloth_mode and not self.load_in_n_bit:
            try:
                self.load_with = "unsloth"
                self.load_method = "Multi"
                return self._load_vlm_with_unsloth(source)
            except Exception as error:
                print(f"Could not load multimodal model with unsloth: {error}")
        elif self.unsloth_mode and self.load_in_n_bit:
            try:
                self.load_with = "unsloth"
                self.load_method = "Multi"
                return self._load_vlm_with_unsloth_quantized(source)
            except Exception as error:
                print(f"Could not load multimodal model with unsloth quantized: {error}")

        # fall-back doing quantizing in hf
        try:
            self.load_with = "huggingface"
            self.load_method = "Multi"
            if not self.load_in_n_bit:
                model = AutoModel.from_pretrained(source, cache_dir=self.cache_dir)
            else:
                model = AutoModel.from_pretrained(
                    source,
                    cache_dir=self.cache_dir,
                    quantization_config=self.bnb_config,
                )
            processor = AutoProcessor.from_pretrained(source, cache_dir=self.cache_dir)
            return model, processor
        except Exception as error:
            raise RuntimeError(f"Could not load multimodal model: {error}") from error

    def _load_special_vision(self, source):
        # try to load special vision model with unsloth or fall back to hf
        if self.unsloth_mode and not self.load_in_n_bit:
            try:
                self.load_with = "unsloth"
                self.load_method = "SpecialVL"
                return self._load_vlm_with_unsloth(source)
            except Exception as error:
                print(f"Could not load special vision model with unsloth: {error}")
        elif self.unsloth_mode and self.load_in_n_bit:
            try:
                self.load_with = "unsloth"
                self.load_method = "SpecialVL"
                return self._load_vlm_with_unsloth_quantized(source)
            except Exception as error:
                print(f"Could not load special vision model with unsloth quantized: {error}")

        # fall-back doing quantizing in hf
        try:
            if not self.load_in_n_bit:
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
                    quantization_config=self.bnb_config,
                )
            tokenizer = self._load_tokenizer(source, trust_remote_code=True)
            self.load_with = "huggingface"
            self.load_method = "SpecialVL"
            return model, tokenizer
        except Exception as error:
            raise RuntimeError(f"Could not load special vision model: {error}") from error

    def _load_custom(self, source):
        # try to load custom model with unsloth or fall back to hf
        if self.unsloth_mode and not self.load_in_n_bit:
            try:
                if MULTI in self.MODEL_TYPES or SPECIAL_VL in self.MODEL_TYPES:
                    self.load_with = "unsloth"
                    self.load_method = "Custom"
                    return self._load_vlm_with_unsloth(source)
                self.load_with = "unsloth"
                self.load_method = "Custom"
                return self._load_llm_with_unsloth(source)
            except Exception as error:
                print(f"Could not load custom model with unsloth: {error}")
        elif self.unsloth_mode and self.load_in_n_bit:
            try:
                if MULTI in self.MODEL_TYPES or SPECIAL_VL in self.MODEL_TYPES:
                    self.load_with = "unsloth"
                    self.load_method = "Custom"
                    return self._load_vlm_with_unsloth_quantized(source)
                self.load_with = "unsloth"
                self.load_method = "Custom"
                return self._load_llm_with_unsloth_quantized(source)
            except Exception as error:
                print(f"Could not load custom model with unsloth quantized: {error}")

        # fall-back doing quantizing in hf
        if "AutoModelForCausalLM" in self.auto_map:
            if not self.load_in_n_bit:
                model = AutoModelForCausalLM.from_pretrained(
                    source,
                    trust_remote_code=True,
                    cache_dir=self.cache_dir,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    source,
                    trust_remote_code=True,
                    cache_dir=self.cache_dir,
                    quantization_config=self.bnb_config,
                )
        elif "AutoModel" in self.auto_map:
            if not self.load_in_n_bit:
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
                    quantization_config=self.bnb_config,
                )
        else:
            if not self.load_in_n_bit:
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
                    quantization_config=self.bnb_config,
                )

        tokenizer = self._load_tokenizer(source, trust_remote_code=True)
        try:
            processor = AutoProcessor.from_pretrained(
                source,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
            )
            self.load_with = "huggingface"
            self.load_method = "Custom"
            return model, tokenizer, processor
        except Exception:
            self.load_with = "huggingface"
            self.load_method = "Custom"
            return model, tokenizer

    def _load_llm_with_unsloth(self,source):
        model,tokenizer = FastLanguageModel.from_pretrained(
            model_name = source,
            max_seq_length=self.max_seq_length,
            use_gradient_checkpointing="unsloth"
        )
        return model,tokenizer

    def _load_vlm_with_unsloth(self,source):
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=source,
            max_seq_length=self.max_seq_length,
            use_gradient_checkpointing="unsloth"
        )
        return model, tokenizer

    def _load_llm_with_unsloth_quantized(self,source):
        model,tokenizer = FastLanguageModel.from_pretrained(
            model_name=source,
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.load_4_bit,
            load_in_8bit=self.load_8_bit,
            load_in_16bit=self.load_16_bit,
            use_gradient_checkpointing="unsloth"
        )
        return model,tokenizer

    def _load_vlm_with_unsloth_quantized(self,source):
        model,tokenizer = FastVisionModel.from_pretrained(
            model_name=source,
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.load_4_bit,
            load_in_8bit=self.load_8_bit,
            load_in_16bit=self.load_16_bit,
            use_gradient_checkpointing="unsloth"
        )
        return model,tokenizer

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
