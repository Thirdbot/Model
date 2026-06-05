import json
from pathlib import Path

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
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training

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
    Get a chat template or create a new one from types (this will be in a new config as template).
    Load model that is undergoing a transformation like Quantized after finished training for smoothly loading
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
        self.modality = None # vision / lang
        self.unsloth_attempted = False
        self.unsloth_error = None
        self.fallback_reason = None
        self.lora_applied = False
        self.lora_backend = None
        self.lora_reason = None

        # Lora Parameters for unsloth and hf
        self.r = 16
        self.lora_alpha = 32
        self.dropout = 0.05
        self.use_lora = False if self.full_finetuning else True
        self.peft_config = LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.dropout,
            bias="none"
        )

    def status_report(self):
        print(f"""
        Currently, loading {self.repo_id_or_model_path} from {self.cache_dir}
        Snapshot path: {self.snapshot_path}
        Loading model type: {self.model_type}
        Detected model traits: {self.MODEL_TYPES}
        Architectures: {self.architectures}
        Auto map keys: {list(self.auto_map.keys()) if self.auto_map else []}
        Unsloth requested: {self.unsloth_mode}
        Unsloth attempted: {self.unsloth_attempted}
        Unsloth error: {self.unsloth_error}
        Fallback reason: {self.fallback_reason}
        Loading with Quantized n bits: {self.load_in_n_bit}
        Loading with Framework: {self.load_with}
        Loading with method: {self.load_method}
        LoRA requested: {self.use_lora}
        LoRA applied: {self.lora_applied}
        LoRA backend: {self.lora_backend}
        LoRA reason: {self.lora_reason}
        
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
        if self._looks_like_vision_language():
            if MULTI not in collectible_types:
                collectible_types.append(MULTI)
            if SPECIAL_VL not in collectible_types:
                collectible_types.append(SPECIAL_VL)

        return collectible_types

    def _looks_like_vision_language(self):
        model_type = (self.model_type or "").lower()
        if any(marker in model_type for marker in ("vl", "vision", "llava", "video")):
            return True

        source_path = Path(self.source)
        processor_files = [
            "processor_config.json",
            "preprocessor_config.json",
            "image_processor_config.json",
            "video_preprocessor_config.json",
        ]
        return any((source_path / file_name).exists() for file_name in processor_files)

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

    def _record_unsloth_failure(self, message, error):
        self.unsloth_attempted = True
        self.unsloth_error = f"{message}: {error}"
        self.fallback_reason = "Unsloth load failed; using Hugging Face fallback."
        print(self.unsloth_error)

    def _download_model(self):
        self.snapshot_path = snapshot_download(
            self.repo_id_or_model_path,
            repo_type="model",
            cache_dir=self.cache_dir,
        )
        self.source = self.snapshot_path
        return self.snapshot_path

    def _apply_lora(self,model):
        if not self.use_lora:
            self.lora_applied = False
            self.lora_backend = None
            self.lora_reason = "Full finetuning requested; LoRA disabled."
            return model

        if self.load_with == "unsloth":
            from unsloth import FastVisionModel,FastLanguageModel
            # load peft with unsloth
            if self.modality == "vision":
                # load peft with VisionModel
                self.lora_applied = True
                self.lora_backend = "unsloth"
                self.lora_reason = "Applied Unsloth vision LoRA."
                return FastVisionModel.get_peft_model(
                    model,
                    finetune_vision_layers=False,
                    finetune_language_layers=True,
                    finetune_attention_modules=True,
                    finetune_mlp_modules=True,
                    r=self.r,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.dropout,
                    bias="none",
                )
            elif self.modality == "lang":
                self.lora_applied = True
                self.lora_backend = "unsloth"
                self.lora_reason = "Applied Unsloth language LoRA."
                return FastLanguageModel.get_peft_model(
                    model,
                    r=self.r,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.dropout,
                    bias="none",
                )
            self.lora_applied = False
            self.lora_backend = "unsloth"
            self.lora_reason = "Skipped LoRA because modality is unknown."
            return model

        if self.load_with == "huggingface":
            # load peft with huggingface
            if self.load_in_n_bit:
                model = prepare_model_for_kbit_training(model)
            self.lora_applied = True
            self.lora_backend = "huggingface"
            self.lora_reason = "Applied Hugging Face PEFT LoRA."
            return get_peft_model(model,self.peft_config)

        self.lora_applied = False
        self.lora_backend = self.load_with
        self.lora_reason = "Skipped LoRA because model backend is unknown."
        return model

    def _model_solver(self):
        if SPECIAL_VL in self.MODEL_TYPES:
            self.model, self.tokenizer = self._load_special_vision(self.source)
            self.model = self._apply_lora(self.model)
            return self.model, self.tokenizer

        if CUSTOM in self.MODEL_TYPES:
            loaded = self._load_custom(self.source)
            self.model = loaded[0]
            self.tokenizer = loaded[1]
            if len(loaded) == 3:
                self.processor = loaded[2]
            self.model = self._apply_lora(self.model)
            if self.processor is not None:
                return self.model, self.tokenizer, self.processor
            return self.model, self.tokenizer

        if MULTI in self.MODEL_TYPES:
            self.model, self.processor = self._load_multi(self.source)
            self.model = self._apply_lora(self.model)
            return self.model, self.processor

        if CAUSALLM in self.MODEL_TYPES:
            self.model, self.tokenizer = self._load_causal(self.source)
            self.model = self._apply_lora(self.model)
            return self.model, self.tokenizer

        self.model, self.tokenizer = self._load_auto(self.source)
        self.model = self._apply_lora(self.model)
        return self.model, self.tokenizer

    def _load_causal(self, source):
        # try to load causal model with unsloth or fall back to hf
        # options to not quantize in unsloth
        if self.unsloth_mode and not self.load_in_n_bit:
            try:
                self.load_with = "unsloth"
                self.load_method = "Causal"
                self.modality = "lang"
                self.unsloth_attempted = True
                return self._load_llm_with_unsloth(source)
            except Exception as error:
                self._record_unsloth_failure("Could not load causal model with unsloth", error)
        # options to quantize in unsloth
        elif self.unsloth_mode and self.load_in_n_bit:
            try:
                self.load_with = "unsloth"
                self.load_method = "Causal"
                self.modality = "lang"
                self.unsloth_attempted = True
                return self._load_llm_with_unsloth_quantized(source)
            except Exception as error:
                self._record_unsloth_failure("Could not load causal model with unsloth quantized", error)

        # fall-back doing quantizing in hf
        try:
            # not quantize in hf model
            if not self.load_in_n_bit:
                model = AutoModelForCausalLM.from_pretrained(source, cache_dir=self.cache_dir)
                tokenizer = self._load_tokenizer(source)
                self.load_with = "huggingface"
                self.load_method = "Causal"
                self.modality = "lang"
                return model, tokenizer
            # quantize in hf model
            elif self.load_in_n_bit:
                model = AutoModelForCausalLM.from_pretrained(source,
                                                             cache_dir=self.cache_dir,
                                                             quantization_config=self.bnb_config)
                tokenizer = self._load_tokenizer(source)
                self.load_with = "huggingface"
                self.load_method = "Causal"
                self.modality = "lang"
                return model, tokenizer
        # fall-back not doing quantizing or load in unsloth
        except Exception:
            model = AutoModel.from_pretrained(source, cache_dir=self.cache_dir)
            tokenizer = self._load_tokenizer(source)
            self.load_with = "huggingface"
            self.load_method = "Causal"
            self.modality = "lang"
            return model, tokenizer

    def _load_auto(self, source):
        # try to load auto model with unsloth or fall back to hf
        if self.unsloth_mode and not self.load_in_n_bit:
            try:
                self.load_with = "unsloth"
                self.load_method = "Auto"
                self.modality = "lang"
                self.unsloth_attempted = True
                return self._load_llm_with_unsloth(source)
            except Exception as error:
                self._record_unsloth_failure("Could not load auto model with unsloth", error)
        elif self.unsloth_mode and self.load_in_n_bit:
            try:
                self.load_with = "unsloth"
                self.load_method = "Auto"
                self.modality = "lang"
                self.unsloth_attempted = True
                return self._load_llm_with_unsloth_quantized(source)
            except Exception as error:
                self._record_unsloth_failure("Could not load auto model with unsloth quantized", error)

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
            self.modality = "lang"
            return model, tokenizer
        except Exception as error:
            raise RuntimeError(f"Could not load model with AutoModel: {error}") from error

    def _load_multi(self, source):
        # try to load multimodal model with unsloth or fall back to hf
        if self.unsloth_mode and not self.load_in_n_bit:
            try:
                self.load_with = "unsloth"
                self.load_method = "Multi"
                self.modality = "vision"
                self.unsloth_attempted = True
                return self._load_vlm_with_unsloth(source)
            except Exception as error:
                self._record_unsloth_failure("Could not load multimodal model with unsloth", error)
        elif self.unsloth_mode and self.load_in_n_bit:
            try:
                self.load_with = "unsloth"
                self.load_method = "Multi"
                self.modality = "vision"
                self.unsloth_attempted = True
                return self._load_vlm_with_unsloth_quantized(source)
            except Exception as error:
                self._record_unsloth_failure("Could not load multimodal model with unsloth quantized", error)

        # fall-back doing quantizing in hf
        try:
            self.load_with = "huggingface"
            self.load_method = "Multi"
            self.modality = "vision"
            if not self.load_in_n_bit:
                model = AutoModel.from_pretrained(source, cache_dir=self.cache_dir)
            else:
                model = AutoModel.from_pretrained(
                    source,
                    cache_dir=self.cache_dir,
                    quantization_config=self.bnb_config,
                )
            processor = AutoProcessor.from_pretrained(
                source,
                cache_dir=self.cache_dir,
                use_fast=False,
            )
            return model, processor
        except Exception as error:
            raise RuntimeError(f"Could not load multimodal model: {error}") from error

    def _load_special_vision(self, source):
        # try to load special vision model with unsloth or fall back to hf
        if self.unsloth_mode and not self.load_in_n_bit:
            try:
                self.load_with = "unsloth"
                self.load_method = "SpecialVL"
                self.modality = "vision"
                self.unsloth_attempted = True
                return self._load_vlm_with_unsloth(source)
            except Exception as error:
                self._record_unsloth_failure("Could not load special vision model with unsloth", error)
        elif self.unsloth_mode and self.load_in_n_bit:
            try:
                self.load_with = "unsloth"
                self.load_method = "SpecialVL"
                self.modality = "vision"
                self.unsloth_attempted = True
                return self._load_vlm_with_unsloth_quantized(source)
            except Exception as error:
                self._record_unsloth_failure("Could not load special vision model with unsloth quantized", error)

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
            self.modality = "vision"
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
                    self.modality = "vision"
                    self.unsloth_attempted = True
                    return self._load_vlm_with_unsloth(source)
                self.load_with = "unsloth"
                self.load_method = "Custom"
                self.modality = "lang"
                self.unsloth_attempted = True
                return self._load_llm_with_unsloth(source)
            except Exception as error:
                self._record_unsloth_failure("Could not load custom model with unsloth", error)
        elif self.unsloth_mode and self.load_in_n_bit:
            try:
                if MULTI in self.MODEL_TYPES or SPECIAL_VL in self.MODEL_TYPES:
                    self.load_with = "unsloth"
                    self.load_method = "Custom"
                    self.modality = "vision"
                    self.unsloth_attempted = True
                    return self._load_vlm_with_unsloth_quantized(source)
                self.load_with = "unsloth"
                self.load_method = "Custom"
                self.modality = "lang"
                self.unsloth_attempted = True
                return self._load_llm_with_unsloth_quantized(source)
            except Exception as error:
                self._record_unsloth_failure("Could not load custom model with unsloth quantized", error)

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
                use_fast=False,
            )
            self.load_with = "huggingface"
            self.load_method = "Custom"
            self.modality = "vision" if MULTI in self.MODEL_TYPES or SPECIAL_VL in self.MODEL_TYPES else "lang"
            return model, tokenizer, processor
        except Exception:
            self.load_with = "huggingface"
            self.load_method = "Custom"
            self.modality = "vision" if MULTI in self.MODEL_TYPES or SPECIAL_VL in self.MODEL_TYPES else "lang"
            return model, tokenizer

    def _load_llm_with_unsloth(self,source):
        from unsloth import FastLanguageModel

        model,tokenizer = FastLanguageModel.from_pretrained(
            model_name = source,
            max_seq_length=self.max_seq_length,
            use_gradient_checkpointing="unsloth"
        )
        return model,tokenizer

    def _load_vlm_with_unsloth(self,source):
        from unsloth import FastVisionModel

        with self._slow_processor_for_unsloth():
            model, tokenizer = FastVisionModel.from_pretrained(
                model_name=source,
                max_seq_length=self.max_seq_length,
                use_gradient_checkpointing="unsloth"
            )
        return model, tokenizer

    def _load_llm_with_unsloth_quantized(self,source):
        from unsloth import FastLanguageModel

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
        from unsloth import FastVisionModel

        with self._slow_processor_for_unsloth():
            model,tokenizer = FastVisionModel.from_pretrained(
                model_name=source,
                max_seq_length=self.max_seq_length,
                load_in_4bit=self.load_4_bit,
                load_in_8bit=self.load_8_bit,
                load_in_16bit=self.load_16_bit,
                use_gradient_checkpointing="unsloth"
            )
        return model,tokenizer

    def _slow_processor_for_unsloth(self):
        class SlowProcessorPatch:
            def __enter__(patch_self):
                patch_self.original = AutoProcessor.from_pretrained

                def from_pretrained_with_slow_processor(*args, **kwargs):
                    kwargs.setdefault("use_fast", False)
                    return patch_self.original(*args, **kwargs)

                AutoProcessor.from_pretrained = from_pretrained_with_slow_processor

            def __exit__(patch_self, exc_type, exc_value, traceback):
                AutoProcessor.from_pretrained = patch_self.original

        return SlowProcessorPatch()

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
