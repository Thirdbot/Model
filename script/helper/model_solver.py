import json
from pathlib import Path

from tokenizers import processors
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig
)
import re
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training,PeftModel

from configs import load_config

model_config_naming = load_config("models")
path_config = load_config("paths")
names = model_config_naming.names

CUSTOM = names.CUSTOM
CAUSALLM = names.CAUSALLM
MULTI = names.MULTI
SPECIAL_VL = names.SPECIAL_VL

SPECIAL_VL_MODELS = model_config_naming.SPECIAL_VL_MODELS

ROOT_PATH = Path(path_config['root']).resolve()
CHECKPOINT_SAVE_PATH = ROOT_PATH / path_config['subdirs']['train_checkpoints']
SAVE_PATH = ROOT_PATH /  path_config['dirs']['train']


class ModelSolver:
    """
    Inspect a local path or Hugging Face repo id, then choose one loader path
    from detected model traits. Hugging Face loaders own cache resolution.
    Get a chat template or create a new one from types (this will be in a new config as template).
    Load model that is undergoing a transformation like Quantized after finished training for smoothly loading
    """

    def __init__(self, repo_id_or_model_path, load_in_n_bit=4, unsloth_mode=True):
        self.repo_id_or_model_path = str(repo_id_or_model_path)
        self.source = self.repo_id_or_model_path
        self._resolve_local_path()
        self.use_lora = False
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
        self.lora_target_modules = []

        # Lora Parameters for unsloth and hf
        self.r = 16
        self.lora_alpha = 32
        self.dropout = 0.05

        self.peft_config = LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.dropout,
            bias="none",
            target_modules = ["q_proj", "v_proj"]

        )

    def status_report(self):
        print(f"""
        Currently, loading {self.repo_id_or_model_path}
        Source used by loader: {self.source}
        Loading model type: {self.model_type}
        Detected model traits: {self.MODEL_TYPES}
        Architectures: {self.architectures}
        Auto map keys: {list(self.auto_map.keys()) if self.auto_map else []}
        unsloth requested: {self.unsloth_mode}
        unsloth attempted: {self.unsloth_attempted}
        unsloth error: {self.unsloth_error}
        Fallback reason: {self.fallback_reason}
        Loading with Quantized n bits: {self.load_in_n_bit}
        Loading with Framework: {self.load_with}
        Loading with method: {self.load_method}
        LoRA requested: {self.use_lora}
        LoRA applied: {self.lora_applied}
        LoRA backend: {self.lora_backend}
        LoRA reason: {self.lora_reason}
        LoRA target modules: {self.lora_target_modules}
        
        """)
    def solve(self):
        return self._model_solver()

    def load_save_model(self, at_dataset, method="sft"):
        save_dirs, check_dirs = self._find_model(
            self.repo_id_or_model_path,
            at_dataset,
            method,
        )

        # prefer latest checkpoint, fallback to latest saved dir
        candidates = check_dirs or save_dirs
        if not candidates:
            raise FileNotFoundError("No saved model/checkpoint found.")

        adapter_path = self._latest_checkpoint(candidates[-1])

        return self.load_trained_model(
            base_model=self.repo_id_or_model_path,
            adapter_path=adapter_path,
        )

    @staticmethod
    def load_trained_model(base_model, adapter_path):
        model = AutoModelForImageTextToText.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype="auto",
        )

        model = PeftModel.from_pretrained(model, adapter_path)

        processor = AutoProcessor.from_pretrained(base_model, use_fast=False)

        return model, processor

    @staticmethod
    def _latest_checkpoint(out_dir):
        out_dir = Path(out_dir)
        ckpts = sorted(
            out_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        return ckpts[-1] if ckpts else out_dir

    @staticmethod
    def _find_model(name, at_dataset, method):
        save_root = SAVE_PATH / method / name / at_dataset
        check_root = CHECKPOINT_SAVE_PATH / method / name / at_dataset

        save_dirs = sorted(save_root.iterdir()) if save_root.exists() else []
        check_dirs = sorted(check_root.iterdir()) if check_root.exists() else []

        return save_dirs, check_dirs
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
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            return AutoTokenizer.from_pretrained(
                source,
                trust_remote_code=trust_remote_code,
                use_fast=False,
            )

    def _record_unsloth_failure(self, message, error):
        self.unsloth_attempted = True
        self.unsloth_error = f"{message}: {error}"
        self.fallback_reason = "Unsloth load failed; using Hugging Face fallback."
        print(self.unsloth_error)

    def _apply_lora(self,model):
        if not self.load_in_n_bit:
            self.lora_applied = False
            self.lora_backend = None
            self.lora_reason = "Full finetuning requested;"
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
                model = prepare_model_for_kbit_training(model) # just quantize model and left sftTrainer handle pefft config
            # self.peft_config = self._build_peft_config()
            self.lora_applied = False
            self.lora_backend = "huggingface"
            self.lora_reason = "LoRA deferred to SFTTrainer via peft_config."
            # return get_peft_model(model,self.peft_config)
            return model

        self.lora_applied = False
        self.lora_backend = self.load_with
        self.lora_reason = "Skipped LoRA because model backend is unknown."
        return model

    def _build_peft_config(self):
        from peft import TaskType
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.dropout,
            bias="none",
            target_modules=self._select_lora_target_modules(),
            task_type=TaskType.CAUSAL_LM

        )

    def _select_lora_target_modules(self):
        if self.modality == "vision" or MULTI in self.MODEL_TYPES or SPECIAL_VL in self.MODEL_TYPES:
            self.lora_target_modules = self._vision_language_target_modules()
        elif CAUSALLM in self.MODEL_TYPES:
            self.lora_target_modules = self._causal_lm_target_modules()
        elif CUSTOM in self.MODEL_TYPES:
            self.lora_target_modules = self._custom_target_modules()
        else:
            self.lora_target_modules = self._auto_target_modules()

        return self.lora_target_modules

    def _causal_lm_target_modules(self):
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    def _vision_language_target_modules(self):
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    def _custom_target_modules(self):
        if self.modality == "vision" or MULTI in self.MODEL_TYPES or SPECIAL_VL in self.MODEL_TYPES:
            return self._vision_language_target_modules()
        return self._causal_lm_target_modules()

    def _auto_target_modules(self):
        model_type = (self.model_type or "").lower()
        if any(marker in model_type for marker in ("bert", "roberta", "deberta", "electra")):
            return ["query", "key", "value", "dense"]
        return self._causal_lm_target_modules()

    def _model_solver(self):
        if SPECIAL_VL in self.MODEL_TYPES:
            loaded = self._normalize_loaded_components(self._load_special_vision(self.source))
            self.model = loaded[0]
            self.tokenizer = loaded[1]
            if len(loaded) == 3:
                self.processor = loaded[2]
            self.model = self._apply_lora(self.model)
            if self.processor is not None:
                return self.model, self.tokenizer, self.processor
            return self.model, self.tokenizer

        if CUSTOM in self.MODEL_TYPES:
            loaded = self._normalize_loaded_components(self._load_custom(self.source))
            self.model = loaded[0]
            self.tokenizer = loaded[1]
            if len(loaded) == 3:
                self.processor = loaded[2]
            self.model = self._apply_lora(self.model)
            if self.processor is not None:
                return self.model, self.tokenizer, self.processor
            return self.model, self.tokenizer

        if MULTI in self.MODEL_TYPES:
            loaded = self._normalize_loaded_components(self._load_multi(self.source))
            self.model = loaded[0]
            self.tokenizer = loaded[1]
            if len(loaded) == 3:
                self.processor = loaded[2]
            self.model = self._apply_lora(self.model)
            if self.processor is not None:
                return self.model, self.tokenizer, self.processor
            return self.model, self.tokenizer

        if CAUSALLM in self.MODEL_TYPES:
            loaded = self._normalize_loaded_components(self._load_causal(self.source))
            self.model = loaded[0]
            self.tokenizer = loaded[1]
            self.model = self._apply_lora(self.model)
            return self.model, self.tokenizer

        loaded = self._normalize_loaded_components(self._load_auto(self.source))
        self.model = loaded[0]
        self.tokenizer = loaded[1]
        self.model = self._apply_lora(self.model)
        return self.model, self.tokenizer

    def _normalize_loaded_components(self, loaded):
        model = loaded[0]
        tokenizer = loaded[1] if len(loaded) > 1 else None
        processor = loaded[2] if len(loaded) > 2 else None

        if processor is None and hasattr(tokenizer, "image_processor"):
            processor = tokenizer
            tokenizer = getattr(processor, "tokenizer", None)

        if tokenizer is None and processor is not None:
            tokenizer = getattr(processor, "tokenizer", None)

        self._normalize_eos_token(tokenizer)
        processor_tokenizer = getattr(processor, "tokenizer", None)
        self._normalize_eos_token(processor_tokenizer)

        if processor is not None and tokenizer is not None and hasattr(processor, "tokenizer"):
            processor.tokenizer = tokenizer

        if processor is not None:
            return model, tokenizer, processor
        return model, tokenizer

    @staticmethod
    def _normalize_eos_token(tokenizer):
        if tokenizer is None:
            return

        vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
        current_eos = getattr(tokenizer, "eos_token", None)

        if current_eos and current_eos != "<EOS_TOKEN>":
            return
        if "<|im_end|>" not in vocab:
            return

        tokenizer.eos_token = "<|im_end|>"
        if hasattr(tokenizer, "convert_tokens_to_ids"):
            eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            if eos_token_id is not None:
                tokenizer.eos_token_id = eos_token_id

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
                model = AutoModelForCausalLM.from_pretrained(source)
                tokenizer = self._load_tokenizer(source)
                self.load_with = "huggingface"
                self.load_method = "Causal"
                self.modality = "lang"
                return model, tokenizer
            # quantize in hf model
            elif self.load_in_n_bit:
                model = AutoModelForCausalLM.from_pretrained(source,
                                                             quantization_config=self.bnb_config)
                tokenizer = self._load_tokenizer(source)
                self.load_with = "huggingface"
                self.load_method = "Causal"
                self.modality = "lang"
                return model, tokenizer
        # fall-back not doing quantizing or load in unsloth
        except Exception:
            model = AutoModel.from_pretrained(source)
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
                model = AutoModel.from_pretrained(source)
            else:
                model = AutoModel.from_pretrained(
                    source,
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
                model = AutoModelForImageTextToText.from_pretrained(source)
            else:
                model = AutoModelForImageTextToText.from_pretrained(
                    source,
                    quantization_config=self.bnb_config,
                )
            processor = AutoProcessor.from_pretrained(
                source,
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
                model = AutoModelForImageTextToText.from_pretrained(
                    source,
                    trust_remote_code=True,
                )
            else:
                model = AutoModelForImageTextToText.from_pretrained(
                    source,
                    trust_remote_code=True,
                    quantization_config=self.bnb_config,
                )
            tokenizer = self._load_tokenizer(source, trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(
                source,
                trust_remote_code=True,
                use_fast=False,
            )
            self.load_with = "huggingface"
            self.load_method = "SpecialVL"
            self.modality = "vision"
            return model, tokenizer, processor
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
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    source,
                    trust_remote_code=True,
                    quantization_config=self.bnb_config,
                )
        elif "AutoModel" in self.auto_map:
            if not self.load_in_n_bit:
                model = AutoModel.from_pretrained(
                    source,
                    trust_remote_code=True,
                )
            else:
                model = AutoModel.from_pretrained(
                    source,
                    trust_remote_code=True,
                    quantization_config=self.bnb_config,
                )
        else:
            if not self.load_in_n_bit:
                model = AutoModel.from_pretrained(
                    source,
                    trust_remote_code=True,
                )
            else:
                model = AutoModel.from_pretrained(
                    source,
                    trust_remote_code=True,
                    quantization_config=self.bnb_config,
                )

        tokenizer = self._load_tokenizer(source, trust_remote_code=True)
        try:
            processor = AutoProcessor.from_pretrained(
                source,
                trust_remote_code=True,
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
            )
        except Exception:
            return None

    def _resolve_local_path(self):
        path = Path(self.repo_id_or_model_path).expanduser()
        if path.exists():
            self.source = str(path)

if __name__ == "__main__":
    repo_id = "OpenGVLab/InternVL3-1B"
    model_solver = ModelSolver(repo_id)
    model_solver.solve()
