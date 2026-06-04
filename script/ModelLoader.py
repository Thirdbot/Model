import os
from pathlib import Path

from transformers import AutoConfig, AutoModel, AutoTokenizer
from unsloth import FastVisionModel

"""
Model loader from huggingface for Trainer
"""

def load_hf_visiontotext_model(model_path):
    model_path = str(model_path)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    return model,tokenizer

def load_unsloth_visiontotext_model(model_path):

    model,tokenizer = FastVisionModel.from_pretrained(
        str(model_path)
    )
    return model,tokenizer
