"""
Run cloud gpu
"""

from pathlib import Path
from script.cloud.modal_setup import setup

UV_DEPS = Path(__file__).parent / "pyproject.toml"
APP_NAME = "SFTModelTrainer"
app,trainer = setup(APP_NAME,main_file="sft_trainer.py")

@app.local_entrypoint()
def main():
    trainer.remote()