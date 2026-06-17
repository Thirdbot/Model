"""
Run cloud gpu
"""

from pathlib import Path
from script.cloud.modal_setup import setup

UV_DEPS = Path(__file__).parent / "pyproject.toml"
APP_NAME = "GRPOModelTrainer"
app,trainer = setup(APP_NAME,main_file="grpo_trainer.py")

@app.local_entrypoint()
def main():
    trainer.remote()