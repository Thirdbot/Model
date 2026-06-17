import os
import subprocess
import modal
from configs.configs import load_config
from pathlib import Path

import sys

UV_DEPS = Path(load_config('paths')['root']).resolve() / "pyproject.toml"
VOLUME_NAME = "ModelTrainer-volume" # use same volume for every trainer
MODAL_WORK_DIR = "/root/Model"
VOL_DIR = "/vol"
MODAL_HF_CACHE = f"{VOL_DIR}/hf"
PYTHON_VERSION = "3.11"

def setup(appname,main_file,gpu="A100-40GB",cpu=8,memory=64):
    # create volumes
    volume = modal.Volume.from_name(VOLUME_NAME,create_if_missing=True)

    # repository dependencies
    image = (
        modal.Image.debian_slim(python_version=PYTHON_VERSION).pip_install_from_pyproject(
            UV_DEPS
        ).workdir(MODAL_WORK_DIR)
    ).env(
        {
            "MODEL_ROOT": "/vol",
            "HF_HOME":f"{MODAL_HF_CACHE}",
            "HF_DATASETS_CACHE":f"{MODAL_HF_CACHE}/datasets",
            "HF_HUB_CACHE":f"{MODAL_HF_CACHE}/hub",
            "WANDB_DIR": "/vol/wandb",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    ).add_local_dir(
        "script",f"{MODAL_WORK_DIR}/script"
    ).add_local_dir(
        "configs",f"{MODAL_WORK_DIR}/configs"
    ).add_local_dir(
        "templates",f"{MODAL_WORK_DIR}/templates"
    ).add_local_file(
        f"{main_file}",remote_path=f"{MODAL_WORK_DIR}/{main_file}"
    )

    # call functions associate with the image and all code as 1 run
    app = modal.App(appname,image=image)

    @app.function(
            gpu=gpu,
            cpu=cpu,
            memory=memory * 1024,
            timeout=60 * 60 * 24,
            volumes={VOL_DIR: volume},
            secrets=[
                modal.Secret.from_dotenv('.env')
            ],serialized=True
        )
    def train():
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{MODAL_WORK_DIR}:{env.get('PYTHONPATH', '')}"

        try:
            subprocess.run(
                [sys.executable, f"{main_file}"],
                cwd=MODAL_WORK_DIR,
                env=env,
                check=True,
            )
        finally:
            volume.commit()
    return app,train
