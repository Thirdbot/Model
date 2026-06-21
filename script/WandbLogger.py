"""
Logging model/parameters configs/architecture each build
"""
import os
from pathlib import Path
from uuid import uuid4

import wandb

from configs.configs import load_config


class WandbLogger:
    def __init__(self,
                  project="model",
                  run_name=None,
                  mode="offline",
                  entity=None,
                  group=None,
                  tags=None,
                  config=None,
                  save_dir=None,):
        root = Path(load_config("paths")["root"])
        default_save_dir = root / load_config("paths")["subdirs"]["wandb"]

        self.run = None
        self.project = project
        self.run_name = run_name
        self.mode = mode
        self.entity = entity
        self.group = group
        self.tags = tags
        self.config = config
        self.save_dir = Path(save_dir) if save_dir else default_save_dir

    @staticmethod
    def make_run_name(model_name, dataset_name, trainer_name):
        model = model_name.replace("/", "--")
        dataset = dataset_name.replace("/", "--")
        return f"{trainer_name}-{model}-{dataset}-{uuid4().hex[:8]}"

    def start(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        os.environ["WANDB_MODE"] = self.mode
        os.environ["WANDB_DIR"] = self.save_dir.as_posix()
        os.environ["WANDB_PROJECT"] = self.project

        self.run = wandb.init(
            project=self.project,
            name=self.run_name,
            mode=self.mode,
            entity=self.entity,
            group=self.group,
            tags=self.tags,
            config=self.config,
            dir=self.save_dir.as_posix(),
        )
        return self.run

    def finish(self):
        if self.run is not None:
            self.run.finish()

    def log_best_checkpoint(self, checkpoint_path, metric=None):
        if self.run is None or checkpoint_path is None:
            return

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            return

        payload = {"best_checkpoint": checkpoint_path.as_posix()}
        if metric is not None:
            payload["best_metric"] = metric
        wandb.log(payload)

        artifact = wandb.Artifact(
            name=f"{self.run_name}-best-checkpoint",
            type="model",
            metadata=payload,
        )
        artifact.add_dir(checkpoint_path.as_posix())
        self.run.log_artifact(artifact)

    @staticmethod
    def trainer_report_to():
        return ["wandb"]
