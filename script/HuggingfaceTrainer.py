"""
Trainer for huggingface model, suit for training custom model.
"""
from pathlib import Path
from script.custom.reward import combined_reward
from script.WandbLogger import WandbLogger

from configs.configs import load_config

class HFTrainer:
    def __init__(self,train_data,test_data,eval_data,model,tokenizer,processor,model_name,dataset_name,
                 peft_config=None,selected_trainer='sft',sft_config=None,grpo_config=None,collator=None,
                 wandb_logger=None,epochs=1,batch_size=1,num_generation=2,generation_batch_size=2):

        self.train_data = train_data
        self.test_data = test_data
        self.eval_data = eval_data
        self.peft_config = peft_config

        self.epochs = epochs
        self.processor = processor
        self.tokenizer = tokenizer
        self.batch_size = batch_size

        self.model = model
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.root_path = Path(load_config("paths")['root'])
        self.model_save_path = self.root_path / load_config("paths")['dirs']['saves'] / selected_trainer / self.model_name / self.dataset_name # grpo / sft / custom_*
        self.model_save_checkpoint_path = self.root_path / load_config("paths")['subdirs']['train_checkpoints'] / selected_trainer / self.model_name / self.dataset_name # grpo / sft / custom_*
        self.selected_trainer = selected_trainer
        self.wandb = wandb_logger or WandbLogger(
            project="model",
            run_name=WandbLogger.make_run_name(self.model_name, self.dataset_name, self.selected_trainer),
            group=self.selected_trainer,
            tags=[self.model_name, self.dataset_name, self.selected_trainer],
            config={
                "model_name": self.model_name,
                "dataset_name": self.dataset_name,
                "trainer": self.selected_trainer,
            },
        )
        self.sft_config = sft_config or {
            "output_dir":self.model_save_checkpoint_path.as_posix(),
            "per_device_train_batch_size":self.batch_size,
            "gradient_accumulation_steps":1,
            "gradient_checkpointing": False,
            "learning_rate":2e-5,
            "max_length":2048,
            "dataset_text_field":"text",
            "dataset_kwargs": {"skip_prepare_dataset": True},
            "save_strategy": "epoch",
            "save_total_limit": 2,
            "logging_steps": 10,
            "num_train_epochs": self.epochs,
            "eval_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "remove_unused_columns":False, # no drop and should not drop unless you want to drop the columns
            "disable_tqdm": False,
            "report_to": self.wandb.trainer_report_to(),
            "bf16": False,
            "fp16": False,
        }

        self.grpo_config = grpo_config or {
            "output_dir": self.model_save_checkpoint_path.as_posix(),
            "per_device_train_batch_size": self.batch_size,
            "num_generations":num_generation,
            "generation_batch_size":generation_batch_size,
            "gradient_accumulation_steps": 8,
            "gradient_checkpointing": False,
            "learning_rate": 1e-6,
            "max_prompt_length": 1024,
            "max_completion_length": 512,
            "save_strategy": "epoch",
            "save_total_limit": 2,
            "logging_steps": 5,
            "num_train_epochs": self.epochs,
            "eval_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "remove_unused_columns": False, # no drop and should not drop unless you want to drop the columns
            "disable_tqdm": False,
            "report_to": self.wandb.trainer_report_to(),
            "bf16": False,
            "fp16": False,
        }

        self.sft = self._set_sft_config(self.sft_config)
        self.grpo = self._set_grpo_config(self.grpo_config)

        self.collator = collator

    @staticmethod
    def _set_sft_config(config):
        from trl import SFTConfig
        return SFTConfig(
            **config
        )

    @staticmethod
    def _set_grpo_config(config):
        from trl import GRPOConfig
        return GRPOConfig(
            **config
        )

    def train_hf_model(self):
        self.wandb.start()
        try:
            match self.selected_trainer:
                case 'grpo':
                    from trl import GRPOTrainer
                    processing_class = self.processor or self.tokenizer
                    trainer = GRPOTrainer(
                        model=self.model,
                        train_dataset=self.train_data,
                        eval_dataset=self.eval_data,
                        processing_class=processing_class,
                        peft_config=self.peft_config,
                        reward_funcs=combined_reward,
                        args=self.grpo,
                    )
                    trainer.train()
                case 'sft':
                    from trl import SFTTrainer, SFTConfig
                    processing_class = self.processor or self.tokenizer
                    trainer = SFTTrainer(
                        model= self.model,
                        train_dataset= self.train_data,
                        eval_dataset= self.eval_data,
                        processing_class= processing_class,
                        data_collator= self.collator,
                        peft_config = self.peft_config,
                        args= self.sft,
                    )
                    trainer.train()
                case _:
                    raise ValueError("Invalid trainer selected.")
            self._save_and_log_best_model(trainer)
        finally:
            self.wandb.finish()

    def _save_and_log_best_model(self, trainer):
        best_checkpoint = trainer.state.best_model_checkpoint
        best_metric = trainer.state.best_metric
        if best_checkpoint is None:
            print("No best checkpoint found; skipping best model save.")
            return

        self.model_save_path.mkdir(parents=True, exist_ok=True)
        trainer.save_model(self.model_save_path.as_posix())
        if self.processor is not None:
            self.processor.save_pretrained(self.model_save_path.as_posix())
        elif self.tokenizer is not None:
            self.tokenizer.save_pretrained(self.model_save_path.as_posix())

        self.wandb.log_best_checkpoint(best_checkpoint, best_metric)
        print(f"Best checkpoint: {best_checkpoint}")
        print(f"Best model saved to: {self.model_save_path}")
