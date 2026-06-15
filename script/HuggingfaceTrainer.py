"""
Trainer for huggingface model, suit for training custom model.
"""
from pathlib import Path

from trl import GRPOTrainer, SFTTrainer, SFTConfig, GRPOConfig
from trl.rewards import format_rewards,think_format_reward,other_rewards


from configs import load_config


class HFTrainer:
    def __init__(self,train_data,test_data,eval_data,model,tokenizer,processor,model_name,dataset_name,selected_trainer='sft',sft_config=None,grpo_config=None,collator=None):

        self.train_data = train_data
        self.test_data = test_data
        self.eval_data = eval_data

        self.processor = processor
        self.tokenizer = tokenizer
        self.model = model
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.root_path = Path(load_config("paths")['root'])
        self.model_save_path = self.root_path / load_config("paths")['dirs']['saves'] / self.model_name / self.dataset_name
        self.model_save_checkpoint_path = self.root_path / load_config("paths")['subdirs']['train_checkpoints'] / selected_trainer / self.model_name / self.dataset_name
        self.selected_trainer = selected_trainer
        self.sft_config = sft_config or {
            "output_dir":self.model_save_checkpoint_path.as_posix(),
            "per_device_train_batch_size":1,
            "gradient_accumulation_steps":8,
            "learning_rate":2e-5,
            "max_length":2048,
            "dataset_text_field":"text",
            "save_steps": 10,
            "logging_steps": 1,
            "max_steps": 10,
            "remove_unused_columns":False, # no drop and should not drop unless you want to drop the columns
            }
        self.grpo_config = grpo_config or {
            "output_dir": self.model_save_checkpoint_path.as_posix(),
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-6,
            "max_prompt_length": 1024,
            "max_completion_length": 512,
            "save_steps": 10,
            "logging_steps": 1,
            "max_steps": 10,
            "remove_unused_columns": False, # no drop and should not drop unless you want to drop the columns
        }
        self.sft = self._set_sft_config(self.sft_config)
        self.grpo = self._set_grpo_config(self.grpo_config)

        self.collator = collator

    @staticmethod
    def _set_sft_config(config):
        return SFTConfig(
            **config
        )

    @staticmethod
    def _set_grpo_config(config):
        return GRPOConfig(
            **config
        )

    def train_hf_model(self):
        match self.selected_trainer:
            case 'grpo':
                trainer = GRPOTrainer(
                    model=self.model,
                    train_dataset=self.train_data,
                    eval_dataset=self.eval_data,
                    reward_funcs=think_format_reward,
                    args=self.grpo,
                )
                trainer.train()
            case 'sft':
                trainer = SFTTrainer(
                    model= self.model,
                    train_dataset= self.train_data,
                    eval_dataset= self.eval_data,
                    processing_class= self.processor or self.tokenizer,
                    data_collator= self.collator,
                    args= self.sft,
                )
                trainer.train()
            case _:
                raise ValueError("Invalid trainer selected.")

# sft training and grpo training

if __name__ == "__main__":
    from script.HuggingfaceDownload import solve_model, solve_dataset
    from script.DatatemplateEditor import Template
    from helper.Collator import Collator

    model_solver, loaded_model = solve_model("geshang/Seg-R1-3B",
                                             load_in_n_bit=16,
                                             unsloth_mode=False)
    model, tokenizer = loaded_model[:2]
    processor = loaded_model[-1] if len(loaded_model) == 3 else None

    dataset_path = "/home/third/Desktop/simulationv2/Dataset/multimodal_multi_image_dataset.csv"
    # dataset = read_csv(dataset_path)
    dataset_solver, dataset = solve_dataset(
        # "SakanaAI/JA-Multi-Image-VQA" #,
        "geshang/FCoT"
    )
    dataset = dataset['train']

    key_map = {
        "image": ["image"],
        "text": ["thinking", "problem", "solution"],
    }

    key_owner = {
        "system": ["system_prompt"],
        "user": ["problem", "image"],
        "assistant": ["thinking", "solution"],
    }

    template = Template(dataset=dataset, tokenizer=tokenizer, model_name="geshang/Seg-R1-3B",
                        dataset_name="geshang/FCoT", key_map=key_map, key_owner=key_owner)
    train_dataset, eval_dataset, test_dataset = template.solve()
    print(f"{train_dataset[0]}\n\n{eval_dataset[0]}\n\n{test_dataset[0]}")

    vision_collator = Collator(dataset=dataset, tokenizer=tokenizer, processor=processor).vision_language_collate
    trainer = HFTrainer(model_name="geshang/Seg-R1-3B",
                        dataset_name="geshang/FCoT",
                        train_data=train_dataset,
                        eval_data=eval_dataset,
                        test_data=test_dataset,
                        model=model,
                        tokenizer=tokenizer,
                        processor=processor,
                        collator=vision_collator,
                        selected_trainer='sft')
    trainer.train_hf_model()