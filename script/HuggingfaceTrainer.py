"""
Trainer for huggingface model, suit for training custom model.
"""
from trl import GRPOTrainer, SFTTrainer
from trl.rewards import format_rewards,think_format_reward,other_rewards


def train_hf_model(config,dataset,model,selected_trainer='sft'):
    match selected_trainer:
        case 'grpo':
            trainer = GRPOTrainer(
                model=model,
                train_dataset=dataset,
                eval_dataset=dataset,
                reward_funcs=think_format_reward
            )
            trainer.train()
        case 'sft':
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
            )
            trainer.train()
        case _:
            raise ValueError("Invalid trainer selected.")

# supervise training
# grpo training