import sys
from pathlib import Path

from script.HuggingfaceTrainer import HFTrainer
from script.helper.Collator import Collator

from script.helper.FolderManager import  manager
from script.HuggingfaceDownload import solve_dataset, solve_model
from script.DatatemplateEditor import Template

def train_model(model_repo_id,
                dataset_repo_id,
                unsloth_mode=False,
                load_in_n_bit=None,
                key_map=None,
                key_owner=None,
                add_prompt_gen=False,
                epochs=1,
                batch_size=1,
                train_mode='sft',
                resume_model_type='sft'):
    is_peft_applied = False
    # download model and dataset
    model_solver, loaded_model = solve_model(model_repo_id,
                                             load_in_n_bit=load_in_n_bit,
                                             unsloth_mode=unsloth_mode
                                             )
    model,tokenizer = loaded_model[:2]

    dataset_solver, dataset = solve_dataset(
        dataset_repo_id
    )

    print("model source:", model_solver.source)
    print("dataset:", dataset)

    processor = loaded_model[-1] if len(loaded_model) == 3 else None

    try:
        #load local if exists
        model, processor = model_solver.load_save_model(
            at_dataset=dataset_repo_id,
            method=resume_model_type,
        )
        print(f"load local model from:{model_repo_id}/{dataset_repo_id}")
        is_peft_applied = True
    except Exception as e:
        print("could not load local model, will train from scratch")

    model_solver.status_report()
    dataset_solver.status_report()

    try:
        if dataset_solver.needs_conversion:
            raise RuntimeError(dataset_solver.conversion_reason)
    except Exception as e:
        print("Dataset needs conversion:")

    model.gradient_checkpointing_disable()
    if hasattr(model, "base_model"):
        model.base_model.gradient_checkpointing_disable()
    model.config.use_cache = False
    model.enable_input_require_grads()

    model.train()

    dataset = dataset['train']

    template = Template(dataset=dataset, tokenizer=tokenizer, model_name=model_repo_id, dataset_name=dataset_repo_id,
                        key_map=key_map, key_owner=key_owner,set_add_generation_prompt=add_prompt_gen,temp_for=train_mode,
                        additional_images=[],model=model,processor=processor)
    model = template.model
    tokenizer = template.tokenizer
    processor = template.processor

    train_dataset, eval_dataset, test_dataset = template.solve()
    print(f"{train_dataset[0]}\n\n{eval_dataset[0]}\n\n{test_dataset[0]}")

    vision_collator = Collator(dataset=dataset, tokenizer=tokenizer, processor=processor).vision_language_collate
    # check if collator is working
    # batch = vision_collator([train_dataset[0]])
    # print((batch["labels"] != -100).sum())
    trainer = HFTrainer(model_name=model_repo_id,
                        dataset_name=dataset_repo_id,
                        train_data=train_dataset,
                        eval_data=eval_dataset,
                        test_data=test_dataset,
                        model=model,
                        tokenizer=tokenizer,
                        processor=processor,
                        collator=vision_collator,
                        selected_trainer=train_mode,
                        peft_config=model_solver.peft_config if not is_peft_applied else None,
                        epochs=epochs,
                        batch_size=batch_size,
                        )
    trainer.train_hf_model()

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent
    sys.path.insert(0, str(ROOT))
    # example keys mapping for sft training
    key_map = {
        "image": ["images"],
        "text": ["instruction", "question", "evidence", "reason", "answer"],
    }

    key_owner = {
        "system": ["instruction"],
        "user": ["question", "images"],
        "assistant": ["evidence", "reason", "answer"],
    }
    manager() # create folder
    train_model(model_repo_id="geshang/Seg-R1-3B", dataset_repo_id="thirdExec/synthetic-seismic-vlm",
                unsloth_mode=False, load_in_n_bit=4,add_prompt_gen=False,
                key_map=key_map, key_owner=key_owner, train_mode='sft',resume_model_type='sft',
                epochs=100,batch_size=1
                )

"""
replicate training from paper first for stable pipeline. then keep training,changing architecture,changing dataset https://arxiv.org/pdf/2506.22624
"""

"""Steps would be
1. model get solve 
2. dataset get solve
3. model check and dataset check compatibility both types and template if exist
4. model and dataset alignment by template and target training
5. select training and report config
6. let it run
7. report
"""
