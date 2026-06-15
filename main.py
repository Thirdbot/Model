import os
import sys
from pathlib import Path

from script.HuggingfaceTrainer import HFTrainer
from script.helper.Collator import Collator

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from script.FolderManager import  manager
from script.HuggingfaceDownload import solve_dataset, solve_model
from script.DatatemplateEditor import Template

if __name__ == "__main__":

    # create folders
    manager()

    # download model and dataset
    model_solver, loaded_model = solve_model("geshang/Seg-R1-3B",
                                             load_in_n_bit=4,
                                             unsloth_mode=True )
    dataset_solver, dataset = solve_dataset(
        "thirdExec/synthetic-seismic-vlm",
    )

    print("model source:", model_solver.source)
    print("dataset:", dataset)


    model_solver.status_report()
    dataset_solver.status_report()

    if dataset_solver.needs_conversion:
        raise RuntimeError(dataset_solver.conversion_reason)

    model,tokenizer = loaded_model[:2]
    # model.print_trainable_parameters()
    model.gradient_checkpointing_disable()
    if hasattr(model, "base_model"):
        model.base_model.gradient_checkpointing_disable()
    model.config.use_cache = False
    model.enable_input_require_grads()
    # for name, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(name)
    model.train()
    processor = loaded_model[-1] if len(loaded_model) == 3 else None

    dataset = dataset['train']

    key_map = {
        "image": ["images"],
        "text": ["instruction", "problem","thinking","solution","answer"],
    }

    key_owner = {
        "system": ["instruction"],
        "user": ["problem", "images"],
        "assistant": ["thinking","solution"],
    }

    template = Template(dataset=dataset, tokenizer=tokenizer, model_name="geshang/Seg-R1-3B", dataset_name="thirdExec/synthetic-seismic-vlm",
                        key_map=key_map, key_owner=key_owner,set_add_generation_prompt=False)

    train_dataset, eval_dataset, test_dataset = template.solve()
    print(f"{train_dataset[0]}\n\n{eval_dataset[0]}\n\n{test_dataset[0]}")

    vision_collator = Collator(dataset=dataset, tokenizer=tokenizer, processor=processor).vision_language_collate
    # check if collator is working
    # batch = vision_collator([train_dataset[0]])
    # print((batch["labels"] != -100).sum())
    trainer = HFTrainer(model_name="geshang/Seg-R1-3B",
                        dataset_name="thirdExec/synthetic-seismic-vlm",
                        train_data=train_dataset,
                        eval_data=eval_dataset,
                        test_data=test_dataset,
                        model=model,
                        tokenizer=tokenizer,
                        processor=processor,
                        collator=vision_collator,
                        selected_trainer='sft')
    trainer.train_hf_model()


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