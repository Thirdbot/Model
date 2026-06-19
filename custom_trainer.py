from pathlib import Path

from torch.utils.data import DataLoader

from configs import load_config
from script.CustomTrainerForMask import save_vlm_and_mask_decoder, train_mask_decoder_loop
from script.HuggingfaceDownload import solve_model, solve_dataset
from script.DatatemplateEditor import Template
from script.helper.Collator import Collator
from script.helper.MaskDecoder import MaskDecoder
from script.custom.CustomModel import AddModelToken, VLMWithMaskDecoder

ADDITIONAL_TOKENS = [
    "<region>",
    "</region>",
    "<object>",
    "</object>",
    "<class_id>",
    "</class_id>",
    "<color>",
    "</color>",
    "<evidence>",
    "</evidence>",
    "<bbox>",
    "</bbox>",
    "<think>",
    "</think>",
    "<answer>",
    "</answer>",
    "<SEG>",
]


def train_model(model_repo_id,
                dataset_repo_id,
                unsloth_mode=False,
                load_in_n_bit=None,
                key_map=None,
                key_owner=None,
                add_prompt_gen=False,
                epochs=1,
                batch_size=1,
                train_mode='custom_sft',
                resume_model_type='sft'
                ):
    is_peft_applied = False
    model_solver, loaded_model = solve_model(
        model_repo_id,
        load_in_n_bit=load_in_n_bit,
        unsloth_mode=unsloth_mode,
    )

    model, tokenizer = loaded_model[:2]
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

    token_helper = AddModelToken(
        model,
        tokenizer=tokenizer,
        processor=processor,
        additional_tokens=ADDITIONAL_TOKENS,
        seg_token="<SEG>",
    )
    model = token_helper.get_model()
    tokenizer = token_helper.get_tokenizer()
    seg_token_id = token_helper.seg_token_id

    dataset_solver, dataset = solve_dataset(dataset_repo_id)
    dataset = dataset["train"]

    template = Template(
        dataset=dataset,
        tokenizer=tokenizer,
        model_name=model_repo_id,
        dataset_name=dataset_repo_id,
        key_map=key_map,
        key_owner=key_owner,
        temp_for=train_mode,
        set_add_generation_prompt=add_prompt_gen,
        additional_images=["masks"],
        additional_tokens=ADDITIONAL_TOKENS,
    )

    train_dataset, eval_dataset, test_dataset = template.solve()
    print(f"{train_dataset[0]}\n\n{eval_dataset[0]}\n\n{test_dataset[0]}")

    collator = Collator(
        dataset=dataset,
        tokenizer=tokenizer,
        processor=processor,
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator.tasks_collate,
    )

    hidden_size = model.config.hidden_size
    mask_decoder = MaskDecoder(
        hidden_size=hidden_size,
    ).cuda()

    custom_model = VLMWithMaskDecoder(
        vlm=model,
        mask_decoder=mask_decoder,
        seg_token_id=seg_token_id,
    )

    model = train_mask_decoder_loop(
        model=custom_model,
        dataloader=dataloader,
        epochs=epochs,
        peft_config=model_solver.peft_config if not is_peft_applied else None,
        device="cuda",
    )
    root_path = Path(load_config("paths")['root'])
    model_save_path = root_path / load_config("paths")['dirs']['saves'] / train_mode / model_repo_id / dataset_repo_id  # grpo / sft / custom_*
    save_vlm_and_mask_decoder(model, tokenizer, processor,output_dir=model_save_path)

if __name__ == "__main__":
    key_map = {
        "image": ["images", "masks"],
        "text": ["instruction","reason", "question","evidence", "answer"],
    }

    key_owner = {
        "system": ["instruction"],
        "user": ["question", "images"],
        "assistant": ["evidence","reason", "answer"],
    }

    train_model(model_repo_id="geshang/Seg-R1-3B",
                dataset_repo_id="thirdExec/synthetic-seismic-vlm",
                unsloth_mode=False,
                load_in_n_bit=4,
                key_map=key_map,
                key_owner=key_owner,
                add_prompt_gen=False,
                train_mode='custom_sft',
                epochs=100,
                batch_size=1)
