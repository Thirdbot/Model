from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs import load_config
from script.CustomTrainerForMask import save_vlm_and_mask_decoder, train_mask_decoder_loop
from script.HuggingfaceDownload import solve_model, solve_dataset
from script.DatatemplateEditor import Template
from script.WandbLogger import WandbLogger
from script.helper.Collator import Collator
from script.helper.MaskDecoder import MaskDecoder
from script.custom.CustomModel import VLMWithMaskDecoder


def load_mask_decoder_if_exists(mask_decoder, model_save_path, device="cuda"):
    mask_decoder_path = model_save_path / "mask_decoder.pt"
    if not mask_decoder_path.exists():
        print("could not load local mask decoder, will train from scratch")
        return mask_decoder

    ckpt = torch.load(mask_decoder_path, map_location=device)
    state_dict = (
        ckpt["mask_decoder_state_dict"]
        if isinstance(ckpt, dict) and "mask_decoder_state_dict" in ckpt
        else ckpt
    )
    mask_decoder.load_state_dict(state_dict)
    print(f"load local mask decoder from: {mask_decoder_path}")
    return mask_decoder


def load_resume_model_with_fallback(model_solver, dataset_repo_id, resume_model_type, fallback_model_type="sft"):
    model_types = [resume_model_type]
    if fallback_model_type and fallback_model_type not in model_types:
        model_types.append(fallback_model_type)

    last_error = None
    for model_type in model_types:
        try:
            model, processor = model_solver.load_save_model(
                at_dataset=dataset_repo_id,
                method=model_type,
            )
            print(f"load local {model_type} model from: {model_solver.repo_id_or_model_path}/{dataset_repo_id}")
            return model, processor, model_type
        except Exception as error:
            last_error = error
            print(f"could not load local {model_type} model")

    print(f"could not load local model, will train from scratch: {last_error}")
    return None, None, None


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
                resume_model_type='custom_sft',
                lambda_mask=2.0,
                bce_weight=1.0,
                dice_weight=2.0,
                wandb_logger=None,
                ):
    wandb_logger = wandb_logger or WandbLogger(
        project="model",
        run_name=WandbLogger.make_run_name(model_repo_id, dataset_repo_id, train_mode),
        group=train_mode,
        tags=[model_repo_id, dataset_repo_id, train_mode],
        config={
            "model_name": model_repo_id,
            "dataset_name": dataset_repo_id,
            "trainer": train_mode,
            "epochs": epochs,
            "batch_size": batch_size,
            "lambda_mask": lambda_mask,
            "bce_weight": bce_weight,
            "dice_weight": dice_weight,
        },
    )
    wandb_logger.start()

    is_peft_applied = False
    try:
        model_solver, loaded_model = solve_model(
            model_repo_id,
            load_in_n_bit=load_in_n_bit,
            unsloth_mode=unsloth_mode,
        )

        model, tokenizer = loaded_model[:2]
        processor = loaded_model[-1] if len(loaded_model) == 3 else None

        resumed_model, resumed_processor, loaded_model_type = load_resume_model_with_fallback(
            model_solver=model_solver,
            dataset_repo_id=dataset_repo_id,
            resume_model_type=resume_model_type,
            fallback_model_type="sft",
        )
        if resumed_model is not None:
            model = resumed_model
            processor = resumed_processor
            tokenizer = getattr(processor, "tokenizer", tokenizer)
            is_peft_applied = True

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
            model=model,
            processor=processor,
        )
        model = template.model
        tokenizer = template.tokenizer
        processor = template.processor
        seg_token_id = template.seg_token_id

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

        root_path = Path(load_config("paths")['root'])
        model_save_path = root_path / load_config("paths")['dirs']['saves'] / train_mode / model_repo_id / dataset_repo_id  # grpo / sft / custom_*

        hidden_size = model.config.hidden_size
        mask_decoder = MaskDecoder(
            hidden_size=hidden_size,
        ).cuda()
        mask_decoder = load_mask_decoder_if_exists(
            mask_decoder=mask_decoder,
            model_save_path=model_save_path,
            device="cuda",
        )

        custom_model = VLMWithMaskDecoder(
            vlm=model,
            mask_decoder=mask_decoder,
            seg_token_id=seg_token_id,
            lambda_mask=lambda_mask,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
        )

        model = train_mask_decoder_loop(
            model=custom_model,
            dataloader=dataloader,
            epochs=epochs,
            peft_config=model_solver.peft_config if not is_peft_applied else None,
            device="cuda",
            wandb_logger=wandb_logger,
        )
        save_vlm_and_mask_decoder(model, tokenizer, processor,output_dir=model_save_path)
    finally:
        wandb_logger.finish()

if __name__ == "__main__":
    key_map = {
        "image": ["images"],
        "text": ["instruction", "question", "evidence", "answer"],
    }

    key_owner = {
        "system": ["instruction"],
        "user": ["question", "images"],
        "assistant": ["evidence", "answer"],
    }

    train_model(model_repo_id="Qwen/Qwen2-VL-2B-Instruct",
                dataset_repo_id="thirdExec/synthetic-seismic-vlm",
                unsloth_mode=False,
                load_in_n_bit=8,
                key_map=key_map,
                key_owner=key_owner,
                add_prompt_gen=False,
                train_mode='custom_sft',
                epochs=100,
                batch_size=1)
