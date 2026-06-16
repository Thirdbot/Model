# infer_sft.py
import json
from pathlib import Path

import torch

from script.FolderManager import manager
from script.HuggingfaceDownload import solve_dataset, solve_model
from script.DatatemplateEditor import Template


def to_device(batch, device):
    return {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def get_images(example):
    images = example.get("images", None)
    if images is None:
        return None

    if not isinstance(images, list):
        images = [images]

    out = []
    for image in images:
        if getattr(image, "mode", None) != "RGB":
            image = image.convert("RGB")
        out.append(image)
    return out


@torch.inference_mode()
def generate_one(model, processor, example, max_new_tokens=512):
    text = example["text"]
    images = get_images(example)

    batch = processor(
        text=[text],
        images=images,
        padding=True,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    batch = to_device(batch, device)

    generated = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

    prompt_len = batch["input_ids"].shape[-1]
    new_tokens = generated[:, prompt_len:]

    tokenizer = getattr(processor, "tokenizer", processor)
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]


def main():
    manager()

    model_name = "geshang/Seg-R1-3B"
    dataset_name = "thirdExec/synthetic-seismic-vlm"

    # Load base solver only so we can call your saved-model loader.
    model_solver, loaded_model = solve_model(
        model_name,
        load_in_n_bit=4,
        unsloth_mode=False,
    )

    # Load trained SFT model / adapter.
    model, processor = model_solver.load_save_model(
        at_dataset=dataset_name,
        method="sft",
    )

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = loaded_model[1]

    model.eval()
    model.config.use_cache = True

    dataset_solver, dataset = solve_dataset(dataset_name)
    if dataset_solver.needs_conversion:
        raise RuntimeError(dataset_solver.conversion_reason)

    dataset = dataset["train"]

    key_map = {
        "image": ["images"],
        "text": ["instruction", "problem", "thinking", "solution", "answer"],
    }

    # Important for inference:
    # assistant is empty, so template ends at assistant generation prompt.
    key_owner = {
        "system": ["instruction"],
        "user": ["problem", "images"],
        "assistant": [],
    }

    template = Template(
        dataset=dataset,
        tokenizer=tokenizer,
        model_name=model_name,
        dataset_name=dataset_name,
        key_map=key_map,
        key_owner=key_owner,
        set_add_generation_prompt=True,
        temp_for="sft",
    )

    train_data, eval_data, test_data = template.solve()

    out_path = Path("logs/inference_sft_predictions.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for i, example in enumerate(test_data):
            pred = generate_one(
                model=model,
                processor=processor,
                example=example,
                max_new_tokens=512,
            )

            row = {
                "idx": i,
                "prediction": pred,
            }

            # keep prompt text for debugging
            if "text" in example:
                row["prompt"] = example["text"]

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            print(f"\n--- sample {i} ---")
            print(pred)

    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()