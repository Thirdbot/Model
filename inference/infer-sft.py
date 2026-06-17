# infer_sft.py
import json
from pathlib import Path

import torch

from script.helper.FolderManager import manager
from script.HuggingfaceDownload import solve_dataset, solve_model
from script.DatatemplateEditor import Template

from PIL import Image, ImageDraw, ImageFont
import textwrap


def save_preview(example, prediction, out_dir, idx):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = get_images(example)

    if not images:
        canvas = Image.new("RGB", (900, 400), "white")
    else:
        img = images[0].copy().convert("RGB")
        img.thumbnail((700, 700))
        canvas = Image.new("RGB", (900, img.height + 260), "white")
        canvas.paste(img, (10, 10))

    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    y = canvas.height - 240
    draw.text((10, y), f"Sample {idx}", fill="black", font=font)
    y += 30

    wrapped = textwrap.wrap(prediction, width=90)
    for line in wrapped[:10]:
        draw.text((10, y), line, fill="black", font=font)
        y += 24

    out_path = out_dir / f"sample_{idx:04d}.png"
    canvas.save(out_path)
    return out_path

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

            preview_path = save_preview(
                example=example,
                prediction=pred,
                out_dir="logs/inference_previews",
                idx=i,
            )

            row = {
                "idx": i,
                "prediction": pred,
                "preview_path": str(preview_path),
            }

            if "text" in example:
                row["prompt"] = example["text"]

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            print(f"\n--- sample {i} ---")
            print(pred)
            print("preview:", preview_path)

if __name__ == "__main__":
    main()