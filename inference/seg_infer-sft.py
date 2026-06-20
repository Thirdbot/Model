# infer_mask_sft.py

import json
from pathlib import Path
import textwrap

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image, ImageDraw, ImageFont
from peft import PeftModel

from script.helper.FolderManager import manager
from script.HuggingfaceDownload import solve_dataset, solve_model
from script.DatatemplateEditor import Template

from script.helper.MaskDecoder import MaskDecoder
from script.custom.CustomModel import get_image_features, get_seg_counts


MODEL_SAVE_DIR = Path("outputs/mask_model")
LORA_DIR = MODEL_SAVE_DIR / "vlm_lora"
TOKENIZER_DIR = MODEL_SAVE_DIR / "tokenizer"
MASK_DECODER_PATH = MODEL_SAVE_DIR / "mask_decoder.pt"

SEG_TOKEN = "<SEG>"


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


def get_images_matching_text(example):
    """
    Critical for your dataset:
    some examples have 8 images but text only has 4 <|image_pad|>.
    Qwen-VL requires image count to match placeholders.
    """
    text = example["text"]
    images = get_images(example)

    if images is None:
        return None

    n_img = text.count("<|image_pad|>")

    if n_img > 0:
        images = images[:n_img]

    return images


def get_hidden_size(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    cfg = base.config

    if hasattr(cfg, "hidden_size"):
        return cfg.hidden_size

    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        return cfg.text_config.hidden_size

    raise ValueError("Could not infer hidden_size from model config.")


def get_seg_hidden(hidden, input_ids, seg_token_id):
    """
    hidden:    [B, L, D]
    input_ids: [B, L]
    return:   [B, D]
    """
    seg_pos = input_ids.eq(seg_token_id)

    if not seg_pos.any():
        raise ValueError("No <SEG> token found in sequence.")

    has_seg = seg_pos.any(dim=1)

    if not has_seg.all():
        bad_rows = (~has_seg).nonzero(as_tuple=True)[0].tolist()
        raise ValueError(f"No <SEG> token found in rows: {bad_rows}")

    seg_idx = seg_pos.float().argmax(dim=1)
    b_idx = torch.arange(input_ids.size(0), device=input_ids.device)

    return hidden[b_idx, seg_idx]


def logits_to_mask(mask_logits, image_size, threshold=0.5):
    """
    mask_logits: [1, 1, H, W]
    image_size: PIL size = (W, H)
    """
    prob = torch.sigmoid(mask_logits)

    target_w, target_h = image_size

    prob = F.interpolate(
        prob,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )

    mask = prob[0, 0].detach().float().cpu().numpy()
    mask = (mask > threshold).astype(np.uint8) * 255

    return Image.fromarray(mask, mode="L")


def make_overlay(image, mask, alpha=0.45):
    image = image.convert("RGBA")
    mask = mask.convert("L")

    red = Image.new("RGBA", image.size, (255, 0, 0, 0))
    red_arr = np.array(red)

    alpha_arr = np.array(mask).astype(np.float32) / 255.0
    red_arr[..., 3] = (alpha_arr * int(255 * alpha)).astype(np.uint8)

    red = Image.fromarray(red_arr, mode="RGBA")
    return Image.alpha_composite(image, red)


def clean_prediction(text):
    text = text.replace(SEG_TOKEN, "")
    text = text.replace("<|im_end|>", "")
    text = text.replace("<|endoftext|>", "")
    return text.strip()


def save_preview(example, prediction, pred_mask, overlay, out_dir, idx):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = get_images_matching_text(example)

    if not images:
        canvas = Image.new("RGB", (1000, 500), "white")
    else:
        raw = images[0].copy().convert("RGB")
        raw.thumbnail((420, 420))

        ov = overlay.copy().convert("RGB")
        ov.thumbnail((420, 420))

        canvas_h = max(raw.height, ov.height) + 280
        canvas = Image.new("RGB", (1000, canvas_h), "white")

        canvas.paste(raw, (10, 10))
        canvas.paste(ov, (500, 10))

    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    y = canvas.height - 260
    draw.text((10, y), f"Sample {idx}", fill="black", font=font)
    y += 30

    wrapped = textwrap.wrap(prediction, width=95)
    for line in wrapped[:10]:
        draw.text((10, y), line, fill="black", font=font)
        y += 24

    out_path = out_dir / f"sample_{idx:04d}.png"
    canvas.save(out_path)

    mask_path = out_dir / f"sample_{idx:04d}_mask.png"
    overlay_path = out_dir / f"sample_{idx:04d}_overlay.png"

    pred_mask.save(mask_path)
    overlay.save(overlay_path)

    return out_path, mask_path, overlay_path


@torch.inference_mode()
def generate_one_with_mask(
    model,
    processor,
    mask_decoder,
    seg_token_id,
    example,
    max_new_tokens=512,
    threshold=0.5,
):
    text = example["text"]

    # Force assistant-side <SEG> so mask decoder has a query token.
    if SEG_TOKEN not in text.split("<|im_start|>assistant")[-1]:
        text = text + SEG_TOKEN

    images = get_images_matching_text(example)

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

    prediction = tokenizer.batch_decode(
        new_tokens,
        skip_special_tokens=False,
    )[0]

    prediction = clean_prediction(prediction)

    # second forward pass to get <SEG> hidden state
    forward_batch = dict(batch)
    forward_batch["input_ids"] = generated
    forward_batch["attention_mask"] = torch.ones_like(generated)

    forward_batch.pop("output_hidden_states", None)
    forward_batch.pop("return_dict", None)
    forward_batch.pop("use_cache", None)

    outputs = model(
        **forward_batch,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )

    hidden = outputs.hidden_states[-1]

    seg_hidden = get_seg_hidden(
        hidden=hidden,
        input_ids=generated,
        seg_token_id=seg_token_id,
    )

    mask_decoder = mask_decoder.to(seg_hidden.device)

    image_features = get_image_features(
        outputs=outputs,
        seg_counts=get_seg_counts(generated, seg_token_id),
        num_seg_tokens=seg_hidden.shape[0],
    )
    if image_features is not None:
        image_features = image_features.to(
            device=seg_hidden.device,
            dtype=seg_hidden.dtype,
        )

    mask_logits = mask_decoder(seg_hidden, image_features=image_features)

    if mask_logits.ndim == 3:
        mask_logits = mask_logits.unsqueeze(1)

    image_for_size = images[0]
    pred_mask = logits_to_mask(
        mask_logits=mask_logits,
        image_size=image_for_size.size,
        threshold=threshold,
    )

    overlay = make_overlay(image_for_size, pred_mask)


    return prediction, pred_mask, overlay
def find_processor_from_loaded_model(loaded_model):
    """
    Find processor inside solve_model(...) returned object.
    loaded_model may be tuple/list with model, processor, tokenizer, etc.
    """
    if not isinstance(loaded_model, (tuple, list)):
        return None

    for obj in loaded_model:
        # Qwen processor usually has both tokenizer and image_processor
        if hasattr(obj, "tokenizer") and (
            hasattr(obj, "image_processor") or hasattr(obj, "image_processor_class")
        ):
            return obj

    # fallback: any object with tokenizer
    for obj in loaded_model:
        if hasattr(obj, "tokenizer"):
            return obj

    return None

def load_saved_mask_model(model_solver_loaded):
    if isinstance(model_solver_loaded, (tuple, list)):
        base_model = model_solver_loaded[0]
    else:
        base_model = model_solver_loaded

    base_processor = find_processor_from_loaded_model(model_solver_loaded)

    if base_processor is None:
        raise ValueError(
            "Could not find processor from solve_model output. "
            "Print loaded_model types to inspect it."
        )

    processor_path = TOKENIZER_DIR if TOKENIZER_DIR.exists() else MODEL_SAVE_DIR

    # Important: use SAME class as base processor, not AutoProcessor
    processor = type(base_processor).from_pretrained(str(processor_path))

    if LORA_DIR.exists():
        model = PeftModel.from_pretrained(base_model, str(LORA_DIR))
    else:
        model = PeftModel.from_pretrained(base_model, str(MODEL_SAVE_DIR))

    model.eval()
    model.config.use_cache = True

    tokenizer = getattr(processor, "tokenizer", processor)
    seg_token_id = tokenizer.convert_tokens_to_ids(SEG_TOKEN)

    if seg_token_id is None or seg_token_id == tokenizer.unk_token_id:
        raise ValueError(
            f"{SEG_TOKEN} not found in saved processor/tokenizer at {processor_path}."
        )

    hidden_size = get_hidden_size(model)

    mask_decoder = MaskDecoder(hidden_size=hidden_size)

    ckpt = torch.load(MASK_DECODER_PATH, map_location="cpu")

    if isinstance(ckpt, dict) and "mask_decoder_state_dict" in ckpt:
        mask_decoder.load_state_dict(ckpt["mask_decoder_state_dict"])
    else:
        mask_decoder.load_state_dict(ckpt)

    mask_decoder.eval()

    return model, processor, mask_decoder, seg_token_id
def main():
    manager()

    model_name = "geshang/Seg-R1-3B"
    dataset_name = "thirdExec/synthetic-seismic-vlm"

    # Load base model same as your SFT script.
    model_solver, loaded_model = solve_model(
        model_name,
        load_in_n_bit=4,
        unsloth_mode=False,
    )

    # Instead of model_solver.load_save_model(...),
    # load LoRA + tokenizer + mask decoder from outputs/mask_model.
    model, processor, mask_decoder, seg_token_id = load_saved_mask_model(
        loaded_model
    )

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = loaded_model[1]

    dataset_solver, dataset = solve_dataset(dataset_name)
    if dataset_solver.needs_conversion:
        raise RuntimeError(dataset_solver.conversion_reason)

    dataset = dataset["train"]

    key_map = {
        "image": ["images", "masks"],
        "text": ["instruction", "question", "evidence", "answer"],
    }

    key_owner = {
        "system": ["instruction"],
        "user": ["question", "images"],
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
        additional_images=["masks"],
        model=model,
        processor=processor,
    )
    model = template.model
    processor = template.processor
    seg_token_id = template.seg_token_id

    train_data, eval_data, test_data = template.solve()

    out_path = Path("outputs/mask_model/inference_mask_predictions.jsonl")
    preview_dir = Path("outputs/mask_model/inference_previews")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for i, example in enumerate(test_data):
            pred, pred_mask, overlay = generate_one_with_mask(
                model=model,
                processor=processor,
                mask_decoder=mask_decoder,
                seg_token_id=seg_token_id,
                example=example,
                max_new_tokens=512,
                threshold=0.5,
            )

            preview_path, mask_path, overlay_path = save_preview(
                example=example,
                prediction=pred,
                pred_mask=pred_mask,
                overlay=overlay,
                out_dir=preview_dir,
                idx=i,
            )

            row = {
                "idx": i,
                "prediction": pred,
                "preview_path": str(preview_path),
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
            }

            if "text" in example:
                row["prompt"] = example["text"]

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            print(f"\n--- sample {i} ---")
            print(pred)
            print("preview:", preview_path)
            print("mask:", mask_path)
            print("overlay:", overlay_path)


if __name__ == "__main__":
    main()
