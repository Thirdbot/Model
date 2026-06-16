import torch
import torch.nn.functional as F
from peft import get_peft_model, prepare_model_for_kbit_training
from CustomModel import VLMWithMaskDecoder

def dice_loss_from_logits(logits, target, eps=1.0):
    pred = torch.sigmoid(logits)
    target = target.float()

    pred = pred.flatten(1)
    target = target.flatten(1)

    inter = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)

    return 1.0 - ((2 * inter + eps) / (denom + eps)).mean()


def get_seg_hidden(hidden, input_ids, seg_token_id):
    # hidden: [B, T, D]
    # input_ids: [B, T]
    seg_mask = input_ids.eq(seg_token_id)

    if not seg_mask.any():
        raise ValueError("No <SEG> token found in batch.")

    # require exactly one <SEG> per sample
    if not torch.all(seg_mask.sum(dim=1) == 1):
        raise ValueError("Each sample must contain exactly one <SEG> token.")

    b_idx, t_idx = seg_mask.nonzero(as_tuple=True)
    return hidden[b_idx, t_idx]  # [B, D]


def train_mask_decoder_loop(
    model,
    tokenizer,
    token_id,
    dataloader,
    mask_decoder,
    peft_config=None,
    lr=2e-5,
    mask_weight=1.0,
    grad_accum_steps=1,
    device="cuda",
):
    # add token to model
    model,tokenizer = AddModelToken(model,tokenizer)
    # 2. if quantized custom loop, attach LoRA manually
    if peft_config is not None:
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)

    model.config.use_cache = False
    model.train()
    mask_decoder.train()

    # 3. train only LoRA + mask decoder
    for p in model.parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if "lora" in name.lower():
            p.requires_grad = True

    for p in mask_decoder.parameters():
        p.requires_grad = True

    params = [
        p for p in list(model.parameters()) + list(mask_decoder.parameters())
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(params, lr=lr)

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(dataloader):
        gt_mask = batch.pop("masks").to(device).float()  # [B, 1, H, W]

        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        outputs = model(
            **batch,
            output_hidden_states=True,
            return_dict=True,
        )

        text_loss = outputs.loss

        hidden = outputs.hidden_states[-1]
        seg_hidden = get_seg_hidden(
            hidden=hidden,
            input_ids=batch["input_ids"],
            seg_token_id=token_id,
        )

        # Your mask decoder decides this API.
        # Minimal expected output: [B, 1, H, W]
        mask_logits = mask_decoder(seg_hidden)

        if mask_logits.shape[-2:] != gt_mask.shape[-2:]:
            mask_logits = F.interpolate(
                mask_logits,
                size=gt_mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        bce = F.binary_cross_entropy_with_logits(mask_logits, gt_mask)
        dice = dice_loss_from_logits(mask_logits, gt_mask)
        mask_loss = bce + dice

        loss = text_loss + mask_weight * mask_loss
        loss = loss / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % 10 == 0:
            print(
                f"step={step} "
                f"loss={loss.item() * grad_accum_steps:.4f} "
                f"text={text_loss.item():.4f} "
                f"mask={mask_loss.item():.4f}"
            )

    return model, mask_decoder


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from script.HuggingfaceDownload import solve_model, solve_dataset
    from script.DatatemplateEditor import Template
    from script.helper.Collator import Collator
    from script.helper.MaskDecoder import MaskDecoder
    from script.CustomModel import AddModelToken, VLMWithMaskDecoder

    model_solver, loaded_model = solve_model(
        "geshang/Seg-R1-3B",
        load_in_n_bit=4,
        unsloth_mode=False,
    )

    model, tokenizer = loaded_model[:2]
    processor = loaded_model[-1] if len(loaded_model) == 3 else None

    # Add <SEG>
    token_helper = AddModelToken(model, tokenizer=tokenizer, processor=processor)
    model = token_helper.get_model()
    tokenizer = token_helper.get_tokenizer()
    seg_token_id = token_helper.seg_token_id

    dataset_solver, dataset = solve_dataset("thirdExec/synthetic-seismic-vlm")
    dataset = dataset["train"]

    key_map = {
        "image": ["images", "mask_images"],
        "text": ["thinking", "problem", "solution"],
    }

    key_owner = {
        "system": ["system_prompt"],
        "user": ["problem", "images"],
        "assistant": ["thinking", "solution"],
    }

    template = Template(
        dataset=dataset,
        tokenizer=tokenizer,
        model_name="geshang/Seg-R1-3B",
        dataset_name="thirdExec/synthetic-seismic-vlm",
        key_map=key_map,
        key_owner=key_owner,
        is_output_mask=True,
        temp_for="sft",
    )

    train_dataset, eval_dataset, test_dataset = template.solve()

    collator = Collator(
        dataset=dataset,
        tokenizer=tokenizer,
        processor=processor,
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collator.tasks_collate,
    )

    hidden_size = model.config.hidden_size
    mask_decoder = MaskDecoder(
        hidden_size=hidden_size,
        output_size=256,
    ).cuda()

    custom_model = VLMWithMaskDecoder(
        vlm=model,
        mask_decoder=mask_decoder,
        seg_token_id=seg_token_id,
    )

    train_mask_decoder_loop(
        model=custom_model,
        tokenizer=tokenizer,
        token_id=seg_token_id,
        dataloader=dataloader,
        mask_decoder=mask_decoder,
        peft_config=model_solver.peft_config,
        device="cuda",
    )