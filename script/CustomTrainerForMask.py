import torch
from peft import get_peft_model, prepare_model_for_kbit_training

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
    dataloader,
    epochs=10,
    peft_config=None,
    lr=2e-5,
    grad_accum_steps=1,
    device="cuda",
):
    # 2. if quantized custom loop, attach LoRA manually
    if peft_config is not None:
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)

    model.config.use_cache = False
    model.train()

    params = [
        p for p in list(model.parameters())
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(params, lr=lr)

    optimizer.zero_grad(set_to_none=True)
    ...
    global_step = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        for step, batch in enumerate(dataloader):
            gt_mask = batch.pop("masks")

            gt_mask = gt_mask.to(device).float()
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            outputs = model(
                gt_mask,
                **batch,
                output_hidden_states=True,
                return_dict=True,
            )

            if isinstance(outputs, dict):
                loss = outputs["loss"]
                text_loss = outputs.get("text_loss", loss)
                mask_loss = outputs.get("mask_loss", None)
            else:
                loss = outputs.loss
                text_loss = getattr(outputs, "text_loss", loss)
                mask_loss = getattr(outputs, "mask_loss", None)

            loss.backward()

            if global_step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % 10 == 0:
                print(
                    f"step={step} "
                    f"loss={loss.item() * grad_accum_steps:.4f} "
                    f"text loss={text_loss.item():.4f} "
                    f"mask loss={mask_loss.item():.4f}"
                )
            global_step += 1

    return model

def save_vlm_and_mask_decoder(
    model,
    tokenizer=None,
    processor=None,
    output_dir = None,
    extra_config=None,
):

    vlm_dir = output_dir.joinpath("vlm_lora")
    tokenizer_dir = output_dir.joinpath("tokenizer")

    # Save LoRA adapter if model is PEFT model.
    # For PEFT, this saves adapter weights, not full base model.
    model.vlm.save_pretrained(vlm_dir)

    if tokenizer is not None:
        tokenizer.save_pretrained(tokenizer_dir)

    if processor is not None:
        processor.save_pretrained(tokenizer_dir)

    # Save custom mask decoder
    torch.save(
        {
            "mask_decoder_state_dict": model.mask_decoder.state_dict(),
            "mask_decoder_class": model.mask_decoder.__class__.__name__,
            "extra_config": extra_config or {},
        },
        output_dir.joinpath("mask_decoder.pt"),
    )
    print(f"Mask decoder saved to {output_dir}")