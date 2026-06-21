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

    b_idx, t_idx = seg_mask.nonzero(as_tuple=True)
    return hidden[b_idx, t_idx]  # [num_seg_tokens, D]


def train_mask_decoder_loop(
    model,
    dataloader,
    epochs=10,
    peft_config=None,
    lr=2e-5,
    mask_lr=None,
    vlm_lr=None,
    freeze_vlm=False,
    grad_accum_steps=1,
    device="cuda",
    wandb_logger=None,
):
    # Attach LoRA to the VLM only. The mask decoder remains a normal trainable
    # module, and model.vlm.save_pretrained(...) writes adapter_config.json.
    if peft_config is not None:
        model.vlm = prepare_model_for_kbit_training(model.vlm)
        model.vlm = get_peft_model(model.vlm, peft_config)

    if freeze_vlm:
        model.vlm.eval()
        for param in model.vlm.parameters():
            param.requires_grad = False

    model.config.use_cache = False
    model.train()
    if freeze_vlm:
        model.vlm.eval()
        model.mask_decoder.train()

    mask_lr = mask_lr or lr
    vlm_lr = vlm_lr or lr
    param_groups = []

    mask_params = [
        p for p in model.mask_decoder.parameters()
        if p.requires_grad
    ]
    if mask_params:
        param_groups.append({"params": mask_params, "lr": mask_lr})

    vlm_params = [
        p for p in model.vlm.parameters()
        if p.requires_grad
    ]
    if vlm_params:
        param_groups.append({"params": vlm_params, "lr": vlm_lr})

    if not param_groups:
        raise ValueError("No trainable parameters found for custom mask training.")

    params = [
        param
        for group in param_groups
        for param in group["params"]
    ]

    print(
        "custom train config: "
        f"freeze_vlm={freeze_vlm} "
        f"mask_lr={mask_lr} "
        f"vlm_lr={vlm_lr} "
        f"trainable_mask_params={sum(p.numel() for p in mask_params)} "
        f"trainable_vlm_params={sum(p.numel() for p in vlm_params)}"
    )

    optimizer = torch.optim.AdamW(param_groups)

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
                weighted_bce_loss = outputs.get("weighted_bce_loss", None)
                dice_loss = outputs.get("dice_loss", None)
            else:
                loss = outputs.loss
                text_loss = getattr(outputs, "text_loss", loss)
                mask_loss = getattr(outputs, "mask_loss", None)
                weighted_bce_loss = getattr(outputs, "weighted_bce_loss", None)
                dice_loss = getattr(outputs, "dice_loss", None)

            (loss / grad_accum_steps).backward()

            if (global_step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % 10 == 0:
                loss_value = loss.item()
                text_loss_value = text_loss.item()
                mask_loss_value = (
                    mask_loss.item()
                    if mask_loss is not None
                    else float("nan")
                )
                weighted_bce_value = (
                    weighted_bce_loss.item()
                    if weighted_bce_loss is not None
                    else float("nan")
                )
                dice_value = (
                    dice_loss.item()
                    if dice_loss is not None
                    else float("nan")
                )
                print(
                    f"global_step={global_step} "
                    f"batch_step={step} "
                    f"loss={loss_value:.4f} "
                    f"text loss={text_loss_value:.4f} "
                    f"mask loss={mask_loss_value:.4f} "
                    f"weighted bce={weighted_bce_value:.4f} "
                    f"dice={dice_value:.4f}"
                )
                if wandb_logger is not None and wandb_logger.run is not None:
                    text_loss_weight = (
                        outputs.get("text_loss_weight", None)
                        if isinstance(outputs, dict)
                        else getattr(outputs, "text_loss_weight", None)
                    )
                    wandb_logger.run.log(
                        {
                            "train/loss": loss_value,
                            "train/text_loss": text_loss_value,
                            "train/text_loss_weight": (
                                text_loss_weight
                                if text_loss_weight is not None
                                else float("nan")
                            ),
                            "train/mask_loss": mask_loss_value,
                            "train/weighted_bce_loss": weighted_bce_value,
                            "train/dice_loss": dice_value,
                            "train/mask_lr": mask_lr,
                            "train/vlm_lr": vlm_lr,
                            "train/freeze_vlm": float(freeze_vlm),
                            "train/epoch": epoch + 1,
                            "train/global_step": global_step,
                            "train/batch_step": step,
                        },
                        step=global_step,
                    )
            global_step += 1

        if global_step % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return model

def save_vlm_and_mask_decoder(
    model,
    tokenizer=None,
    processor=None,
    output_dir = None,
    extra_config=None,
):

    output_dir.mkdir(parents=True, exist_ok=True)
    vlm_dir = output_dir.joinpath("vlm_lora")
    tokenizer_dir = output_dir.joinpath("tokenizer")
    vlm_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

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
            "mask_decoder_config": {
                "width_size": getattr(model.mask_decoder, "width_size", None),
                "height_size": getattr(model.mask_decoder, "height_size", None),
                "feature_channels": getattr(model.mask_decoder, "feature_channels", None),
                "feature_height": getattr(model.mask_decoder, "feature_height", None),
                "feature_width": getattr(model.mask_decoder, "feature_width", None),
            },
            "extra_config": extra_config or {},
        },
        output_dir.joinpath("mask_decoder.pt"),
    )
    print(f"Mask decoder saved to {output_dir}")
