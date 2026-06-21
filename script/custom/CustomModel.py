import torch
import torch.nn.functional as F


def get_seg_mask(input_ids, seg_token_id, labels=None):
    seg_pos = input_ids.eq(seg_token_id)  # [B, L]
    if labels is not None:
        seg_pos = seg_pos & labels.ne(-100)
    return seg_pos


def get_seg_hidden(hidden, input_ids, seg_token_id, labels=None):
    seg_pos = get_seg_mask(input_ids, seg_token_id, labels=labels)

    if not seg_pos.any():
        raise ValueError("No <SEG> token found in batch.")

    b_idx, t_idx = seg_pos.nonzero(as_tuple=True)

    return hidden[b_idx, t_idx]  # [num_seg_tokens, D]


def get_seg_counts(input_ids, seg_token_id, labels=None):
    return get_seg_mask(input_ids, seg_token_id, labels=labels).sum(dim=1).tolist()


def get_image_features(outputs, seg_counts, num_seg_tokens):
    image_features = getattr(outputs, "image_hidden_states", None)
    if image_features is None and isinstance(outputs, dict):
        image_features = outputs.get("image_hidden_states", None)
    if isinstance(image_features, (list, tuple)):
        image_features = image_features[-1] if image_features else None
    if image_features is None or not torch.is_tensor(image_features):
        return None

    if image_features.ndim == 3:
        batch_or_images, patches, channels = image_features.shape
        height = int(patches ** 0.5)
        height = max(height, 1)
        width = (patches + height - 1) // height
        pad = height * width - patches
        if pad:
            image_features = F.pad(image_features, (0, 0, 0, pad))
        image_features = image_features.transpose(1, 2).reshape(
            batch_or_images,
            channels,
            height,
            width,
        )
    elif image_features.ndim != 4:
        return None

    if image_features.shape[0] == num_seg_tokens:
        return image_features
    if image_features.shape[0] == len(seg_counts):
        return image_features.repeat_interleave(
            torch.tensor(seg_counts, device=image_features.device),
            dim=0,
        )
    if image_features.shape[0] == 1:
        return image_features.expand(num_seg_tokens, -1, -1, -1)

    return image_features.mean(dim=0, keepdim=True).expand(num_seg_tokens, -1, -1, -1)


class VLMWithMaskDecoder(torch.nn.Module):
    def __init__(
        self,
        vlm,
        mask_decoder,
        seg_token_id,
        lambda_mask=1.0,
        bce_weight=1.0,
        dice_weight=1.0,
        text_loss_weight=1.0,
    ):
        super().__init__()
        self.vlm = vlm
        self.mask_decoder = mask_decoder
        self.seg_token_id = seg_token_id
        self.lambda_mask = lambda_mask
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.text_loss_weight = text_loss_weight
        self.config = vlm.config
        self.generation_config = getattr(vlm, "generation_config", None)

    def forward(self, mask=None, **batch):
        # remove custom / unwanted keys
        batch.pop("masks", None)
        batch.pop("mask", None)
        batch.pop("num_items_in_batch", None)

        # remove duplicate keys that you set manually
        batch.pop("output_hidden_states", None)
        batch.pop("return_dict", None)

        outputs = self.vlm(
            **{k: v for k, v in batch.items() if k != "masks"},
            output_hidden_states=True,
            return_dict=True,
        )

        text_loss = outputs.loss

        hidden = outputs.hidden_states[-1]
        labels = batch.get("labels")
        seg_pos = get_seg_mask(batch["input_ids"], self.seg_token_id, labels=labels)

        if seg_pos.sum().item() == 0:
            raise ValueError("No <SEG> token found in batch.")

        seg_counts = get_seg_counts(batch["input_ids"], self.seg_token_id, labels=labels)
        seg_hidden = get_seg_hidden(
            hidden,
            batch["input_ids"],
            self.seg_token_id,
            labels=labels,
        )
        decoder_param = next(self.mask_decoder.parameters())

        seg_hidden = seg_hidden.to(
            device=decoder_param.device,
            dtype=decoder_param.dtype,
        )

        image_features = get_image_features(
            outputs=outputs,
            seg_counts=seg_counts,
            num_seg_tokens=seg_hidden.shape[0],
        )
        if image_features is not None:
            image_features = image_features.to(
                device=decoder_param.device,
                dtype=decoder_param.dtype,
            )

        mask_logits = self.mask_decoder(seg_hidden, image_features=image_features)

        if mask is None:
            return {
                "loss": text_loss,
                "mask_logits": mask_logits,
            }

        target = mask.float()

        if mask_logits.ndim == 3:
            mask_logits = mask_logits.unsqueeze(1)

        target = target.to(mask_logits.device).float()

        if target.ndim == 2:
            target = target.unsqueeze(0).unsqueeze(0)
        elif target.ndim == 3:
            target = target.unsqueeze(1)

        if target.max() > 1:
            target = (target > 0).float()

        if mask_logits.shape[0] != target.shape[0]:
            raise ValueError(
                f"Pred/target batch mismatch: "
                f"mask_logits={mask_logits.shape}, target={target.shape}. "
                f"Each <SEG> token must have exactly one target mask."
            )

        # CRITICAL FIX: resize target to decoder output size
        if target.shape[-2:] != mask_logits.shape[-2:]:
            target = F.interpolate(
                target,
                size=mask_logits.shape[-2:],
                mode="nearest",
            )
        positive = target.sum()
        negative = target.numel() - positive
        pos_weight = (negative / positive.clamp_min(1.0)).clamp(max=100.0)
        weighted_bce = F.binary_cross_entropy_with_logits(
            mask_logits,
            target,
            pos_weight=pos_weight.detach(),
        )

        pred = torch.sigmoid(mask_logits)
        dice = 1 - (2 * (pred * target).sum() + 1) / (
            pred.sum() + target.sum() + 1
        )

        mask_loss = self.bce_weight * weighted_bce + self.dice_weight * dice
        loss = self.text_loss_weight * text_loss + self.lambda_mask * mask_loss

        return {
            "loss": loss,
            "text_loss": text_loss,
            "text_loss_weight": self.text_loss_weight,
            "mask_loss": mask_loss,
            "weighted_bce_loss": weighted_bce,
            "dice_loss": dice,
            "lambda_mask": self.lambda_mask,
            "bce_weight": self.bce_weight,
            "dice_weight": self.dice_weight,
            "mask_logits": mask_logits,
        }
