import torch
import torch.nn.functional as F


def get_seg_hidden(hidden, input_ids, seg_token_id):
    seg_pos = input_ids.eq(seg_token_id)  # [B, L]

    if not seg_pos.any():
        raise ValueError("No <SEG> token found in batch.")

    b_idx, t_idx = seg_pos.nonzero(as_tuple=True)

    return hidden[b_idx, t_idx]  # [num_seg_tokens, D]


class VLMWithMaskDecoder(torch.nn.Module):
    def __init__(self, vlm, mask_decoder, seg_token_id):
        super().__init__()
        self.vlm = vlm
        self.mask_decoder = mask_decoder
        self.seg_token_id = seg_token_id
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
        seg_pos = batch["input_ids"].eq(self.seg_token_id)

        if seg_pos.sum().item() == 0:
            raise ValueError("No <SEG> token found in batch.")

        seg_hidden = get_seg_hidden(hidden, batch["input_ids"], self.seg_token_id)
        decoder_param = next(self.mask_decoder.parameters())

        seg_hidden = seg_hidden.to(
            device=decoder_param.device,
            dtype=decoder_param.dtype,
        )
        
        mask_logits = self.mask_decoder(seg_hidden)

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
        bce = F.binary_cross_entropy_with_logits(mask_logits, target)

        pred = torch.sigmoid(mask_logits)
        dice = 1 - (2 * (pred * target).sum() + 1) / (
            pred.sum() + target.sum() + 1
        )

        loss = text_loss + bce + dice

        return {
            "loss": loss,
            "text_loss": text_loss,
            "mask_loss": bce + dice,
            "mask_logits": mask_logits,
        }
