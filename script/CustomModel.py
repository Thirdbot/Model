import torch
import torch.nn.functional as F


def get_seg_hidden(hidden, input_ids, seg_token_id):
    import torch

    seg_pos = input_ids.eq(seg_token_id)  # [B, L]
    has_seg = seg_pos.any(dim=1)

    if not has_seg.all():
        bad_rows = (~has_seg).nonzero(as_tuple=True)[0].tolist()
        raise ValueError(f"No <SEG> token found in rows: {bad_rows}")

    # first <SEG> per sample
    seg_idx = seg_pos.float().argmax(dim=1)
    b_idx = torch.arange(input_ids.size(0), device=input_ids.device)

    return hidden[b_idx, seg_idx]  # [B, D]

class AddModelToken:
    def __init__(self,model,tokenizer=None,processor=None):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.SEG_TOKEN = "<SEG>"
        self.real_tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.tokenizer

        added = self.real_tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.SEG_TOKEN]}
        )

        if added > 0:
            model.resize_token_embeddings(len(tokenizer))
        self.seg_token_id = tokenizer.convert_tokens_to_ids(self.SEG_TOKEN)


    def get_model(self):
        return self.model
    def get_tokenizer(self):
        return self.real_tokenizer

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

        # batch mismatch check
        if mask_logits.shape[0] != target.shape[0]:
            raise ValueError(
                f"Pred/target batch mismatch: "
                f"mask_logits={mask_logits.shape}, gt_mask={gt_mask.shape}. "
                f"Probably multiple <SEG> tokens or wrong mask batching."
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