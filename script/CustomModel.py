import torch
import torch.nn.functional as F


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

    def forward(self, mask=None, **batch):
        outputs = self.vlm(
            **{k: v for k, v in batch.items() if k != "mask"},
            output_hidden_states=True,
            return_dict=True,
        )

        text_loss = outputs.loss

        hidden = outputs.hidden_states[-1]
        seg_pos = batch["input_ids"].eq(self.seg_token_id)

        if seg_pos.sum().item() == 0:
            raise ValueError("No <SEG> token found in batch.")

        seg_hidden = hidden[seg_pos]

        mask_logits = self.mask_decoder(seg_hidden)

        if mask is None:
            return {
                "loss": text_loss,
                "mask_logits": mask_logits,
            }

        target = mask.float()

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