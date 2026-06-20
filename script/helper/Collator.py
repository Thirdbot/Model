import numpy
import torch
from PIL import Image

from script.helper.special_tokens import SEG_TOKEN


class Collator:
    ASSISTANT_MARKER = "<|im_start|>assistant\n"

    def __init__(self,dataset,tokenizer,processor):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.processor = processor
        self.text_tokenizer = self._resolve_text_tokenizer(tokenizer, processor)
        self.assistant_marker_ids = self.text_tokenizer(
            self.ASSISTANT_MARKER,
            add_special_tokens=False,
        )["input_ids"]
        self.seg_token_id = self.text_tokenizer.convert_tokens_to_ids(SEG_TOKEN)

    def text_collate(self,examples):
        print("using text collate")
        texts = [ex["text"] for ex in examples]

        batch = self.text_tokenizer(texts,
                                    padding=True,
                                    truncation=True,
                                    return_tensors="pt")
        batch["labels"] = self._assistant_only_labels(batch)
        return batch

    def vision_language_collate(self,examples):
        texts = [ex["text"] for ex in examples]
        images = []
        for ex in examples:
            example_images = ex["images"]
            if not isinstance(example_images, list):
                example_images = [example_images]
            for image in example_images:
                if getattr(image, "mode", None) != "RGB":
                    image = image.convert("RGB")
                images.append(image)

        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = self._assistant_only_labels(batch)
        return batch

    def tasks_collate(self, examples):
            texts = [ex["text"] for ex in examples]

            images = []
            for ex in examples:
                example_images = ex["images"]
                if not isinstance(example_images, list):
                    example_images = [example_images]
                for image in example_images:
                    if getattr(image, "mode", None) != "RGB":
                        image = image.convert("RGB")
                    images.append(image)

            batch = self.processor(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt",

            )

            batch["labels"] = self._assistant_only_labels(batch)

            masks = []
            mask_counts = []
            for ex in examples:
                example_masks = self._extract_masks(ex)
                mask_counts.append(len(example_masks))
                masks.extend(example_masks)

            self._validate_seg_mask_alignment(examples, batch, mask_counts)

            if masks:
                batch["masks"] = torch.stack(masks, dim=0)
            return batch

    def _extract_masks(self, example):
        mask = None
        for mask_key in (
            "target_mask",
            "primary_mask",
            "mask_image",
            "mask_images",
            "target_masks",
            "masks",
        ):
            if mask_key in example and example[mask_key] is not None:
                mask = example[mask_key]
                break

        if mask is None:
            return []

        if isinstance(mask, list):
            masks = [self._mask_to_tensor(item) for item in mask]
        else:
            masks = [self._mask_to_tensor(mask)]

        return masks

    def _validate_seg_mask_alignment(self, examples, batch, mask_counts):
        if self.seg_token_id is None or self.seg_token_id < 0:
            raise ValueError(f"{SEG_TOKEN} is not registered in the tokenizer.")

        seg_counts = batch["input_ids"].eq(self.seg_token_id).sum(dim=1).tolist()
        for idx, (seg_count, mask_count) in enumerate(zip(seg_counts, mask_counts)):
            if seg_count == mask_count:
                continue

            raw_text = examples[idx].get("text", "")
            raw_seg_count = raw_text.count(SEG_TOKEN)
            assistant_pos = raw_text.find(self.ASSISTANT_MARKER)
            preview_start = assistant_pos if assistant_pos >= 0 else 0
            preview = raw_text[preview_start:preview_start + 700].replace("\n", "\\n")
            raise ValueError(
                "SEG/mask mismatch in collator: "
                f"sample_index={idx}, tokenized_seg_count={seg_count}, "
                f"raw_text_seg_count={raw_seg_count}, mask_count={mask_count}. "
                f"Each {SEG_TOKEN} token must map to exactly one mask. "
                f"text_preview={preview}"
            )

    @staticmethod
    def _mask_to_tensor(mask):
        if torch.is_tensor(mask):
            tensor = mask.detach().clone()
        elif isinstance(mask, Image.Image):
            arr = numpy.array(mask.convert("L"))
            tensor = torch.from_numpy((arr > 0).astype("float32"))
        else:
            arr = numpy.asarray(mask)
            tensor = torch.from_numpy(arr)

        tensor = tensor.float()
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.max() > 1:
            tensor = (tensor > 0).float()
        return tensor

    def _assistant_only_labels(self, batch):
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100

        for row_idx, input_ids in enumerate(batch["input_ids"].tolist()):
            assistant_start = self._find_subsequence(input_ids, self.assistant_marker_ids)
            if assistant_start is None:
                labels[row_idx, :] = -100
                continue

            answer_start = assistant_start + len(self.assistant_marker_ids)
            labels[row_idx, :answer_start] = -100

        return labels

    @staticmethod
    def _find_subsequence(values, pattern):
        if not pattern:
            return None
        pattern_length = len(pattern)
        for index in range(0, len(values) - pattern_length + 1):
            if values[index:index + pattern_length] == pattern:
                return index
        return None

    @staticmethod
    def _resolve_text_tokenizer(tokenizer, processor):
        candidates = [
            getattr(processor, "tokenizer", None),
            getattr(tokenizer, "tokenizer", None),
            tokenizer,
        ]

        for candidate in candidates:
            if candidate is None:
                continue
            if hasattr(candidate, "image_processor"):
                continue
            if hasattr(candidate, "encode") or hasattr(candidate, "batch_decode"):
                return candidate

        raise TypeError("Could not resolve a text tokenizer from tokenizer/processor.")
