import numpy
import torch
from PIL import Image


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

            # Keep every segmentation mask. The model flattens all <SEG> tokens
            # across the batch and expects the same number of target masks.
            masks = []
            for ex in examples:
                mask = None
                for mask_key in (
                    "target_mask",
                    "primary_mask",
                    "mask_image",
                    "mask_images",
                    "target_masks",
                    "masks",
                ):
                    if mask_key in ex and ex[mask_key] is not None:
                        mask = ex[mask_key]
                        break

                if mask is None:
                    continue

                if isinstance(mask, list):
                    masks.extend(self._mask_to_tensor(item) for item in mask)
                else:
                    masks.append(self._mask_to_tensor(mask))

            if masks:
                batch["masks"] = torch.stack(masks, dim=0)
            return batch

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
