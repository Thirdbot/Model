import numpy
import torch


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
        from PIL import Image
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

            # minimal mask support
            masks = []
            for ex in examples:
                mask = (
                        ex.get("target_mask")
                        or ex.get("primary_mask")
                        or ex.get("mask_image")
                        or ex.get("mask_images")
                        or ex.get("target_masks")
                        or ex.get("masks")
                )

                if isinstance(mask, list):
                    mask = mask[0]
                if torch.is_tensor(mask):
                    mask = mask

                elif isinstance(mask, Image.Image):
                    # PIL PNG mask -> grayscale numpy array -> tensor
                    arr = numpy.array(mask.convert("L"))
                    mask = torch.from_numpy(arr)
                else:
                    # numpy array or other array-like object
                    arr = numpy.asarray(mask)
                    mask = torch.from_numpy(arr)

                masks.append(mask)

            if isinstance(masks, list):
                gt_mask = [
                    torch.as_tensor(m) if not torch.is_tensor(m) else m
                    for m in masks
                ]
                batch["masks"] = torch.stack(gt_mask, dim=0)
            return batch

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
