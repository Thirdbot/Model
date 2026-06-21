import numpy
import torch
from PIL import Image

from script.helper.special_tokens import SEG_TOKEN


class Collator:

    def __init__(self,tokenizer,processor,set_add_generation_prompt=False,dataset=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.processor = processor
        self.set_add_generation_prompt = set_add_generation_prompt
        self.text_tokenizer = self._resolve_text_tokenizer(tokenizer, processor)
        self.seg_token_id = self.text_tokenizer.convert_tokens_to_ids(SEG_TOKEN)

    def _format_messages(self, messages):
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=self.set_add_generation_prompt,
        )

    def _format_prompt_messages(self, messages):
        prompt_messages = [
            message
            for message in messages
            if message.get("role") != "assistant"
        ]
        return self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @staticmethod
    def _assistant_messages(messages):
        return [
            message
            for message in messages
            if message.get("role") == "assistant"
        ]

    def _format_assistant_messages(self, messages):
        assistant_messages = self._assistant_messages(messages)
        return "\n".join(
            str(content.get("text", ""))
            for message in assistant_messages
            for content in message.get("content", [])
            if isinstance(content, dict) and content.get("type") == "text"
        )


    def vision_language_collate(self,examples):
        texts = [self._format_messages(ex["messages"]) for ex in examples]
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
        batch["labels"] = self._assistant_only_labels(examples, batch)
        return batch

    def tasks_collate(self, examples):
        texts = [self._format_messages(ex["messages"]) for ex in examples]

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
        batch["labels"] = self._assistant_only_labels(examples, batch)

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

    def _assistant_only_labels(self, examples, batch):
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100

        prompt_texts = [
            self._format_prompt_messages(example["messages"])
            for example in examples
        ]
        prompt_inputs = self.processor(
            text=prompt_texts,
            images=self._collect_images(examples),
            padding=True,
            return_tensors="pt",
        )
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1).tolist()

        for row_idx, prompt_length in enumerate(prompt_lengths):
            labels[row_idx, :prompt_length] = -100

        return labels

    @staticmethod
    def _collect_images(examples):
        images = []
        for ex in examples:
            example_images = ex["images"]
            if not isinstance(example_images, list):
                example_images = [example_images]
            for image in example_images:
                if getattr(image, "mode", None) != "RGB":
                    image = image.convert("RGB")
                images.append(image)
        return images

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

        seg_counts = (
            batch["input_ids"].eq(self.seg_token_id)
            & batch["labels"].ne(-100)
        ).sum(dim=1).tolist()
        for idx, (seg_count, mask_count) in enumerate(zip(seg_counts, mask_counts)):
            if seg_count == mask_count:
                continue

            raw_text = examples[idx].get("text", "")
            if not raw_text and "messages" in examples[idx]:
                raw_text = self._format_assistant_messages(examples[idx]["messages"])
            raw_seg_count = raw_text.count(SEG_TOKEN)
            raise ValueError(
                "SEG/mask mismatch in collator: "
                f"sample_index={idx}, tokenized_seg_count={seg_count}, "
                f"raw_text_seg_count={raw_seg_count}, mask_count={mask_count}. "
                f"Each {SEG_TOKEN} token must map to exactly one mask. "
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
