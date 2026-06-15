class Collator:
    ASSISTANT_MARKER = "<|im_start|>assistant\n"

    def __init__(self,dataset,tokenizer,processor):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.processor = processor
        self.text_tokenizer = getattr(processor, "tokenizer", None) or tokenizer
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

    def tasks_collate(self):
        print("using tasks collate")
        pass

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
