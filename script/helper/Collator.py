class Collator:
    def __init__(self,dataset,tokenizer,processor):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.processor = processor

    def text_collate(self,examples):
        print("using text collate")
        texts = [ex["text"] for ex in examples]

        batch = self.tokenizer(texts,
                               padding=True,
                               truncation=True,
                               return_tensors="pt")
        batch["labels"] = batch["input_ids"].clone()
        return batch

    def vision_language_collate(self,examples):
        print("using vision language collate")
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
        batch["labels"] = batch["input_ids"].clone()
        return batch

    def tasks_collate(self):
        print("using tasks collate")
        pass
