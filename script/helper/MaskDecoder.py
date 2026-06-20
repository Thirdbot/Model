import requests
import torch
import torch.nn as nn
from PIL import Image
from transformers import Sam2Model, Sam2Processor


class MaskDecoder(nn.Module):
    def __init__(self, hidden_size, width_size=500,height_size=100):
        super().__init__()
        self.width_size = width_size
        self.height_size = height_size
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, width_size * height_size),
        )

    def forward(self, seg_hidden):
        logits = self.net(seg_hidden)
        return logits.view(seg_hidden.size(0), 1, self.height_size, self.width_size)

class SamDecoder:
    def __init__(self):
        self.model = Sam2Model.from_pretrained("facebook/sam2-hiera-base-plus")
        self.processor = Sam2Processor.from_pretrained("facebook/sam2-hiera-base-plus")

    def forward(self,image,bbox,input_labels):
        inputs = self.processor(images=image, input_points=bbox, input_labels=input_labels,
                           return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
        masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        return masks

if __name__ == "__main__":
    image_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg"
    raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    input_points = [[[[500, 375]]]]  # Single point click, 4 dimensions (image_dim, object_dim, point_per_object_dim, coordinates)
    input_labels = [[[1]]]
    mask_decoder = SamDecoder()
    masks = mask_decoder.forward(raw_image,input_points,input_labels)
    print(masks)