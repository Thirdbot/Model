from datasets import load_dataset
from PIL import Image
import random
import numpy
import torch
if __name__ == "__main__":
    dataset = load_dataset("thirdExec/synthetic-seismic-vlm")


    print(f"columns:{dataset.column_names}")

    for split in dataset.keys():
        dataset = dataset[split]
        row = random.randint(0, len(dataset)-1)
        selected_dataset = dataset[row]
        print(f"split:{split} at row:{row}")
        for col in dataset.column_names:
            if not isinstance(selected_dataset[col],Image.Image):
                print(f"{col}: {selected_dataset[col]}")
            else:
                image = selected_dataset[col]
                image.show()