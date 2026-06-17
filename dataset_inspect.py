from datasets import load_dataset
from PIL import Image
import random
if __name__ == "__main__":
    dataset = load_dataset("thirdExec/synthetic-seismic-vlm")


    print(f"columns:{dataset.column_names}")

    for split in dataset.keys():
        dataset = dataset[split]
        row = random.randint(0, len(dataset))
        selected_dataset = dataset[row]
        print(f"split:{split} at row:{row}")
        for col in dataset.column_names:
            if not isinstance(selected_dataset[col],Image.Image):
                print(f"{col}: {selected_dataset[col]}")
            else:
                selected_dataset[col].show(title=col)
