import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from script.FolderManager import  manager
from script.HuggingfaceDownload import solve_dataset, solve_model
from script.DatatemplateEditor import Template

if __name__ == "__main__":

    # create folders
    manager()


    # download model and dataset
    model_solver, loaded_model = solve_model("geshang/Seg-R1-3B",
                                             load_in_n_bit=16,
                                             unsloth_mode=False )
    dataset_solver, dataset = solve_dataset(
        "thirdExec/synthetic-seismic-vlm",
    )


    print("model source:", model_solver.source)
    print("dataset:", dataset)


    model_solver.status_report()
    dataset_solver.status_report()

    if dataset_solver.needs_conversion:
        raise RuntimeError(dataset_solver.conversion_reason)

    model,tokenizer = loaded_model[:2]

    dataset = dataset['train']

    key_map = {
        "image": ["image_paths"],
        "text": ["instruction", "question","reason","","answer"],
    }

    key_owner = {
        "system": ["instruction"],
        "user": ["question", "image_paths"],
        "assistant": ["reason", "answer"],
    }

    template = Template(dataset=dataset, tokenizer=tokenizer, model_name="geshang/Seg-R1-3B", dataset_name="thirdExec/synthetic-seismic-vlm",
                        key_map=key_map, key_owner=key_owner,system_message="""
                        You are a seismic segmentation assistant.
                        Use all provided images as evidence.
                        Output coordinates only in the coordinate system of Picture 1, the global image.
                        Return exactly:
                        <think>...</think>
                        <bbox>[x1,y1,x2,y2]</bbox>
                """)

    print("model template:",tokenizer.chat_template)

    messages = [
        {
            ""
        }
    ]

    chat_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    )

    # print(tokenizer.decode(chat_inputs[0]))

"""
replicate training from paper first for stable pipeline. then keep training,changing architecture,changing dataset https://arxiv.org/pdf/2506.22624
"""

"""Steps would be
1. model get solve 
2. dataset get solve
3. model check and dataset check compatibility both types and template if exist
4. model and dataset alignment by template and target training
5. select training and report config
6. let it run
7. report
"""