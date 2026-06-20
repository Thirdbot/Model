"""
Model Loader and Data Loader for inference
"""
import modal

from script.helper.model_solver import ModelSolver
from configs.configs import load_config
from pathlib import Path

root = Path(load_config("paths")['root']).resolve()
save_dir = root / load_config("paths")['dirs']['saves']

def find_models():
    save_types = list(save_dir.iterdir())
    for save_type in save_types: # sft, grpo, custom_sft
        models = list(save_type.iterdir())
        for model in models: # modal_name and probably dataset_name
            print(model)

class ModelLoader:
    def __init__(self,dataset_repo_id,model_repo_id_or_path,resume_model_type='sft'):
        self.model_repo_id_or_path = model_repo_id_or_path
        model_solver = ModelSolver(model_repo_id_or_path)
        model, processor = model_solver.load_save_model(
            at_dataset=dataset_repo_id,
            method=resume_model_type,
        ) # find existing save

if __name__ == '__main__':
    available_types = ['sft', 'grpo', 'custom_sft']
    find_models()
    model_repo_id_or_path = ""
    dataset_repo_id = ""
    select_model = save_dir.joinpath(available_types[0] , model_repo_id_or_path , dataset_repo_id)
