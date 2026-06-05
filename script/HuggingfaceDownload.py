from script.helper.data_solver import DataSolver
from script.helper.model_solver import ModelSolver

"""
It needs to handle the download of the model and dataset from huggingface hub
"""

def solve_model(model_repo_id_or_path, load_in_n_bit=4, unsloth_mode=True):
    solver = ModelSolver(model_repo_id_or_path, load_in_n_bit=load_in_n_bit, unsloth_mode=unsloth_mode)
    loaded = solver.solve()
    return solver, loaded

def solve_dataset(dataset_repo_id_or_path):
    solver = DataSolver(dataset_repo_id_or_path)
    dataset = solver.solve()
    return solver, dataset

def download_dataset(dataset_repo_id_or_path):
    solver, dataset = solve_dataset(dataset_repo_id_or_path)
    return dataset

def download_hf_dataset(dataset_repo_id):

    return download_dataset(dataset_repo_id)
