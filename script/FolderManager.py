from contextlib import contextmanager
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs import load_config
"""
1. It needs to create the folder structure from config
2. It needs to clean up the folder structure (if allowed)
"""

@contextmanager
def create_folders(root_path, folders, clean_up=False):
    created_folders = []
    for folder in folders:
        folder_path = root_path / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        created_folders.append(folder_path)
    yield {
        "folders":created_folders
    }

@contextmanager
def folder_manager(configuration='paths'):
    settings = load_config(configuration)
    folder_to_create = settings.create
    root_path = Path(settings.root).resolve()
    with create_folders(root_path, folder_to_create) as folders:
        yield {
            "root_path":root_path,
            "folders":folders["folders"],
        }

def manager(configuration='paths'):
    with folder_manager(configuration) as context:
        print(f"folders created at {context['root_path']}")


if __name__ == "__main__":
    '''
    run with python -m script.FolderManager
    '''
    manager()
