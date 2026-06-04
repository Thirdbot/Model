from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path
from configs import load_config
"""
1. It needs to create the folder structure from config
2. It needs to clean up the folder structure (if allowed)
"""

def load_config_to_path(root_path,config):
    folder = [Path(root_path) / Path(value.name) for key,value in config.items()]
    sub_folder = [Path(root_path) / Path(value.name) / Path(value.sub_name) for key,value in config.items()]
    return folder,sub_folder

@contextmanager
def folder_manager(folders,clean_up=False):

    created_folder = []
    # creation
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        created_folder.append(folder)
    yield {"folders":folders}

    if clean_up:
        for folder in created_folder:
            with suppress(OSError):
                # remove if allowed
                folder.rmdir() # remove only empty folder

@contextmanager
def file_manager(root_path,config_name):
    folders_dict = load_config(config_name)
    folder,sub_folder = load_config_to_path(root_path,folders_dict)
    with ExitStack() as stack:
        folder = folder_manager(folder)
        sub_folder = folder_manager(sub_folder) # clean up this sub-folder
        yield {
            "root_path":root_path,
            "folders":stack.enter_context(folder)["folders"],
            "sub_folders":stack.enter_context(sub_folder)["folders"]
        }

def manager(root_path:Path):
    with file_manager(root_path,"Paths") as f_context:
        print(
            f"folder created:{f_context['sub_folders']} at {f_context['root_path']}")
