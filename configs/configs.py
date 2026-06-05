import hydra

def load_config(config_name:str = "paths"):
    '''
    read any yaml files and return as dict
    :param config_name:
    :return: dict
    '''
    with hydra.initialize(config_path="", version_base=None):
        cfg = hydra.compose(config_name=config_name)
    return cfg


if __name__ == "__main__":
    print(load_config())
