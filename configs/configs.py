import hydra

def load_config(config_name:str = "Paths"):
    with hydra.initialize(config_path="", version_base=None):
        cfg = hydra.compose(config_name=config_name)
    return cfg


if __name__ == "__main__":
    print(load_config())