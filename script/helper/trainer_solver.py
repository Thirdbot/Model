from trl import SFTTrainer

from configs import load_config


class TrainerSolver:
    """
    Needs to solve the trainer function including SFTTrainer
    and supported multiple types of models like UnslothModel
    and HFModel and PeftLoaded Model and will be working
    with runner for tracking experiment
    """

    def __init__(self,model=None,tokenizer=None,processor=None,dataset=None,training_args=None,tracking_args=None):
        self.config = load_config("models.yaml")

        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataset = dataset
        self.training_args = training_args
        self.tracking_args = tracking_args

    def solve(self):
        pass

    def _trainer(self):
        pass
    
if __name__ == "__main__":
    pass