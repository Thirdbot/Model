"""
Run cloud gpu
"""

from script.cloud.modal_setup import setup

APP_NAME = "GRPOModelTrainer"
app,trainer = setup(APP_NAME,main_file="grpo_trainer.py")

@app.local_entrypoint()
def main():
    trainer.remote()