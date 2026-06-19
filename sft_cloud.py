"""
Run cloud gpu
"""

from script.cloud.modal_setup import setup

APP_NAME = "SFTModelTrainer"
app,trainer = setup(APP_NAME,main_file="sft_trainer.py")

@app.local_entrypoint()
def main():
    trainer.remote()