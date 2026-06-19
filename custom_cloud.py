"""
train custom on cloud
"""
from script.cloud.modal_setup import setup

APP_NAME = "CUSTOMModelTrainer"
app,trainer = setup(APP_NAME,main_file="custom_trainer.py")

@app.local_entrypoint()
def main():
    trainer.remote()