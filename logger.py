# logger.py
import os
import wandb
import torch

class Logger:

    def __init__(self, experiment_name, project='CityScapes', entity='vitsiozo', models_dir='models'):
        self.experiment_name = experiment_name
        self.project = project
        self.entity = entity
        self.models_dir = models_dir
        self.logger = None

    def start(self, settings):
        self.logger = wandb.init(project=self.project, entity=self.entity, name=self.experiment_name, config = settings, settings=wandb.Settings(start_method="thread"))

    def watch(self, model, log="all", log_freq=1000):
        if self.logger:
            self.logger.watch(model, log=log, log_freq=log_freq)
        else:
            print("Logger is not initialized. Call start() to initialize wandb.")

    def log(self, data, step=None):
        if self.logger:  
            self.logger.log(data, step=step)
        else:
            print("wandb logger is not initialized.")

    def log_model_artifact(self, model, model_name, metadata={}):
        if self.logger:
            try:
                model_path = os.path.join(self.models_dir, f"{model_name}.pth")
                torch.save(model.state_dict(), model_path)  
                artifact = wandb.Artifact(model_name, type='model', metadata=metadata)
                artifact.add_file(model_path)
                self.logger.log_artifact(artifact)
            except Exception as e:
                print(f"Failed to save or log model artifact: {e}")
        else:
            print("wandb logger is not initialized.")