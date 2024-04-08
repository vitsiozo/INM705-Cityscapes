# logger.py
import wandb
import torch

class Logger:

    def __init__(self, experiment_name, project='CityScapes', entity='vitsiozo'):
        self.experiment_name = experiment_name
        self.project = project
        self.entity = entity
        self.logger = None

    def start(self):
        self.logger = wandb.init(project=self.project, entity=self.entity, name=self.experiment_name, settings=wandb.Settings(start_method="thread"))

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
            artifact = wandb.Artifact(model_name, type='model', metadata=metadata)
            model_path = f"{model_name}.pth"
            torch.save(model.state_dict(), model_path)  
            artifact.add_file(model_path)
            self.logger.log_artifact(artifact)
        else:
            print("wandb logger is not initialized.")