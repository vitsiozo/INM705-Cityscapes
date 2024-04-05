# logger.py
import wandb

class Logger:

    def __init__(self, experiment_name, project='CityScapes', entity='vitsiozo'):
        self.experiment_name = experiment_name
        self.project = project
        self.entity = entity
        self.logger = None

    def start(self):
        self.logger = wandb.init(project=self.project, entity=self.entity, name=self.experiment_name, settings=wandb.Settings(start_method="thread"))
    
    def log(self, data, step=None):
        if self.logger:
            self.logger.log(data, step=step)
        else:
            print("Logger is not started.")

    def watch(self, model, log="all", log_freq=1000):
        if self.logger:
            self.logger.watch(model, log=log, log_freq=log_freq)
        else:
            print("Logger is not initialized. Call start() to initialize wandb.")
