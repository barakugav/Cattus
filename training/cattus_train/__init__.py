from .train_process import Config


def train(config: Config, run_id=None):
    from .train_process import TrainProcess

    tp = TrainProcess(config, run_id=run_id)
    tp.run_training_loop()
