# Tiling smaller areas from larger geo data, while keeping repeatability
# - can do multiple tiling schemes, is reproducible and fixed across trainings

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

class TrainerHandler():

    def __init__(self, settings, wandb_logger=None):
        self.settings = settings

        experiment_path = self.settings.experiment_folder

        self.has_validation = False # Then it just uses train + test
        if settings.dataset.val_csv != "":
            self.has_validation = True # Will also use val

        self.loss_name = "loss" # by not specifying any extra name, it shows all losses in the same place in wandb
        if wandb_logger is None:
            wandb_logger = WandbLogger(
                name=settings.experiment_name,
                project=settings.wandb.wandb_project,
                entity=settings.wandb.wandb_entity,
            )
        wandb_logger.experiment.config.update(settings)

        callbacks = []

        # Add callbacks to save model checkpoints:
        # Saves last and train best:
        train_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="train_"+self.loss_name,
            mode="min",
            dirpath=".",
            filename="best_train_"+self.loss_name,
            save_last=True # should keep both last and best ... = Q: are they different? I hope so...
        )
        callbacks.append(train_callback)

        if self.has_validation:
            # Saves val best (will be evaluated on a set aside test!)
            val_callback = ModelCheckpoint(
                save_top_k=1,
                monitor="val_"+self.loss_name,
                mode="min",
                dirpath=".",
                filename="best_val_"+self.loss_name,
                save_last=False
            )
            callbacks.append(val_callback)

        self.trainer = Trainer(
            fast_dev_run=False,
            logger=wandb_logger,
            callbacks=callbacks,
            default_root_dir=experiment_path,
            accumulate_grad_batches=1,
            gradient_clip_val=0.0,
            benchmark=False,
            accelerator=settings.training.accelerator,
            devices=settings.training.devices,
            max_epochs=settings.training.max_epochs,
            log_every_n_steps=settings.training.train_log_every_n_steps, ### < care if this is not too small?
        )
