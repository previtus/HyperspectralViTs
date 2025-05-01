import matplotlib
from hyper.utils import check_headless
if check_headless():
    matplotlib.use('agg')
import os
os.environ["WANDB__SERVICE_WAIT"] = "300" # we were getting time outs
import hydra
import omegaconf
from omegaconf import DictConfig
from hyper.data.datamodule import HyperDataModule
from hyper.utils import compare_settings, find_latest_checkpoint_v2
from hyper.models.hyperstarcop_module import HyperStarcopModule
from hyper.models.linknet_module import LinkNetModule
from hyper.models.segtransformer_module import SegTransformerModule
from hyper.models.simplemodels_module import SimpleModelsModule
from hyper.models.efficientvit_module import EfficientViTModule
from hyper.training.trainer import TrainerHandler
from hyper.models.evaluation import evaluate_datamodule
from hyper.utils import copy_file
from timeit import default_timer as timer
import logging
from pytorch_lightning.loggers import WandbLogger

def load_from_checkpoint(load_path, settings, visualiser):
    print("Loading the model from a checkpoint!", load_path)
    if settings.model.architecture == "unet":
        model = HyperStarcopModule.load_from_checkpoint(load_path, settings=settings, visualiser=visualiser)
    elif settings.model.architecture == "linknet":
        model = LinkNetModule.load_from_checkpoint(load_path, settings=settings, visualiser=visualiser)
    elif settings.model.architecture == "segformer":
        model = SegTransformerModule.load_from_checkpoint(load_path, settings=settings, visualiser=visualiser)
    elif settings.model.architecture == "efficientvit":
        model = EfficientViTModule.load_from_checkpoint(load_path, settings=settings, visualiser=visualiser)
    elif settings.model.architecture == "simple":
        model = SimpleModelsModule.load_from_checkpoint(load_path, settings=settings, visualiser=visualiser)
    print("Succesfully loaded!")
    return model

@hydra.main(version_base=None, config_path=".", config_name="settings")
def main(settings : DictConfig) -> None:
    log = logging.getLogger(__name__)
    # We should log all the way from the start (will keep useful prints about the dataset and model)
    log.info("INITIATING LOGGERS")
    wandb_logger = WandbLogger(
        name=settings.experiment_name,
        project=settings.wandb.wandb_project,
        entity=settings.wandb.wandb_entity,
    )

    experiment_path = os.getcwd()
    settings["experiment_path"] = experiment_path
    log.info(f"Trained models will be save to: {experiment_path}")

    print(f"Hydra working directory : {os.getcwd()}")
    print(f"Hydra output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")

    # Dataset
    data_module = HyperDataModule(settings)
    data_module.prepare_data()
    data_module.summarise()
    visualiser = data_module.train_dataset.visualiser

    # Model
    load_path = settings.model.load_path
    auto_continue = settings.model.auto_continue

    # Finetune
    finetune = settings.training.finetuning
    if finetune:
        # Loading and continuing it tricky with PL
        import torch
        ckpt = torch.load(load_path)
        reached_epoch = ckpt["epoch"] + 1
        print("FINETUNING, loaded model reached epoch", reached_epoch, "adding", finetune)
        settings.training.max_epochs = reached_epoch + finetune
        print("FINETUNING reached_epoch=", ckpt["epoch"], "+1, + finetune", finetune, " => new settings.training.max_epochs = ", settings.training.max_epochs)
        # ^ Important that we load this (the finetuned model) before reloading from the "auto_continue" (for continuing with finetuning)

    if auto_continue:
        # automatically find the last checkpoint in the folder with the same name as this experiment
        # (if you can't find any then assume you are the first and start fresh ...)
        directory = os.getcwd()
        print("This run's folder:", directory)
        found, last_checkpoint = find_latest_checkpoint_v2(directory)
        if found:
            print("\n<AutoContinue> Found last checkpoint at:", last_checkpoint,"\n")
            load_path = last_checkpoint
        else:
            print("\n<AutoContinue> Failed finding last checkpoint - will start a new run!\n")

    if load_path == "":
        # normal, training from scratch
        if settings.model.architecture == "unet":
            model = HyperStarcopModule(settings, visualiser)
        elif settings.model.architecture == "linknet":
            model = LinkNetModule(settings, visualiser)
        elif settings.model.architecture == "segformer":
            model = SegTransformerModule(settings, visualiser)
        elif settings.model.architecture == "efficientvit":
            model = EfficientViTModule(settings, visualiser)
        elif settings.model.architecture == "simple":
            model = SimpleModelsModule(settings, visualiser)

        model.summarise()

    else:
        # load
        model = load_from_checkpoint(load_path, settings, visualiser)
        config_path = os.path.join(os.path.dirname(load_path), ".hydra/config.yaml")
        if os.path.exists(config_path):
            loaded_settings = omegaconf.OmegaConf.load(config_path)
            compare_settings(settings, loaded_settings)

    # Trainer handler
    trainer_handler = TrainerHandler(settings, wandb_logger)
    trainer = trainer_handler.trainer

    # Train
    start = timer()
    if load_path == "":
        trainer.fit(model, data_module)
    else:
        # pytorch lightning also needs this to properly init the optimizers
        trainer.fit(model, data_module, ckpt_path=load_path)

    end = timer()
    time = (end - start)
    print("Full training took "+str(time)+"s ("+str(time/60.0)+"min)")
    log.info(f"Finished training!")

    # Save as "final" only if we actually reached the last epoch:
    if trainer.current_epoch == trainer.max_epochs: # same as our setting in settings.training.max_epochs
        epoch_n = str(trainer.current_epoch)
        print("Reached final epoch", trainer.current_epoch, "saving final!")
        # Save model
        if not finetune:
            trainer.save_checkpoint(os.path.join(experiment_path, "final_checkpoint_model_"+epoch_n+"ep.ckpt"))
        else:
            trainer.save_checkpoint(os.path.join(experiment_path, "final_checkpoint_model_finetuned"+str(finetune)+"ep.ckpt"))

        # Backup the final versions of the models ... (makes it ready for a continued run)
        if not finetune:
            copy_file(os.path.join(experiment_path, "best_train_loss.ckpt"), os.path.join(experiment_path, "best_train_loss_"+epoch_n+"ep.ckpt"))
            if data_module.has_validation:
                copy_file(os.path.join(experiment_path, "best_val_loss.ckpt"), os.path.join(experiment_path, "best_val_loss_"+epoch_n+"ep.ckpt"))
        else:
            copy_file(os.path.join(experiment_path, "best_train_loss.ckpt"), os.path.join(experiment_path, "best_train_loss_finetuned"+str(finetune)+"ep.ckpt"))
            if data_module.has_validation:
                copy_file(os.path.join(experiment_path, "best_val_loss.ckpt"), os.path.join(experiment_path, "best_val_loss_finetuned"+str(finetune)+"ep.ckpt"))

        # Run evaluation (default - last)
        add_to_name = "_"+epoch_n+"ep"
        if finetune:
            add_to_name = "_finetuned"+str(finetune)+"ep"
        evaluate_datamodule(model, data_module, settings, plotting=False, add_to_name=add_to_name)

        # Additional evaluation on specific checkpoints
        print("Evaluate specific")
        best_train_loss_path = os.path.join(experiment_path, "best_train_loss.ckpt")
        model = load_from_checkpoint(best_train_loss_path, settings, visualiser)

        add_to_name = "_"+epoch_n+"ep_best_train_loss"
        if finetune:
            add_to_name = "_finetuned"+str(finetune)+"ep_best_train_loss"
        evaluate_datamodule(model, data_module, settings, plotting=False, add_to_name=add_to_name)

        if data_module.has_validation:
            best_val_loss_path = os.path.join(experiment_path, "best_val_loss.ckpt")
            model = load_from_checkpoint(best_val_loss_path, settings, visualiser)

            add_to_name = "_" + epoch_n + "ep_best_val_loss"
            if finetune:
                add_to_name = "_finetuned" + str(finetune) + "ep_best_val_loss"
            evaluate_datamodule(model, data_module, settings, plotting=False, add_to_name=add_to_name)

if __name__ == "__main__":
    main()

