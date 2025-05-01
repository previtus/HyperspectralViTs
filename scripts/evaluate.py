import matplotlib
from hyper.utils import check_headless
if check_headless():
    matplotlib.use('agg')
import torch
import hydra
from omegaconf import DictConfig
from hyper.data.datamodule import HyperDataModule
from hyper.models.hyperstarcop_module import HyperStarcopModule
from hyper.models.linknet_module import LinkNetModule
from hyper.models.segtransformer_module import SegTransformerModule
from hyper.models.simplemodels_module import SimpleModelsModule
from hyper.models.baselinemagic_module import BaselineMagicModule
from hyper.models.efficientvit_module import EfficientViTModule
from hyper.models.evaluation import evaluate_datamodule

@hydra.main(version_base=None, config_path=".", config_name="settings")
def main(settings : DictConfig) -> None:
    # Dataset
    data_module = HyperDataModule(settings)
    data_module.prepare_data()
    data_module.summarise()
    visualiser = data_module.train_dataset.visualiser
    load_path = settings.model.load_path

    plotting = settings.evaluation.plot_save or settings.evaluation.plot_show

    if load_path == "":
        print("Provided model.load_path is empty!")
        assert False

    # Model
    print("Loading model from:", load_path)
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
    elif settings.model.architecture == "baseline_magic":
        model = BaselineMagicModule(settings=settings, visualiser=visualiser, mag1c_band=0)

    model.summarise()

    # Device (gpu or cpu)
    accelerator = settings.training.accelerator
    if accelerator == "gpu":
        device = torch.device("cuda")
        model.to(device)
    elif accelerator == "cpu":
        device = torch.device("cpu")
        model.to(device)

    model.eval() # !

    print("Model device is set to:", model.device)

    # Evaluation
    evaluate_datamodule(model, data_module, settings, plotting=plotting)


if __name__ == "__main__":
    main()
