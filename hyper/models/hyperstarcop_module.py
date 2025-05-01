# Model proposals and adaptations by Vit Ruzicka (in 2024-2025 during PhD)
# Relevant paper: https://doi.org/10.1109/JSTARS.2025.3557527
# - STARCOP models (from https://www.nature.com/articles/s41598-023-44918-6) but with more verbose forward

import torch
import torch.nn
import pytorch_lightning as pl
from hyper.models.model_utils import num_of_params
from hyper.training.image_logger import log_images_from_batch
from hyper.models.evaluation import evaluation_metrics_segmentation, evaluation_metrics_regression
import hyper.training.metrics as metrics
from hyper.models.unet_model_custom import CustomUnet
from hyper.models.siamese_unet_model import SiameseCustomUnet

class HyperStarcopModule(pl.LightningModule):

    def __init__(self, settings, visualiser):
        super().__init__()
        self.settings = settings
        self.custom_config = self.settings.model.hyperstarcop.custom_config

        self.hyperstarcop_settings = self.settings.model.hyperstarcop
        self.visualiser = visualiser
        self.wait_global_steps = self.settings.training.visualiser.wait_global_steps

        self.input_product_names = visualiser.dataset.input_product_names
        self.output_product_names = visualiser.dataset.output_product_names
        self.auxiliary_product_names = visualiser.dataset.auxiliary_product_names

        self.num_channels = len(self.input_product_names)
        self.num_classes = len(self.output_product_names)
        self.lr = self.settings.model.lr

        self.extra_metrics = self.settings.model.extra_metrics

        # segmentation / regression
        self.task = settings.model.task

        self.network = self.create_network()

        # Loss
        if self.settings.model.loss == 'BCEWithLogitsLoss':
            self.loss_name = "BCE"
            self.multiply_loss_by_mag1c = self.settings.model.multiply_loss_by_mag1c
            self.positive_weight = torch.nn.Parameter(torch.tensor(float(self.settings.model.positive_weight)), requires_grad=False)
            reduction = "none" if self.multiply_loss_by_mag1c else "mean"
            self.loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=self.positive_weight,reduction=reduction)
            #self.loss_function = torch.nn.BCEWithLogitsLoss()

        elif self.settings.model.loss == 'MAE':
            self.loss_name = "MAE"
            assert not self.settings.model.multiply_loss_by_mag1c
            self.multiply_loss_by_mag1c = False
            self.loss_function = torch.nn.L1Loss()

        elif self.settings.model.loss == 'MSE':
            self.loss_name = "MSE"
            assert not self.settings.model.multiply_loss_by_mag1c
            self.multiply_loss_by_mag1c = False
            self.loss_function = torch.nn.MSELoss()

        else:
            assert False, "Loss "+self.settings.model.loss+" not implemented!"

        self.log_train_loss_every_batch_i = self.settings.model.log_train_loss_every_batch_i # todo: set to min() if the num of batches is smaller!
        self.log_train_images_every_batch_i = self.settings.model.log_train_images_every_batch_i

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        # self.save_hyperparameters()

        # internals...
        self.validation_step_outputs = []
        self.validation_step_labels = []
        self.test_step_outputs = []
        self.test_step_labels = []

        self.model_desc = "HyperSTARCOP_UNET"

    def create_network(self):
        pretrained = None if self.hyperstarcop_settings.pretrained == 'None' else self.hyperstarcop_settings.pretrained
        activation = None if self.hyperstarcop_settings.activation == 'None' else self.hyperstarcop_settings.activation
        siamese = False if self.custom_config == "None" else self.custom_config.siamese.siamese

        if self.custom_config != "None":
            print("Using custom UNet!")
            print("custom config:", self.custom_config)

        if not siamese:
            return CustomUnet(
                encoder_name=self.hyperstarcop_settings.backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=pretrained,                              # use `imagenet` pre-trained weights for encoder initialization
                in_channels=self.num_channels,                           # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=self.num_classes,                                # model output channels (number of classes in your dataset)
                activation=activation,      # activation function, default is None
                # encoder_depth=4 # Depth parameter specify a number of downsampling operations in encoder, so you can make your model lighter if specify smaller depth
                custom_config=self.custom_config
            )

        else:
            return SiameseCustomUnet(
                encoder_name=self.hyperstarcop_settings.backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=pretrained,                              # use `imagenet` pre-trained weights for encoder initialization
                in_channels=self.num_channels,                           # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=self.num_classes,                                # model output channels (number of classes in your dataset)
                activation=activation,      # activation function, default is None
                # encoder_depth=4 # Depth parameter specify a number of downsampling operations in encoder, so you can make your model lighter if specify smaller depth
                custom_config=self.custom_config
            )

        # return smp.Unet(
        #     encoder_name=self.hyperstarcop_settings.backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        #     encoder_weights=pretrained,                              # use `imagenet` pre-trained weights for encoder initialization
        #     in_channels=self.num_channels,                           # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        #     classes=self.num_classes,                                # model output channels (number of classes in your dataset)
        #     activation=activation,      # activation function, default is None
        #     # encoder_depth=4 # Depth parameter specify a number of downsampling operations in encoder, so you can make your model lighter if specify smaller depth
        # )

    def summarise(self):
        pretrained = "(not pretrained)" if self.hyperstarcop_settings.pretrained == 'None' else "(pretrained on "+self.hyperstarcop_settings.pretrained+")"

        params = num_of_params(self.network)
        M_params = round(params / 1000 / 1000, 2)
        print("[Model] U-Net model with the", self.hyperstarcop_settings.backbone, "backbone", pretrained,
              "with", str(M_params)+"M parameters.")
        print("- Input num channels =", self.num_channels, ", Output num classes=", self.num_classes)
        if self.custom_config is not None:
            print("- using custom settings:", self.custom_config)

    def training_step(self, batch, batch_idx) -> float:
        x, y = batch["x"], batch["y"]
        predictions = self.forward(x)  # (B, 1, H, W) (predictions torch.cuda.FloatTensor, y torch.cuda.FloatTensor)
        loss = self.loss_function(predictions, y)  # (B, 1, W, H)

        if self.multiply_loss_by_mag1c:
           weight_loss = batch["weight_mag1c"] # mag1c weight
           loss = torch.mean(loss * weight_loss)

        # log loss
        if (batch_idx % self.log_train_loss_every_batch_i) == 0:
            self.log("train_loss", loss.detach(), prog_bar=True)

        # log images
        if (batch_idx % self.log_train_images_every_batch_i) == 0 and self.global_step > self.wait_global_steps:
            log_images_from_batch("train", batch, predictions, batch_idx, self)

        return loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_channels, H, W) input tensor

        Returns:
            (B, 1, H, W) prediction of the network
        """
        return self.network(x, current_epoch=self.current_epoch)

    def printed_forward(self, x: torch.Tensor) -> torch.Tensor:
        # NEW - when using the custom UNET model
        return self.network.printed_forward(x, current_epoch=self.current_epoch)

    def log(self, *args, **kwargs):
        try:
            super().log(*args, **kwargs)
        except Exception as e:
            print(f"Bug logging {e}")

    def val_step(self, batch, batch_idx: int, prefix: str = "val"):
        x, y = batch["x"], batch["y"]
        y_long = y.long()  # (B, 1, H, C)

        predictions = self.forward(x)
        loss = self.loss_function(predictions, y)

        if self.multiply_loss_by_mag1c:
            weight_loss = batch["weight_mag1c"]
            loss = torch.mean(loss * weight_loss)

        self.log(f"{prefix}_loss", loss.detach(), on_epoch=True, prog_bar=True)
        # on_epoch=True => Automatically accumulates and logs at the end of the epoch.

        # log images
        if batch_idx == 0 and self.global_step > self.wait_global_steps:
            log_images_from_batch(prefix, batch, predictions, batch_idx, self)

        del batch

        # logging for metrics at the end of the validation
        _p = predictions.detach()
        if self.task == "segmentation":
            _y = y_long.detach()
        elif self.task == "regression":
            _y = y.detach()

        if prefix == "test":
            self.test_step_outputs.append(_p)
            self.test_step_labels.append(_y)

        elif prefix == "val":
            self.validation_step_outputs.append(_p)
            self.validation_step_labels.append(_y)

    def validation_step(self, batch, batch_idx: int):
        return self.val_step(batch, batch_idx, prefix="val")

    def test_step(self, batch, batch_idx: int):
        # isn't called during training (during fit call, needs its own test call)
        return self.val_step(batch, batch_idx, prefix="test")

    def val_epoch_end(self, prefix):
        if prefix == "test":
            outputs = self.test_step_outputs
            labels = self.test_step_labels

        elif prefix == "val":
            outputs = self.validation_step_outputs
            labels = self.validation_step_labels

        # Evaluate all the accumulated labels and predictions
        all_preds = torch.cat(outputs, dim=0)
        all_y = torch.cat(labels, dim=0)

        if self.task == "segmentation":
            metric_functions = metrics.METRICS_CONFUSION_MATRIX
            for value in self.extra_metrics: metric_functions.append(value)
            metrics_dict = evaluation_metrics_segmentation(all_y, all_preds, self.device, metric_functions = metric_functions)
        elif self.task == "regression":
            metric_functions = ["MSE", "MAE"]
            for value in self.extra_metrics: metric_functions.append(value)
            metrics_dict = evaluation_metrics_regression(all_y, all_preds, self.device, metric_functions=metric_functions)
        for key in metrics_dict.keys():
            self.log(f"{prefix}_{key}", metrics_dict[key])

        # free memory
        if prefix == "test":
            self.test_step_outputs.clear()
            self.test_step_labels.clear()

        elif prefix == "val":
            self.validation_step_outputs.clear()
            self.validation_step_labels.clear()

    def on_validation_epoch_end(self) -> None:
        self.val_epoch_end(prefix="val")

    def on_test_epoch_end(self) -> None:
        self.val_epoch_end(prefix="test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), self.lr)
        return optimizer

    def legacy_load_starcop_weights(self, starcop_model_path):
        print("Loading HyperSTARCOP model weights from", starcop_model_path, "(backwards compatibility feature!)")
        # We made couple of changes ...
        # - renamed pos_weight -> positive_weight in settings
        # - no longer using few params in the model (instead they are in the data loader)
        # - Importantly! Keep the same normalisation and bands order as we had in STARCOP (so set specific_products: ['mag1c', 640, 550, 460])
        weights = torch.load(starcop_model_path)
        weights["state_dict"]["positive_weight"] = weights["state_dict"]["pos_weight"]
        unexpected = ["pos_weight", "normalizer.offsets_input", "normalizer.factors_input", "normalizer.clip_min_input", "normalizer.clip_max_input"]
        for u in unexpected:
            del weights["state_dict"][u]
        self.load_state_dict(weights["state_dict"])

        assert self.settings.dataset.input_products.specific_products == ['mag1c', 640, 550, 460]
        assert self.settings.dataset.input_products.band_ranges == []
