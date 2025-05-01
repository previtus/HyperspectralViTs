# Model proposals and adaptations by Vit Ruzicka (in 2024-2025 during PhD)
# Relevant paper: https://doi.org/10.1109/JSTARS.2025.3557527

import numpy as np
import torch.nn
import pytorch_lightning as pl

from hyper.models.model_utils import num_of_params
from hyper.training.image_logger import log_images_from_batch
from hyper.models.evaluation import evaluation_metrics_segmentation, evaluation_metrics_regression
import hyper.training.metrics as metrics

import torch
from torch import nn

from hyper.models.efficientvit_standalone import create_seg_model

# Adaptation of standalone code for EfficientViT model as a PyTorch Lightning module
class EfficientViTModule(pl.LightningModule):

    def __init__(self, settings, visualiser):
        super().__init__()
        self.settings = settings
        self.visualiser = visualiser
        self.wait_global_steps = self.settings.training.visualiser.wait_global_steps

        self.input_product_names = visualiser.dataset.input_product_names
        self.output_product_names = visualiser.dataset.output_product_names
        self.auxiliary_product_names = visualiser.dataset.auxiliary_product_names

        self.num_channels = len(self.input_product_names)
        self.num_classes = len(self.output_product_names)

        # Add extra check for multiple labels
        try:
            # try catch for compatability with older ...
            if settings.model.efficientvit.custom_config.loss_overrides.multilabel_override:
                self.num_classes = settings.model.efficientvit.custom_config.loss_overrides.num_classes
                print("setting self.num_classes to", self.num_classes)
        except:
            print("failed accessing settings in (.custom_config.loss_overrides):", settings.model.efficientvit)
            pass
        print("EffViT debug, num_classes=", self.num_classes)

        # LR: we used mostly 0.00006 in SegFormer
        #     official EffViT seems to use base_lr: 0.00025 for imagenet classification...
        #     exact settings used for semantic segmentation training hasn't been released
        #     However paper states:
        #     - We use the AdamW optimizer with cosine learning rate decay for training our models

        self.lr = self.settings.model.lr # suggested: 0.00006 as in SegFormer
        self.extra_metrics = self.settings.model.extra_metrics

        self.model_version = self.settings.model.efficientvit.backbone
        self.pretrained = False

        # segmentation / regression
        self.task = settings.model.task
        self.network = self.create_network(self.num_classes)

        self.multiply_loss_by_mag1c = self.settings.model.multiply_loss_by_mag1c

        # Loss
        # > most likely BCE ...
        self.loss_name = "loss" # same logs etc...
        if self.settings.model.loss == 'BCEWithLogitsLoss':
            self.positive_weight = torch.nn.Parameter(torch.tensor(float(self.settings.model.positive_weight)), requires_grad=False)
            reduction = "none" if self.multiply_loss_by_mag1c else "mean"
            self.loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=self.positive_weight,reduction=reduction)

            print("EffViT debug, Loss= BCEWithLogitsLoss with pos_weight=", self.positive_weight, "reduction=", reduction)
        
        elif self.settings.model.loss == 'MAE':
            # self.loss_name = "MAE"
            if self.multiply_loss_by_mag1c:
                self.loss_function = torch.nn.L1Loss(reduction="none")
            else:
                self.loss_function = torch.nn.L1Loss(reduction="mean")

        elif self.settings.model.loss == 'MSE':
            # self.loss_name = "MSE"
            if self.multiply_loss_by_mag1c:
                self.loss_function = torch.nn.MSELoss(reduction="none")
            else:
                self.loss_function = torch.nn.MSELoss(reduction="mean")
        else:
            assert False, "Loss "+self.settings.model.loss+" not implemented!"

        self.log_train_loss_every_batch_i = self.settings.model.log_train_loss_every_batch_i
        self.log_train_images_every_batch_i = self.settings.model.log_train_images_every_batch_i

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        # self.save_hyperparameters()

        # internals...
        self.validation_step_outputs = []
        self.validation_step_labels = []
        self.test_step_outputs = []
        self.test_step_labels = []

        self.model_desc = "HyperEfficientViT"


    def create_network(self, num_classes):
        print("Using standard EfficientViT (not pre-trained)")

        num_channels = len(self.input_product_names)  # 86
        custom_config = self.settings.model.efficientvit.custom_config
        model = create_seg_model(name = self.model_version, verbose=True, in_channels=num_channels, n_classes=num_classes,
                                 custom_config = custom_config
                                 )
        self.num_channels = num_channels
        self.num_classes = num_classes
        return model

    def summarise(self):
        pretrained = "(not pretrained)" if not self.pretrained else "(pretrained)"

        params = num_of_params(self.network)
        M_params = round(params / 1000 / 1000, 2)
        print("[Model] EfficientViT model version", self.model_version, pretrained,
              "with", str(M_params)+"M parameters.")
        print("- Input num channels =", self.num_channels, ", Output num classes=", self.num_classes)
        if self.settings.model.efficientvit.custom_config is not None:
            print("- using custom settings:", self.settings.model.efficientvit.custom_config)
        print("(Exact params =", params,")")

    def training_step(self, batch, batch_idx) -> float:
        x, y = batch["x"], batch["y"]
        predictions = self.forward(x)  # (B, 1, H, W)
        loss = self.loss_function(predictions, y)  # (B, 1, W, H)

        if self.multiply_loss_by_mag1c:
            weight_loss = batch["weight_mag1c"]  # mag1c weight
            loss = torch.mean(loss * weight_loss)

        # log loss
        if (batch_idx % self.log_train_loss_every_batch_i) == 0:
            self.log(f"train_loss", loss.detach(), prog_bar=True)

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
        reduced_res = self.network(x)
        # same as in SegFormer implementation ...
        upsampled_output = nn.functional.interpolate(
            reduced_res, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return upsampled_output

    def printed_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network.printed_forward(x)

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
        optimizer = torch.optim.AdamW(self.network.parameters(), self.lr)
        return optimizer
