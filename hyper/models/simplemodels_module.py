# Simple model, not used much

import torch
import torch.nn
import torch.nn as nn
import pytorch_lightning as pl
from hyper.models.model_utils import num_of_params
from hyper.training.image_logger import log_images_from_batch
from hyper.models.evaluation import evaluation_metrics_segmentation, evaluation_metrics_regression
import hyper.training.metrics as metrics

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class SimpleCNN(nn.Module):
    """
    5-layer fully conv CNN [4 with 3x3, 1 with 1x1 kernel]
    """
    def __init__(self, num_channels, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            double_conv(num_channels, 64),
            double_conv(64, 128),
            nn.Conv2d(128, num_classes, 1)
        )
        
    def forward(self, x):
        
        res = self.conv(x)
        
        return res

class SimpleModelsModule(pl.LightningModule):

    def __init__(self, settings, visualiser):
        super().__init__()
        self.settings = settings
        self.custom_config = None

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

        self.model_desc = "Baseline_model_"

    def create_network(self):
        # create a simple model ...
        variant = "cnn5"
        if variant == "cnn5":
            model = SimpleCNN(num_channels=self.num_channels, num_classes=self.num_classes)
            print("<debug, created SimpleCNN model:>")
            print(model)

        else:
            assert False, "Baseline model variant "+variant+" not implemented!"

        return model

    def summarise(self):
        params = num_of_params(self.network)
        M_params = round(params / 1000 / 1000, 2)
        print("[Model] Simple model with", str(M_params)+"M parameters.")
        print("- Input num channels =", self.num_channels, ", Output num classes=", self.num_classes)
        if self.custom_config is not None:
            print("- using custom settings:", self.custom_config)

    def training_step(self, batch, batch_idx) -> float:
        x, y = batch["x"], batch["y"]
        predictions = self.forward(x)  # (B, 1, H, W)
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
        return self.network(x)

    def printed_forward(self, x: torch.Tensor) -> torch.Tensor:
        # return self.network.printed_forward(x) # not implemented ...
        return self.forward(x)

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
        optimizer = torch.optim.Adam(self.network.parameters(), self.lr)
        return optimizer
