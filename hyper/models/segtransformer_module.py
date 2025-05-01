# Model proposals and adaptations by Vit Ruzicka (in 2024-2025 during PhD)
# Relevant paper: https://doi.org/10.1109/JSTARS.2025.3557527

import numpy as np
import torch.nn
import pytorch_lightning as pl

from hyper.models.model_utils import num_of_params
from hyper.training.image_logger import log_images_from_batch
from hyper.models.evaluation import evaluation_metrics_segmentation, evaluation_metrics_regression
import hyper.training.metrics as metrics
from transformers import SegformerForSemanticSegmentation
import torch
from torch import nn

# Source tutorials:
# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SegFormer/Fine_tune_SegFormer_on_custom_dataset.ipynb
# Paper: "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"
# https://arxiv.org/abs/2105.15203

class SegTransformerModule(pl.LightningModule):

    def __init__(self, settings, visualiser):
        super().__init__()
        self.settings = settings

        self.visualiser = visualiser
        self.wait_global_steps = self.settings.training.visualiser.wait_global_steps

        self.input_product_names = visualiser.dataset.input_product_names
        self.output_product_names = visualiser.dataset.output_product_names
        self.auxiliary_product_names = visualiser.dataset.auxiliary_product_names

        self.num_channels = len(self.input_product_names)
        self.num_classes = self.settings.model.num_classes # len(self.output_product_names)
        self.lr = 0.00006 ### self.settings.model.lr

        self.extra_metrics = self.settings.model.extra_metrics

        self.model_version = self.settings.model.transformer.backbone
        self.pretrained = self.settings.model.transformer.pretrained

        # segmentation / regression
        self.task = settings.model.task

        # Loss
        self.loss_name = "loss"

        # Send to the model: multiply_loss_by_mag1c, loss_reduction, positive_weight
        # Note, for multiply_loss_by_mag1c to work, we have to use the overriden loss (loss_overrides.multilabel_override)
        self.multiply_loss_by_mag1c = self.settings.model.multiply_loss_by_mag1c
        self.positive_weight = torch.nn.Parameter(torch.tensor(float(self.settings.model.positive_weight)), requires_grad=False)
        self.loss_reduction = "none" if self.multiply_loss_by_mag1c else "mean"
        print("DEBUG in init, multiply_loss_by_mag1c=",self.multiply_loss_by_mag1c,"positive_weight=",self.positive_weight,"loss_reduction=",self.loss_reduction)

        self.network = self.create_network()

        self.log_train_loss_every_batch_i = self.settings.model.log_train_loss_every_batch_i
        self.log_train_images_every_batch_i = self.settings.model.log_train_images_every_batch_i

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        # self.save_hyperparameters()

        # internals...
        self.validation_step_outputs = []
        self.validation_step_labels = []
        self.test_step_outputs = []
        self.test_step_labels = []

        self.model_desc = "HyperSegFormer"

        # Loss overrides (for sanity checks)
        self.multilabel_override = False
        self.loss_override_to = None
        try:
            self.multilabel_override = self.settings.model.transformer.custom_config.loss_overrides.multilabel_override
            self.loss_override_to = self.settings.model.transformer.custom_config.loss_overrides.loss_override_to
        except:
            print("WARN! FAILED WITH self.settings.model.transformer.custom_config.loss_overrides")
            print("DEBUG self.settings.model.transformer = ", self.settings.model.transformer)

    def create_network(self):
        if self.num_classes == 1:
            self.num_classes = 2 # default 1 class problem, here is a binary classification
        print("number of classes:", self.num_classes)

        if self.num_classes == 2:
            id2label = {0: 'no-plume', 1: 'plume'}
            label2id = {v: k for k, v in id2label.items()}
            print("hardcoded:", id2label)
        else:
            id2label = {}
            for i in range(self.num_classes):
                id2label[i] = 'mineral_'+str(i) # the names don't matter here
            label2id = {v: k for k, v in id2label.items()}
            print("generated:", id2label)

        if self.settings.model.transformer.custom_config_features is not None:
            features = self.settings.model.transformer.custom_config_features
            custom_config = self.settings.model.transformer.custom_config
            # These will override the settings in custom_config
            if features.conv1x1:
                print("~ Turning on CONV")
                # Defaults for Conv:
                custom_config.conv1x1.SegformerOverlapPatchEmbeddings = True
            if features.stride:
                print("~ Turning on STRIDE")
                # Defaults for Stride:
                custom_config.strides.keep_default = False
                custom_config.strides.strides_custom = [2, 2, 2, 2]
            if features.upscale:
                print("~ Turning on UP")
                # Defaults for Up:
                custom_config.upscale.preclassifier = True
                custom_config.upscale.decimate_channels_ratio = 2
                custom_config.upscale.upscale_layers = 2
            self.settings.model.transformer.custom_config = custom_config

        if self.settings.model.transformer.custom_config is None:
            print("Using standard, pretrained SegFormer") # isn't used tbh
            # right now we assume pretrained:
            if self.pretrained:
                model = SegformerForSemanticSegmentation.from_pretrained(self.model_version,
                                                                         num_labels=2,
                                                                         id2label=id2label,
                                                                         label2id=label2id,
                                                                         )
                self.num_channels = 3
                self.num_classes = 2

        else:
            print("Using custom SegFormer!")
            print("custom config:", self.settings.model.transformer.custom_config)

            from hyper.models.segformer_model import custom_Segformer_creator
            image_size = int(self.settings.dataset.tiler.tile_size)  # 128
            num_channels = len(self.input_product_names)  # 60

            model = custom_Segformer_creator(self.model_version, id2label, label2id, num_classes = self.num_classes,
                                     custom_config = self.settings.model.transformer.custom_config,
                                     image_size = image_size, num_channels = num_channels, verbose=False,
                                     # Loss params:
                                     multiply_loss_by_mag1c=self.multiply_loss_by_mag1c, loss_reduction=self.loss_reduction, positive_weight=self.positive_weight,
            )
            self.num_channels = num_channels

        return model

    def summarise(self):
        pretrained = "(not pretrained)" if not self.pretrained else "(pretrained)"

        params = num_of_params(self.network)
        M_params = round(params / 1000 / 1000, 2)
        print("[Model] SegTransformer model version", self.model_version, pretrained,
              "with", str(M_params)+"M parameters.")
        print("- Input num channels =", self.num_channels, ", Output num classes=", self.num_classes)
        if self.settings.model.transformer.custom_config is not None:
            print("- using custom settings:", self.settings.model.transformer.custom_config)

    def training_step(self, batch, batch_idx) -> float:
        pixel_values = batch["x"]
        labels = batch["y"]

        if self.multilabel_override:
            pass
            # pixel_values ~ [16, 86, 64, 64]
            # labels ~ [16, 1, 64, 64]
        else:
            if self.num_classes <= 2:
                # special binary case ( which we have with methane - label should be just like this: )
                labels = labels[:, 0, :, :]  # B,1,W,H => B,W,H
                labels = labels.long()
            else:
                # or multi-hot
                labels = labels.float()

        outputs = self.network(pixel_values=pixel_values, labels=labels)
        loss, logits = outputs.loss, outputs.logits  # [8, 2, 32, 32]

        if self.multiply_loss_by_mag1c:
            weight_loss = batch["weight_mag1c"]  # mag1c weight
            loss = torch.mean(loss * weight_loss)

        # log loss
        if (batch_idx % self.log_train_loss_every_batch_i) == 0:
            self.log(f"train_loss", loss.detach(), prog_bar=True)

        # log images
        if (batch_idx % self.log_train_images_every_batch_i) == 0 and self.global_step > self.wait_global_steps:
            with torch.no_grad():
                upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear",
                                                             align_corners=False)  # [8, 2, 128, 128]
                if self.multilabel_override:
                    predictions = upsampled_logits
                else:
                    if self.num_classes <= 2:
                        predictions = upsampled_logits[:, [1], :, :]  # [B, 1, 128, 128]
                    else:
                        predictions = upsampled_logits
                if not self.multilabel_override:
                    # here we consider 0.5 as the traditional threshold
                    predictions = predictions + torch.ones_like(predictions) * 0.5

            log_images_from_batch("train", batch, predictions, batch_idx, self)

        return loss

    def forward(self, x):
        """ used by evaluation (for now) """
        outputs = self.network(pixel_values=x, labels=None)
        loss, logits = outputs.loss, outputs.logits  # [8, 2, 32, 32]
        upsampled_logits = nn.functional.interpolate(logits, size=x.shape[-2:], mode="bilinear",
                                                     align_corners=False)
        if self.multilabel_override:
            predictions = upsampled_logits
        else:
            if self.num_classes <= 2:
                predictions = upsampled_logits[:, [1], :, :]  # [B, 1, 128, 128]
            else:
                predictions = upsampled_logits
        if not self.multilabel_override:
            # here we consider 0.5 as the traditional threshold
            predictions = predictions + torch.ones_like(predictions) * 0.5

        return predictions

    def printed_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network.printed_forward(x)

    def log(self, *args, **kwargs):
        try:
            super().log(*args, **kwargs)
        except Exception as e:
            print(f"Bug logging {e}")

    def val_step(self, batch, batch_idx: int, prefix: str = "val"):
        pixel_values = batch["x"]
        labels = batch["y"]

        if self.multilabel_override:
            pass
        else:
            if self.num_classes <= 2:
                # special binary case ( which we have with methane - label should be just like this: )
                labels = labels[:, 0, :, :]  # B,1,W,H => B,W,H
                labels = labels.long()
            else:
                # or multi-hot
                labels = labels.float()

        outputs = self.network(pixel_values=pixel_values, labels=labels)
        loss, logits = outputs.loss, outputs.logits  # [8, 2, 32, 32]

        with torch.no_grad():
            upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear",
                                                         align_corners=False)  # [8, 2, 128, 128]

            if self.multilabel_override:
                predictions = upsampled_logits
            else:
                if self.num_classes <= 2:
                    # keeps the predictions unthresholded
                    predictions = upsampled_logits[:,[1],:,:] # [B, 1, 128, 128]
                else:
                    predictions = upsampled_logits
            if not self.multilabel_override:
                # here we consider 0.5 as the traditional threshold
                predictions = predictions + torch.ones_like(predictions) * 0.5

        if self.multiply_loss_by_mag1c:
            assert False, print("not implement, likely different resolution loss [B, output_classes, 32, 32] <> weights [B,1, 128,128]")
            # when we preserve the resolution, we could actually bring this back

        self.log(f"{prefix}_loss", loss.detach(), on_epoch=True, prog_bar=True)
        # on_epoch=True => Automatically accumulates and logs at the end of the epoch.

        # log images
        if batch_idx == 0 and self.global_step > self.wait_global_steps:
            log_images_from_batch(prefix, batch, predictions, batch_idx, self)

        del batch

        # logging for metrics at the end of the validation
        if self.multilabel_override:
            pass
        else:
            if self.num_classes <= 2:
                # not multi-hot
                labels = labels.unsqueeze(1)  # also back to [B,1,W,H]

        if prefix == "test":
            self.test_step_outputs.append(predictions.detach())
            self.test_step_labels.append(labels.detach())

        elif prefix == "val":
            self.validation_step_outputs.append(predictions.detach())
            self.validation_step_labels.append(labels.detach())

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

    # WORKAROUND FIX ~ see https://github.com/Lightning-AI/pytorch-lightning/issues/13246
    def on_load_checkpoint(self, checkpoint):  #checkpoint: Dict[str, Any]
        # This is to robustness to loading with pytorch loading modules when adding new params to the models...
        verbose = True
        if verbose:
            print("DEBUG ENTERED LOAD CHECKPOINT HOOK ~")
            print("Self keys:")
            for self_key in self.state_dict().keys():
                if self_key not in checkpoint["state_dict"].keys():
                    print("SELF HAS", self_key, "(and checkpoint doesn't!), it's value=", self.state_dict()[self_key])
            for checkpoint_key in checkpoint["state_dict"].keys():
                if checkpoint_key not in self.state_dict().keys():
                    print("CHECKPOINT HAS", checkpoint_key, "(and self doesn't!), it's value=", checkpoint["state_dict"][checkpoint_key])
            print("==========")

        legacy_param_keys = ["positive_weight"]
        for key in legacy_param_keys:

            if key not in checkpoint["state_dict"].keys():
                value = self.state_dict()[key]
                print("NonStrictLoading / Filling key", key, "with the default value of", value)
                checkpoint["state_dict"][key] = value

