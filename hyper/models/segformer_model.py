# Model proposals and adaptations by Vit Ruzicka (in 2024-2025 during PhD)
# Relevant paper: https://doi.org/10.1109/JSTARS.2025.3557527


# Background
# - based on sources from PyTorch SegFormer code at https://github.com/huggingface/transformers/tree/main/src/transformers/models/segformer
# - custom changes to adapt the architecture to work better with Remote Sensing data (heavily wip so far, with plans to add more)

import torch
from typing import Optional, Tuple, Union
from transformers import SegformerPreTrainedModel, SegformerModel
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import numpy as np
from transformers.modeling_outputs import SemanticSegmenterOutput, BaseModelOutput
import math
import time

from hyper.models.model_utils import reporting_hook
from transformers.models.segformer.modeling_segformer import SegformerAttention, SegformerDropPath, SegformerMixFFN, SegformerMLP
from hyper.models.custom_layers import UpscaleBlock

VERBOSE = False
VERBOSE_detail = 5

# for better formated prints:
PRINTED_FORWARDS = False
_START = " |\t"
DELAY = _START+"                "

def shape_format(shape, batch_to_B=True):
    """Helps with printing the shapes"""
    out_shape = list(shape)
    if batch_to_B:
        out_shape[0] = "batch" #"B="+str(out_shape[0])

    out_str = "["
    for s in out_shape:
        out_str += str(s)+", "
    out_str = out_str[:-2]
    out_str += "]"

    return out_str

def num_of_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

class SegformerOverlapPatchEmbeddings_HSI(nn.Module):
    """Construct the overlapping patch embeddings."""

    def __init__(self, patch_size, stride, num_channels, hidden_size, custom_config = "None"):
        super().__init__()
        self.conv1x1_active = False

        # (additional) conv 2d with 1x1 kernel size (spectral only)
        if custom_config != "None" and custom_config.conv1x1.SegformerOverlapPatchEmbeddings:
            self.conv1x1_active = True

            dim_of_conv1x1 = num_channels # same as the num of channels

            self.proj_1x1 = nn.Conv2d(
                num_channels,
                dim_of_conv1x1,
                kernel_size=1, # to get 1x1
                # we want the spatial resolution to be kept the same ... set stride and padding accordingly ...
                stride=1,
                padding=0,
            )

        # conv 2d
        self.conv_k = patch_size
        self.proj_kxk = nn.Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values):
        if self.conv1x1_active:
            spectrally_embedded = self.proj_1x1(pixel_values)
            embeddings = self.proj_kxk(spectrally_embedded)
            if PRINTED_FORWARDS:
                global DELAY
                print(DELAY+"    >>OverlapPatchEmbeddings from", shape_format(pixel_values.shape), "-> conv1x1", shape_format(spectrally_embedded.shape), "-> conv"+str(self.conv_k)+"x"+str(self.conv_k), shape_format(embeddings.shape),)
                reporting_hook()

        else:
            embeddings = self.proj_kxk(pixel_values)
            if PRINTED_FORWARDS:
                print(DELAY+"    >>OverlapPatchEmbeddings from", shape_format(pixel_values.shape), "-> conv"+str(self.conv_k)+"x"+str(self.conv_k), shape_format(embeddings.shape),)
                reporting_hook()

        _, _, height, width = embeddings.shape
        # (batch_size, num_channels, height, width) -> (batch_size, num_channels, height*width) -> (batch_size, height*width, num_channels)
        # this can be fed to a Transformer layer
        embeddings = embeddings.flatten(2).transpose(1, 2)
        embeddings = self.layer_norm(embeddings)
        return embeddings, height, width


class SegformerLayer_HSI(nn.Module):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config, hidden_size, num_attention_heads, drop_path, sequence_reduction_ratio, mlp_ratio, custom_config = "None"):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.attention = SegformerAttention(
            config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
        )
        self.drop_path = SegformerDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = SegformerMixFFN(config, in_features=hidden_size, hidden_features=mlp_hidden_size)

    def forward(self, hidden_states, height, width, output_attentions=False):
        self_attention_outputs = self.attention(
            self.layer_norm_1(hidden_states),  # in Segformer, layernorm is applied before self-attention
            height,
            width,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection (with stochastic depth)
        attention_output = self.drop_path(attention_output)
        hidden_states = attention_output + hidden_states

        mlp_output = self.mlp(self.layer_norm_2(hidden_states), height, width)

        # second residual connection (with stochastic depth)
        mlp_output = self.drop_path(mlp_output)
        layer_output = mlp_output + hidden_states

        outputs = (layer_output,) + outputs

        return outputs


class SegformerEncoder_HSI(nn.Module):
    def __init__(self, config, custom_config = "None"):
        super().__init__()
        self.config = config

        # stochastic depth decay rule
        drop_path_decays = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]

        # patch embeddings
        if VERBOSE: print("[patch embeddings]")
        embeddings = []
        for i in range(config.num_encoder_blocks):
            embeddings.append(
                SegformerOverlapPatchEmbeddings_HSI(
                    patch_size=config.patch_sizes[i],
                    stride=config.strides[i],
                    num_channels=config.num_channels if i == 0 else config.hidden_sizes[i - 1],
                    hidden_size=config.hidden_sizes[i],
                    custom_config = custom_config
                )
            )
            if VERBOSE: print("SegformerOverlapPatchEmbeddings (patch_size=",config.patch_sizes[i],", stride=",config.strides[i],", num_channels=", config.num_channels if i == 0 else config.hidden_sizes[i - 1],", hidden_size=",config.hidden_sizes[i],")")
        self.patch_embeddings = nn.ModuleList(embeddings)

        # Transformer blocks
        blocks = []
        cur = 0
        if VERBOSE: print("[Transformer blocks]")

        for i in range(config.num_encoder_blocks):
            # each block consists of layers
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            for j in range(config.depths[i]):
                layers.append(
                    SegformerLayer_HSI(
                        config,
                        hidden_size=config.hidden_sizes[i],
                        num_attention_heads=config.num_attention_heads[i],
                        drop_path=drop_path_decays[cur + j],
                        sequence_reduction_ratio=config.sr_ratios[i],
                        mlp_ratio=config.mlp_ratios[i],
                        custom_config=custom_config
                    )
                )
                if VERBOSE: print("SegformerLayer (hidden_size=", config.hidden_sizes[i], ", num_attention_heads=",
                                  config.num_attention_heads[i], ", drop_path=",
                                  drop_path_decays[cur + j], ", sequence_reduction_ratio=",
                                  config.sr_ratios[i],"mlp_ratio=", config.mlp_ratios[i], ")")

            blocks.append(nn.ModuleList(layers))

        self.block = nn.ModuleList(blocks)

        # Layer norms
        self.layer_norm = nn.ModuleList(
            [nn.LayerNorm(config.hidden_sizes[i]) for i in range(config.num_encoder_blocks)]
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        batch_size = pixel_values.shape[0]

        hidden_states = pixel_values
        for idx, x in enumerate(zip(self.patch_embeddings, self.block, self.layer_norm)):
            embedding_layer, block_layer, norm_layer = x
            # first, obtain patch embeddings
            if PRINTED_FORWARDS:
                jump = "   " * idx
                print(_START+jump+"- TransformerBlock: IN", shape_format(hidden_states.shape))
                reporting_hook()

            hidden_states, height, width = embedding_layer(hidden_states)
            # second, send embeddings through blocks
            for i, blk in enumerate(block_layer):
                layer_outputs = blk(hidden_states, height, width, output_attentions)
                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            if PRINTED_FORWARDS:
                global DELAY
                DELAY = _START+jump+"                   "
                print(_START+jump+"                   ", str(len(block_layer))+"x Att+MixFFN with dims", shape_format(hidden_states.shape))
                reporting_hook()

            # third, apply layer norm
            hidden_states = norm_layer(hidden_states)
            # fourth, optionally reshape back to (batch_size, num_channels, height, width)
            if idx != len(self.patch_embeddings) - 1 or (
                idx == len(self.patch_embeddings) - 1 and self.config.reshape_last_stage
            ):
                hidden_states = hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
            if PRINTED_FORWARDS:
                print(_START+jump+"                   \OUT", shape_format(hidden_states.shape))
                reporting_hook()

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class SegformerDecodeHead_HSI(SegformerPreTrainedModel):
    def __init__(self, config, custom_config = "None", decode_head_dim_override = None):
        super().__init__(config)
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
        for i in range(config.num_encoder_blocks):
            mlp = SegformerMLP(config, input_dim=config.hidden_sizes[i])
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)

        classifier_in_dim = config.decoder_hidden_size
        self.upscale_active = False

        if custom_config != "None" and custom_config.upscale.preclassifier:
            r = custom_config.upscale.decimate_channels_ratio
            self.upscale = UpscaleBlock(in_channels=config.decoder_hidden_size,
                                        out_channels=int(config.decoder_hidden_size / r),
                                        num_layers=custom_config.upscale.upscale_layers)

            # Inserting an upscale layer before the classifier
            self.upscale_active = True
            classifier_in_dim = int(config.decoder_hidden_size / r)

        # We may want to make the output just a simple 1 band (in binary class scenario)
        if decode_head_dim_override is not None:
            num_labels = decode_head_dim_override
        else:
            num_labels = config.num_labels
        self.classifier = nn.Conv2d(classifier_in_dim, num_labels, kernel_size=1)

        self.config = config

    def forward(self, encoder_hidden_states: torch.FloatTensor) -> torch.Tensor:
        batch_size = encoder_hidden_states[-1].shape[0]
        if PRINTED_FORWARDS:
            print(_START+"- Decoder with", len(encoder_hidden_states), "strands:")
            reporting_hook()

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )

            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            if PRINTED_FORWARDS:
                shape_a = shape_format(encoder_hidden_state.shape)
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)

            if PRINTED_FORWARDS:
                shape_b = shape_format(encoder_hidden_state.shape)

            # upsample
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
            )

            if PRINTED_FORWARDS:
                shape_c = shape_format(encoder_hidden_state.shape)
                print(_START+"  +decoder (mlp + up):", shape_a, "=>", shape_b, "=>", shape_c)
                reporting_hook()


            all_hidden_states += (encoder_hidden_state,)

        hidden_states = torch.cat(all_hidden_states[::-1], dim=1)
        if PRINTED_FORWARDS:
            print(_START+"  =decoder (concat):", shape_format(hidden_states.shape))
            reporting_hook()

        hidden_states = self.linear_fuse(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if PRINTED_FORWARDS:
            print(_START+"  decoder (fuse conv1x1 + BN + ReLU + DO):", shape_format(hidden_states.shape))
            reporting_hook()

        ### Upscale layer:
        if self.upscale_active:
            # usually ~ [batch, ch=256, res=32, res=32] => [batch, ch / 2, 2*res, 2*res]
            hidden_states = self.upscale(hidden_states)
            if PRINTED_FORWARDS:
                print(_START+"  upscale (nearest 2x + Conv + Conv):", shape_format(hidden_states.shape))
                reporting_hook()

        # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = self.classifier(hidden_states)
        if PRINTED_FORWARDS:
            print(_START+"  classifier (conv1x1):", shape_format(logits.shape))
            reporting_hook()

        return logits

class SegformerForSemanticSegmentation_HSI(SegformerPreTrainedModel):
    def __init__(self, config, custom_config = "None", multiply_loss_by_mag1c=None, loss_reduction=None, positive_weight=None):
        super().__init__(config)
        # init basic version
        self.segformer = SegformerModel(config)

        # adjust in this:
        self.segformer.encoder = SegformerEncoder_HSI(config, custom_config=custom_config)
        self.segformer.post_init()

        # optional loss overrides
        self.multilabel_override = False
        self.loss_override_to = None
        try:
            self.multilabel_override = custom_config.loss_overrides.multilabel_override
            self.loss_override_to = custom_config.loss_overrides.loss_override_to
        except:
            pass

        if self.multilabel_override:
            # special - set it to 1
            self.decode_head = SegformerDecodeHead_HSI(config, custom_config=custom_config, decode_head_dim_override=1)
        else:
            self.decode_head = SegformerDecodeHead_HSI(config, custom_config=custom_config)

        # Initialize weights and apply final processing
        self.post_init()

        # extra internals:
        self.nan_detected = False
        self.config.semantic_loss_ignore_index = -100 # (not used...)

        self.used_loss = "default_CrossEntropy"

        # Multi-hot losses
        if config.num_labels > 2:
            print("Use one of the multihot losses!")
            self.used_loss = "BCEWithLogitsLoss"
            print("Using loss:", self.used_loss)

            if self.used_loss == "BCEWithLogitsLoss":
                self.loss_criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
            if self.used_loss == "focal_loss":
                import torchvision
                self.loss_criterion = torchvision.ops.sigmoid_focal_loss # < reduction "mean"

        if self.multilabel_override:
            if self.loss_override_to == "BCEWithLogitsLoss":
                self.used_loss = "BCEWithLogitsLoss"
                if multiply_loss_by_mag1c is None:
                    self.loss_criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
                else:
                    if multiply_loss_by_mag1c:
                        print("Creating BCEWithLogitsLoss loss with positive_weight=",positive_weight,"and loss_reduction=",loss_reduction)
                        self.loss_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight, reduction=loss_reduction)
                    else:
                        print("Creating BCEWithLogitsLoss loss with reduction set to mean")
                        self.loss_criterion = torch.nn.BCEWithLogitsLoss(reduction="mean") #fix
            elif self.loss_override_to == "FocalLoss":
                self.used_loss = "FocalLoss"
                import torchvision
                self.loss_criterion = torchvision.ops.sigmoid_focal_loss # < reduction "mean"

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SemanticSegmenterOutput]:

        if PRINTED_FORWARDS:
            print(_START+"- Input:", shape_format(pixel_values.shape))
            reporting_hook()

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        if PRINTED_FORWARDS:
            # "Bottleneck" features
            shapes_str = ""
            for h in encoder_hidden_states:
                shapes_str += shape_format(h.shape) + ", "
            shapes_str = shapes_str[:-2]
            print(_START+"- bottleneck features", shapes_str)
            reporting_hook()

        logits = self.decode_head(encoder_hidden_states)


        loss = None
        if labels is not None:
            # upsample logits to the images' original size
            upsampled_logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            # upsampled_logits ~ [16, 2, 64, 64] torch.cuda.FloatTensor
            # labels ~ [16, 1, 64, 64] torch.cuda.FloatTensor
            if self.used_loss == "default_CrossEntropy":
                if self.config.num_labels > 1:
                    loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
                    loss = loss_fct(upsampled_logits, labels)
                elif self.config.num_labels == 1:
                    valid_mask = ((labels >= 0) & (labels != self.config.semantic_loss_ignore_index)).float()
                    loss_fct = BCEWithLogitsLoss(reduction="none")
                    loss = loss_fct(upsampled_logits.squeeze(1), labels.float())
                    loss = (loss * valid_mask).mean()
                else:
                    raise ValueError(f"Number of labels should be >=0: {self.config.num_labels}")
            elif self.used_loss == "BCEWithLogitsLoss":
                loss = self.loss_criterion(upsampled_logits, labels)
            elif self.used_loss == "FocalLoss":
                loss = self.loss_criterion(upsampled_logits, labels, reduction="mean")

            else:
                print("NO LOSS SETUP")
                assert False
            # print("DEBUG loss", self.used_loss, "is", loss)

        # extra checking for nans:
        if loss is not None:
            if torch.isnan(loss).any():
                if not self.nan_detected:
                    print("") # linebreak
                    print("[INTERNAL] WARNING - First moment of detecting NaN in the loss!, loss=",loss)
                    print(" \ Debug are there nans in the input data? torch.isnan(pixel_values).any():", torch.isnan(pixel_values).any())
                    print(" \ Debug are there nans in the encoder outputs? ~encoder_hidden_states")
                    for h_i, h in enumerate(encoder_hidden_states):
                        print("in", h_i,":",torch.isnan(h).any())

                    print(" \ Debug are there nans in the decoder outputs? torch.isnan(logits).any():", torch.isnan(logits).any())
                    print(" \ Debug are there nans in the labels? torch.isnan(labels).any():", torch.isnan(labels).any())
                    print(" \ Input min:", torch.min(pixel_values))
                    print(" \ Input max:", torch.max(pixel_values))
                    print(" \ Labels min:", torch.min(labels))
                    print(" \ Labels max:", torch.max(labels))
                    self.nan_detected = True

                    ### maybe we can even kill the job?
                    ### or debug some more? - check for nans in the features flowing through the net?
                    # https://discuss.pytorch.org/t/handling-nan-loss/145447
                    # - replace nans by null element (0)? - reset gradients? - or just kill the job?
                    # assert False

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )

    def printed_forward(self, pixel_values, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        global PRINTED_FORWARDS
        global DELAY
        PRINTED_FORWARDS = True
        DELAY_initial = DELAY

        print("<Printed forwards>")

        output = self.forward(pixel_values, labels, output_attentions, output_hidden_states, return_dict)
        PRINTED_FORWARDS = False
        DELAY = DELAY_initial

        print("</END OF Printed forwards>")
        return output


def custom_Segformer_creator(model_version, id2label, label2id, custom_config = "None", image_size = 128, num_channels = 60, num_classes = 2, verbose=False,
                             multiply_loss_by_mag1c=None, loss_reduction=None, positive_weight=None):
    # From pretrained to get the right config ... (This however causes double creation of the model...)
    VERBOSE = False
    model = SegformerForSemanticSegmentation_HSI.from_pretrained(model_version,
                                                             num_labels=num_classes,
                                                             id2label=id2label,
                                                             label2id=label2id,
                                                             )

    # Adjust the config
    model.config.image_size = image_size # int(self.settings.dataset.tiler.tile_size)  # 128
    model.config.num_channels = num_channels # len(self.input_product_names)  # 60

    if custom_config != "None" and not custom_config.strides.keep_default:
        strides_custom = custom_config.strides.strides_custom
        print("(will set strides to...) strides_custom", strides_custom)

        model.config.strides = strides_custom

    # Create our custom version:
    if verbose:
        VERBOSE = True
    model_from_scratch = SegformerForSemanticSegmentation_HSI(model.config, custom_config=custom_config,
                                                              multiply_loss_by_mag1c=multiply_loss_by_mag1c, loss_reduction=loss_reduction, positive_weight=positive_weight)
    VERBOSE = False

    return model_from_scratch


if __name__ == "__main__":
    # Little demo code:

    # model_version = "nvidia/mit-b0"
    model_version = "nvidia/mit-b1"
    # model_version = "nvidia/mit-b2"

    id2label = {0: 'no-plume', 1: 'plume'}
    label2id = {v: k for k, v in id2label.items()}

    custom_config = None
    import types
    custom_config_dict = {
        "conv1x1": {
            # Settings for added spectral convolutions in the following layers:
            "SegformerOverlapPatchEmbeddings" : False,
            # "SegformerOverlapPatchEmbeddings" : True,
        },
        "strides": {
            "keep_default": True, # keep default, at whatever setting they are in this architecture
            # "keep_default": False,

            # "strides_custom": [4,2,2,2]
            "strides_custom": [2,2,2,2]
            # "strides_custom": [1,2,2,2]
        },
        "upscale": {
            "preclassifier": False,
            # "preclassifier": True,  # True to insert an upscale just before the classifier (after the fuse and dropout)
            "decimate_channels_ratio": 2,
            "upscale_layers": 2,  # num of the conv layers to do the upscaling
        }

    }

    custom_config = types.SimpleNamespace()
    custom_config.conv1x1 = types.SimpleNamespace(**custom_config_dict["conv1x1"])
    custom_config.strides = types.SimpleNamespace(**custom_config_dict["strides"])
    custom_config.upscale = types.SimpleNamespace(**custom_config_dict["upscale"])
    # note: we want to access it like the hydra config stuff:
    # custom_config.conv1x1.SegformerOverlapPatchEmbeddings

    VERBOSE = False

    image_size = 64
    band_num = 86

    input_shape = (8, band_num, image_size, image_size)
    example_input = torch.zeros((input_shape))

    baseline_model = custom_Segformer_creator(model_version, id2label, label2id,
                                                  custom_config = "None", image_size = image_size, num_channels = band_num,
                                                  verbose=False)
    M_params = num_of_params(baseline_model) / 1000 / 1000
    print("[baseline] SegTransformer has", str(M_params) + "M parameters.")

    # time / warmup?
    warmup_ = baseline_model.forward( example_input )
    warmup_ = baseline_model.forward( example_input )
    warmup_ = baseline_model.forward( example_input )

    # Baseline model print
    outputs = baseline_model.printed_forward( example_input )

    repeats = 5
    start = time.time()
    for i in range(repeats):
        out_ = baseline_model.forward(example_input)
    end = time.time()

    print("[baseline] time:", (end - start)/ repeats)


    model_from_scratch = custom_Segformer_creator(model_version, id2label, label2id,
                                                  custom_config = custom_config, image_size = image_size, num_channels = band_num,
                                                  verbose=False)
    M_params = num_of_params(model_from_scratch) / 1000 / 1000
    print("[custom] SegTransformer has", str(M_params) + "M parameters.")

    # time / warmup?
    warmup_ = model_from_scratch.forward( example_input )
    warmup_ = model_from_scratch.forward( example_input )
    warmup_ = model_from_scratch.forward( example_input )

    # Forward with heavy debug prints, stating all relevant shapes:
    outputs = model_from_scratch.printed_forward( example_input )

    start = time.time()
    for i in range(repeats):
        out_ = model_from_scratch.forward(example_input)
    end = time.time()

    print("example_input", example_input.shape, "=>", "example_output", outputs.logits.shape)
    print("[custom] time:", (end - start)/repeats)

    # print("model.config:", model_from_scratch.config)
    # print(model_from_scratch)
