# ! Experimental only implementation

# Model proposals and adaptations by Vit Ruzicka (in 2024-2025 during PhD)
# Relevant paper: https://doi.org/10.1109/JSTARS.2025.3557527
#
# - will be based on the implementation of the semantic segmentation library model smp.Unet, but allows for custom bits
# - starts with code from:
#   https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/decoders/unet/model.py

DEBUG_PRINTS = False
# DEBUG_PRINTS = True

import torch
import segmentation_models_pytorch as smp
from hyper.models.model_utils import num_of_params
from typing import Optional, Union, List
from segmentation_models_pytorch.base import SegmentationModel
from segmentation_models_pytorch.encoders._base import EncoderMixin
from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.base.modules import Activation
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from hyper.models.model_utils import reporting_hook

class CustomSegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1, custom_config="None"):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)

class CustomDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
        custom_config="None",
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CustomCenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True, custom_config="None"):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class CustomUnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
        custom_config="None",
    ):
        super().__init__()

        self.fuse_mf_late = False
        if custom_config != "None":
            if custom_config.fuse_mf.late:
                self.fuse_mf_late = True


        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:])

        # Output channels:
        if self.fuse_mf_late:
            skip_channels += [1] # we use the same op as the skip added by adding the MF product
        else:
            skip_channels += [0] # last skip channel is empty

        out_channels = decoder_channels

        if center:
            self.center = CustomCenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            CustomDecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features, optional_late_fuse_product=None):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None

            if self.fuse_mf_late:
                if i == len(self.blocks)-1: # on last one
                    assert skip is None # i think its always None at this stage normally ...
                    # late fusion can occur this way:
                    skip = optional_late_fuse_product

            x = decoder_block(x, skip)
        return x

class CustomMobileNetV2Encoder(torchvision.models.MobileNetV2, EncoderMixin):
    def __init__(self, in_channels, out_channels, depth=5, custom_config="None", **kwargs):
        inverted_residual_setting = [
            # t, c, n, s
            [1, out_channels[1], 1, 1],
            [6, out_channels[2], 2, 2],
            [6, out_channels[3], 3, 2],
            [6, 2*out_channels[3], 4, 2],
            [6, out_channels[4], 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        super().__init__(**kwargs, inverted_residual_setting=inverted_residual_setting)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3
        del self.classifier

        self.stage0_replace_conv1x1 = False
        self.stage0_side_conv1x1 = False
        self.fuse_mf_late = False
        # fade it in:
        self.fuse_mf_use_alpha = False
        self.fadein_start_epoch = None
        self.fadein_end_epoch = None
        self.fadein_frac_n = None

        final_in_channels = in_channels

        if custom_config != "None":
            if custom_config.fuse_mf.late:
                # 1st band is actually MF product, fuse it in later int the model
                self.fuse_mf_late = True
                final_in_channels = in_channels - 1
                if custom_config.fuse_mf.fadein_start_epoch != "None" and custom_config.fuse_mf.fadein_end_epoch != "None":
                    self.fuse_mf_use_alpha = True
                    self.fadein_start_epoch = int(custom_config.fuse_mf.fadein_start_epoch)
                    self.fadein_end_epoch = int(custom_config.fuse_mf.fadein_end_epoch)
                    n = self.fadein_end_epoch - self.fadein_start_epoch
                    self.fadein_frac_n = 1 / n

            if custom_config.conv1x1.encoder_stage0_replace_conv1x1:
                # Added 1x1 Convolution as a layer

                dim_of_conv1x1 = final_in_channels # here keeping the same ...
                final_in_channels = dim_of_conv1x1

                self.proj_stage0_replace_conv1x1 = nn.Conv2d(
                    in_channels,
                    dim_of_conv1x1,
                    kernel_size=1, # to get 1x1
                    # we want the spatial resolution to be kept the same ... set stride and padding accordingly ...
                    stride=1,
                    padding=0,
                )
                self.stage0_conv1x1 = True

            if custom_config.conv1x1.encoder_stage0_side_conv1x1 > 0:
                # Added 1x1 Convolution outputs of which will be appended to the original channels

                dim_of_conv1x1 = custom_config.conv1x1.encoder_stage0_side_conv1x1

                # the in for the stages will be larger ... (the usual one + the one increased here)
                final_in_channels = final_in_channels+dim_of_conv1x1

                self.proj_stage0_side_conv1x1 = nn.Conv2d(
                    in_channels,
                    dim_of_conv1x1,
                    kernel_size=1, # to get 1x1
                    # we want the spatial resolution to be kept the same ... set stride and padding accordingly ...
                    stride=1,
                    padding=0,
                )
                self.stage0_side_conv1x1 = True


        self.set_in_channels(final_in_channels, pretrained=False)

    def get_stages(self):
        return [
            nn.Identity(),
            self.features[:2],
            self.features[2:4],
            self.features[4:7],
            self.features[7:14],
            self.features[14:],
        ]

    def forward(self, x, current_epoch=None):
        stages = self.get_stages()

        optional_late_fuse_product = None

        if self.fuse_mf_late:
            # 1st band is actually MF product, split it and add it later
            if self.fuse_mf_use_alpha:


                alpha = 0.0
                if current_epoch >= self.fadein_end_epoch:
                    alpha = 1.0
                elif current_epoch >= (self.fadein_start_epoch-1):
                    k = current_epoch - (self.fadein_start_epoch-1)
                    alpha = round(k * self.fadein_frac_n, 3) # forbid some crazy vals

                if DEBUG_PRINTS:
                    print("alpha", alpha)


                alpha = 1.0
                optional_late_fuse_product = alpha * x[:,[0],:,:]
            else:
                optional_late_fuse_product = x[:, [0], :, :]
            x = x[:,1:,:,:]

        if self.stage0_side_conv1x1:
            if DEBUG_PRINTS: print("Entering proj_stage0_side_conv1x1 with x shaped", x.shape)
            x_side = self.proj_stage0_side_conv1x1(x)
            if DEBUG_PRINTS: print("\ Concat", x.shape, x_side.shape)
            x = torch.cat((x, x_side), dim=1)
            if DEBUG_PRINTS: print("\ Output", x.shape)

        if self.stage0_replace_conv1x1:
            if DEBUG_PRINTS: print("Entering stage0_replace_conv1x1 with x shaped", x.shape)
            x = self.proj_stage0_replace_conv1x1(x)
            if DEBUG_PRINTS: print("\ Output", x.shape)

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features, optional_late_fuse_product

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("classifier.1.bias", None)
        state_dict.pop("classifier.1.weight", None)
        super().load_state_dict(state_dict, **kwargs)


class CustomUnet(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        encoder_out_channels: List[int] = (3, 16, 24, 32, 96, 1280),
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        custom_config="None",
    ):
        # custom settings for channel numbers (use custom_config as that can be controlled from command line...)
        if custom_config != "None":
            if custom_config.custom_channels.encoder != "None" and custom_config.custom_channels.decoder != "None":
                print("setting encoder_out_channels as", custom_config.custom_channels.encoder)
                print("setting decoder_channels as", custom_config.custom_channels.decoder)
                encoder_out_channels = custom_config.custom_channels.encoder
                decoder_channels = custom_config.custom_channels.decoder

            if custom_config.custom_channels.encoder_depth != "None":
                print("setting encoder_depth as", custom_config.custom_channels.encoder_depth)
                encoder_depth = custom_config.custom_channels.encoder_depth

            if custom_config.custom_channels.output_bands != "None":
                print("setting classes as", custom_config.custom_channels.output_bands)
                classes = custom_config.custom_channels.output_bands

        super().__init__() # after we adjusted the args ^

        self.encoder = CustomMobileNetV2Encoder(in_channels=in_channels, out_channels=encoder_out_channels, depth=encoder_depth, custom_config=custom_config)

        self.decoder = CustomUnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
            custom_config=custom_config,
        )

        self.segmentation_head = CustomSegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
            custom_config=custom_config,
        )

        self.name = "u-{}".format(encoder_name)
        self.classification_head = None # we won't use here
        self.initialize()

        if custom_config != "None":
            print("With custom config:", custom_config)


    def forward(self, x, current_epoch=None):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)


        if current_epoch is not None:
            if DEBUG_PRINTS:
                print("I am currently in epoch", current_epoch)

        features, optional_late_fuse_product = self.encoder(x, current_epoch=current_epoch)
        decoder_output = self.decoder(*features, optional_late_fuse_product=optional_late_fuse_product)

        masks = self.segmentation_head(decoder_output)

        return masks


    def printed_forward(self, x: torch.Tensor, current_epoch=None) -> torch.Tensor:
        # PRINT PASSAGE THROUGH THE MODEL
        # Note: these work for this one particular UNET version, if some parts change (for example different encoder),
        #        then the individual parts might need to be rewritten...
        # 1 ENCODER

        stages = self.encoder.get_stages()

        print("- Input:", x.shape)
        reporting_hook()

        features, optional_late_fuse_product = self.encoder.forward(x)
        for i in range(len(features)):
            print("- Encoder Stage",i,":", features[i].shape)

        # 2 DECODER
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.decoder.center(head)
        print("- Center:", x.shape)
        reporting_hook()

        detailed_decoder = False
        print_attentions = False # by default, they are Identity layers, so it's not needed...
        # detailed_decoder = True # will print each step in the Decoder stage ...
        for i, decoder_block in enumerate(self.decoder.blocks):
            skip = skips[i] if i < len(skips) else None
            if detailed_decoder:
                print("- Decoder Stage",i," - input:", x.shape)
                x = F.interpolate(x, scale_factor=2, mode="nearest")
                print("- F.interpolate x 2:", x.shape)

                if optional_late_fuse_product is not None and i == len(self.decoder.blocks) - 1:
                    assert skip is None
                    skip = optional_late_fuse_product
                    print("- assigning as skip the late fused product:", skip.shape)
                if skip is not None:
                    x = torch.cat([x, skip], dim=1)
                    print("- torch.cat x skip (after cat):", x.shape)
                    x = decoder_block.attention1(x)
                    if print_attentions: print("- attention1:", x.shape, "layer type:", decoder_block.attention1)

                # These three following layers in theory restore the resolution + bake in info from the skip connection
                x = decoder_block.conv1(x)
                print("- conv1:", x.shape, "layer type:", decoder_block.conv1)
                x = decoder_block.conv2(x)
                print("- conv2:", x.shape, "layer type:", decoder_block.conv2)
                x = decoder_block.attention2(x)
                if print_attentions: print("- attention2:", x.shape, "layer type:", decoder_block.attention2)
                print("-> is also the out shape")
                print(".")


            else:
                x = decoder_block(x, skip)
                print("- Decoder Stage",i,":", x.shape)
            reporting_hook()

        # 3 SEGMENTATION HEAD
        x = self.segmentation_head(x) # in theory 3 sublayers...
        print("- Segmentation Head:", x.shape)
        reporting_hook()

        # 4
        #print("self.classification_head is None", self.classification_head)
        return x




if __name__ == "__main__":
    import types
    import omegaconf

    custom_config = None
    custom_config_dict = {
        "conv1x1": {
            # early ~ stage 0 blocks that work in the full res

            # typically choose one
            "encoder_stage0_replace_conv1x1" : False,
            # "encoder_stage0_replace_conv1x1" : True, # True / False
            "encoder_stage0_side_conv1x1" : 0, # 0 / num
            #"encoder_stage0_side_conv1x1" : 14, # 0 / num

            # late ~ potentially working in the feature space of the decoder
        },
        "fuse_mf": {
            # fuse in matched filter band somewhere into the model architecture ...
            # - data comes as: [specific products = also mf] + [all resolved bands]
            # early would mean training with mf on input
            # late add it somewhere in the decoder
            "late": False, # True / False ?
            "fadein_start_epoch": 15,
            # "fadein_end_epoch": 20, # fade in by 0.2
            "fadein_end_epoch": 25, # fade in by 0.1 # (btw 1.0 reached at this-1)
            # "fadein_start_epoch": None,
            # "fadein_end_epoch": None,

        },
        "custom_channels": {
            "encoder": "None",
            "decoder": "None",
            "encoder_depth": "None",
            "output_bands": "None"
        }
    }
    # note: we want to access it like the hydra config stuff
    custom_config = types.SimpleNamespace()
    custom_config.conv1x1 = types.SimpleNamespace(**custom_config_dict["conv1x1"])
    custom_config.fuse_mf = types.SimpleNamespace(**custom_config_dict["fuse_mf"])
    custom_config.custom_channels = types.SimpleNamespace(**custom_config_dict["custom_channels"])

    settings = omegaconf.OmegaConf.load("../../scripts/settings.yaml")
    hyperstarcop_settings = settings.model.hyperstarcop

    VERBOSE = True

    ## DEFAULT MODEL:
    num_channels = 4 # rgb + mf
    # num_channels = 86 # allbands
    # num_channels = 87 # mf + allbands
    num_classes = 1
    batch_size = 8
    pretrained = None if hyperstarcop_settings.pretrained == 'None' else hyperstarcop_settings.pretrained

    model = smp.Unet(
        encoder_name=hyperstarcop_settings.backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=pretrained,  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=num_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=num_classes,  # model output channels (number of classes in your dataset)
        activation=None,  # activation function, default is None
        # encoder_depth=4 # Depth parameter specify a number of downsampling operations in encoder, so you can make your model lighter if specify smaller depth
    )
    params = num_of_params(model)
    # M_params = round(params / 1000 / 1000, 2)
    M_params = params / 1000 / 1000
    print("[Model] U-Net model with the", hyperstarcop_settings.backbone, "backbone", pretrained,
          "with", str(M_params) + "M parameters.")
    print("- Input num channels =", num_channels, ", Output num classes=", num_classes)
    # print(model)

    print("default self.encoder", num_of_params(model.encoder))
    print("default self.decoder", num_of_params(model.decoder))
    print("default self.segmentation_head", num_of_params(model.segmentation_head))

    model = CustomUnet(
        encoder_name=hyperstarcop_settings.backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=pretrained,  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=num_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=num_classes,  # model output channels (number of classes in your dataset)
        activation=None,  # activation function, default is None
        # encoder_depth=4 # Depth parameter specify a number of downsampling operations in encoder, so you can make your model lighter if specify smaller depth
        custom_config=custom_config,
    )
    params = num_of_params(model)
    M_params = round(params / 1000 / 1000, 2)
    print("[CustomModel] U-Net model with the", hyperstarcop_settings.backbone, "backbone", pretrained,
          "with", str(M_params) + "M parameters.")
    print("- Input num channels =", num_channels, ", Output num classes=", num_classes)

    # print("custom self.encoder", num_of_params(model.encoder))
    # print("custom self.decoder", num_of_params(model.decoder))
    # print("custom self.segmentation_head", num_of_params(model.segmentation_head))

    print("Printed forward:")

    input_shape = (batch_size, num_channels, 128, 128)

    example_input = torch.zeros((input_shape))

    # for epoch_num in range(30):
    #     outputs = model.forward( example_input, current_epoch=epoch_num )
    outputs = model.forward( example_input, current_epoch=0 )

    # outputs = model.printed_forward( example_input )
    print("example_input", example_input.shape, "=>", "example_output", outputs.shape)

    # print("blocks")
    # print(model.decoder.blocks)



    ### Custom model for regression ~
    print("================")
    # note - these should be multiples of 8

    # default - 6.65M parameters
    custom_config.custom_channels.encoder_depth = 5
    custom_config.custom_channels.output_bands = num_channels

    model = CustomUnet(
        encoder_name=hyperstarcop_settings.backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=pretrained,  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=num_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        activation=None, # Linear?  # activation function, default is None
        custom_config=custom_config,
    )
    params = num_of_params(model)
    # M_params = round(params / 1000 / 1000, 2)
    M_params = params / 1000 / 1000
    print("With", str(M_params) + "M parameters.")

    input_shape = (batch_size, num_channels, 128, 128)
    example_input = torch.zeros((input_shape))
    # outputs = model.forward( example_input )
    outputs = model.printed_forward( example_input )
    print("example_input", example_input.shape, "=>", "example_output", outputs.shape)
