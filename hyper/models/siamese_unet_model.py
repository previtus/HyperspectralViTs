# ! Experimental only implementation

# Model proposals and adaptations by Vit Ruzicka (in 2024-2025 during PhD)
# Relevant paper: https://doi.org/10.1109/JSTARS.2025.3557527
# - based on previous implementations of Siamese Unets from https://github.com/previtus/ChangeDetectionProject/blob/master/Model2_builder.py
#   but adapted with the current codebase and pytorch

import torch
from hyper.models.model_utils import num_of_params
from typing import Optional, Union, List
from hyper.models.unet_model_custom import CustomUnet, CustomUnetDecoder

class SiameseCustomUnet(CustomUnet):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        custom_config="None",
    ):
        super().__init__(encoder_name=encoder_name,encoder_depth=encoder_depth,encoder_weights=encoder_weights,
        decoder_use_batchnorm=decoder_use_batchnorm,decoder_channels=decoder_channels,decoder_attention_type=decoder_attention_type,
        in_channels=in_channels,classes=classes,activation=activation,aux_params=aux_params,custom_config=custom_config)

        assert custom_config.siamese.siamese == True
        self.siamese_method = custom_config.siamese.method

        out_channels = self.encoder.out_channels
        if self.siamese_method == "concat":
            # if we concatenate two features, we need to also adjust the out channels
            out_channels = [2*ch for ch in out_channels]

            del self.decoder # old one which didn't have adjusted number of channels
            self.decoder = CustomUnetDecoder(
                encoder_channels=out_channels, # < * 2
                decoder_channels=decoder_channels,
                n_blocks=encoder_depth,
                use_batchnorm=decoder_use_batchnorm,
                center=True if encoder_name.startswith("vgg") else False,
                attention_type=decoder_attention_type,
                custom_config=custom_config,
            )

    def forward(self, x, current_epoch=None, verbose=False):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        # INPUT = [batch size, num of times, channels, W, H]
        time_dim = x.shape[1]
        if verbose: print("input has", time_dim, "times")

        self.check_input_shape(x[:,0,:,:]) # pass time 1

        x_event = x[:,0,:,:]
        if verbose: print("x_event", x_event.shape)
        x_before = x[:,1,:,:]
        if verbose: print("x_before", x_before.shape)

        features_event, optional_late_fuse_product_event = self.encoder(x_event, current_epoch=current_epoch)
        if verbose:
            print("features event:")
            for f_i,f in enumerate(features_event):
                print(f_i,"=",f.shape)
        features_before, optional_late_fuse_product_before = self.encoder(x_before, current_epoch=current_epoch)
        if verbose:
            print("features before:")
            for f_i,f in enumerate(features_before):
                print(f_i,"=",f.shape)

        # Join / Process the features here:
        siamese_features = []
        for f_i in range(len(features_event)):

            # I think that for the first index, we could even ignore it ... - shows on the graph visualisation,
            #    but doesn't add more params anyway
            if f_i == 0:
                siamese_features.append(None)
                continue

            f_event = features_event[f_i]
            f_before = features_before[f_i]

            if self.siamese_method == "minus":
                f = f_event - f_before
            elif self.siamese_method == "plus":
                f = f_event + f_before
            elif self.siamese_method == "concat":
                f = torch.cat((f_event, f_before), dim=1)

            siamese_features.append(f)

        # Join / Process the optional product
        optional_late_fuse_product = None
        if optional_late_fuse_product_event is not None:
            if self.siamese_method == "minus":
                optional_late_fuse_product = optional_late_fuse_product_event - optional_late_fuse_product_before
            if self.siamese_method == "plus":
                optional_late_fuse_product = optional_late_fuse_product_event + optional_late_fuse_product_before
            elif self.siamese_method == "concat":
                optional_late_fuse_product = torch.cat((optional_late_fuse_product_event, optional_late_fuse_product_before), dim=1)


        if verbose:
            print("joined features:")
            for f_i,f in enumerate(siamese_features):
                if f is not None:
                    print(f_i,"=",f.shape)
                else:
                    print(f_i, "is None")

        decoder_output = self.decoder(*siamese_features, optional_late_fuse_product=optional_late_fuse_product)

        masks = self.segmentation_head(decoder_output)

        return masks

    def printed_forward(self, x: torch.Tensor, current_epoch=None) -> torch.Tensor:
        return self.forward(x=x,current_epoch=current_epoch,verbose=True)


if __name__ == "__main__":
    import types
    import omegaconf

    custom_config_dict = {
        "siamese": {
            "siamese" : True,
            # "method" : "minus",
            # "method" : "plus",
            "method" : "concat",
        },
        "conv1x1": {
            "encoder_stage0_replace_conv1x1": False,
            "encoder_stage0_side_conv1x1": 0,
        },
        "fuse_mf": {
            "late": False,
            "fadein_start_epoch": None,
            "fadein_end_epoch": None,

        }

    }
    # note: we want to access it like the hydra config stuff
    custom_config = types.SimpleNamespace()
    custom_config.siamese = types.SimpleNamespace(**custom_config_dict["siamese"])
    custom_config.conv1x1 = types.SimpleNamespace(**custom_config_dict["conv1x1"])
    custom_config.fuse_mf = types.SimpleNamespace(**custom_config_dict["fuse_mf"])

    settings = omegaconf.OmegaConf.load("../../scripts/settings.yaml")
    hyperstarcop_settings = settings.model.hyperstarcop

    VERBOSE = True

    ## DEFAULT MODEL:
    num_channels = 86 # allbands
    num_classes = 1
    pretrained = None

    model = SiameseCustomUnet(
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
    print("[Siamese] U-Net model with the", hyperstarcop_settings.backbone, "backbone", pretrained,
          "with", str(M_params) + "M parameters.")
    print("- Input num channels =", num_channels, ", Output num classes=", num_classes)

    print("Printed forward:")
    batch_size = 8
    time_steps = 2
    input_shape = (batch_size, time_steps, num_channels, 128, 128)

    example_input = torch.zeros((input_shape))

    outputs = model.forward( example_input, current_epoch=0 )

    # outputs = model.printed_forward( example_input )
    print("example_input", example_input.shape, "=>", "example_output", outputs.shape)

