# Model proposals and adaptations by Vit Ruzicka (in 2024-2025 during PhD)
# Relevant paper: https://doi.org/10.1109/JSTARS.2025.3557527

import torch
import torch.nn as nn

from hyper.models.efficientvit_building_blocks import EfficientViTSeg, SegHead, build_kwargs_from_config
from hyper.models.efficientvit_building_blocks import efficientvit_backbone_b0, efficientvit_backbone_b1
from hyper.models.efficientvit_building_blocks import efficientvit_backbone_b2, efficientvit_backbone_b3

def create_seg_model(name: str, verbose=False, custom_config="None", **kwargs) -> EfficientViTSeg:
    model_dict = {
        "b0": custom_efficientvit_seg_b0,
        "b1": custom_efficientvit_seg_b1,
        "b2": custom_efficientvit_seg_b2,
        "b3": custom_efficientvit_seg_b3,
        #########################
        # L1 and L2 not converted
    }

    model_id = name.split("-")[0]
    if model_id not in model_dict:
        raise ValueError(f"Do not find {name} in the model zoo. List of models: {list(model_dict.keys())}")
    else:
        model = model_dict[model_id](verbose=verbose, custom_config=custom_config, **kwargs)
    return model


def custom_efficientvit_seg_b0(verbose=False, custom_config="None", **kwargs) -> EfficientViTSeg:
    backbone = efficientvit_backbone_b0(verbose=verbose, custom_config=custom_config, **kwargs)
    head = SegHead(
        verbose = verbose,
        custom_config=custom_config,
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[128, 64, 32],
        stride_list=[32, 16, 8],
        head_width=32,
        head_depth=1,
        expand_ratio=4,
        middle_op="mbconv",
        final_expand=4,
        **build_kwargs_from_config(kwargs, SegHead),
    )
    model = EfficientViTSeg(backbone, head)
    return model

def custom_efficientvit_seg_b1(verbose=False, custom_config="None", **kwargs) -> EfficientViTSeg:

    backbone = efficientvit_backbone_b1(verbose=verbose, custom_config=custom_config, **kwargs)
    head = SegHead(
        verbose = verbose,
        custom_config=custom_config,
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[256, 128, 64],
        stride_list=[32, 16, 8],
        head_width=64,
        head_depth=3,
        expand_ratio=4,
        middle_op="mbconv",
        final_expand=4,
        **build_kwargs_from_config(kwargs, SegHead),
    )
    model = EfficientViTSeg(backbone, head, verbose=verbose)
    return model

def custom_efficientvit_seg_b2(verbose=False, custom_config="None", **kwargs) -> EfficientViTSeg:
    backbone = efficientvit_backbone_b2(verbose=verbose, custom_config=custom_config, **kwargs)
    head = SegHead(
        verbose = verbose,
        custom_config=custom_config,
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[384, 192, 96],
        stride_list=[32, 16, 8],
        head_width=96,
        head_depth=3,
        expand_ratio=4,
        middle_op="mbconv",
        final_expand=4,
        **build_kwargs_from_config(kwargs, SegHead),
    )
    model = EfficientViTSeg(backbone, head)
    return model


def custom_efficientvit_seg_b3(verbose=False, custom_config="None", **kwargs) -> EfficientViTSeg:
    backbone = efficientvit_backbone_b3(verbose=verbose, custom_config=custom_config, **kwargs)
    head = SegHead(
        verbose = verbose,
        custom_config=custom_config,
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        stride_list=[32, 16, 8],
        head_width=128,
        head_depth=3,
        expand_ratio=4,
        middle_op="mbconv",
        final_expand=4,
        **build_kwargs_from_config(kwargs, SegHead),
    )
    model = EfficientViTSeg(backbone, head)
    return model



if __name__ == "__main__":
    import numpy as np
    import time


    def num_of_params(model):
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    image_size = 64
    band_num = 86

    input_shape = (8, band_num, image_size, image_size)
    example_input = torch.zeros((input_shape))


    custom_config = None
    import types
    custom_config_dict = {
        "conv1x1": False,
        "head_stride": 8,
        "upscale_layers": 0
    }
    custom_config = types.SimpleNamespace()
    custom_config.conv1x1 = custom_config_dict["conv1x1"]
    custom_config.head_stride = custom_config_dict["head_stride"]
    custom_config.upscale_layers = custom_config_dict["upscale_layers"]
    print(custom_config)

    model = custom_efficientvit_seg_b1(verbose=True, in_channels=band_num, n_classes=2, custom_config=custom_config)
    M_params = num_of_params(model) / 1000 / 1000
    print("[baseline] EfficientViT has", str(M_params) + "M parameters.")

    # # time / warmup?
    warmup_ = model.forward( example_input )
    warmup_ = model.forward( example_input )
    warmup_ = model.forward( example_input )

    # Baseline model print
    outputs = model.printed_forward( example_input )
    print("Model IN/OUT:", example_input.shape, "=>", outputs.shape)

    repeats = 5
    start = time.time()
    for i in range(repeats):
        out_ = model.forward(example_input)
    end = time.time()

    print("[baseline] time:", (end - start)/ repeats)

