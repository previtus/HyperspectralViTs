# Model proposals and adaptations by Vit Ruzicka (in 2024-2025 during PhD)
# Relevant paper: https://doi.org/10.1109/JSTARS.2025.3557527

# Based on:
# Standalone bit of code to create the EfficientViT architecture
# (source https://github.com/mit-han-lab/efficientvit at commit a838154e985579493ca284adee8cbb91678b078c) ...
# note: alternative implementation would be the one in: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientvit_mit.py
#  ---------------------------
# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023


import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import signature
from functools import partial
from torch.cuda.amp import autocast
from torch.nn.modules.batchnorm import _BatchNorm
from hyper.models.custom_layers import UpscaleBlock

PRINTED_FORWARDS = False
INTENSE_PRINT = False
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

class EfficientViTSeg(nn.Module):
    def __init__(self, backbone, head, verbose=False) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if PRINTED_FORWARDS:
            print("EfficientViTSeg input", shape_format(x.shape))
        feed_dict = self.backbone(x)
        feed_dict = self.head(feed_dict)

        return feed_dict["segout"]

    def printed_forward(self, x: torch.Tensor) -> torch.Tensor:
        global PRINTED_FORWARDS
        PRINTED_FORWARDS = True
        print("")
        print("<Printed forwards>")
        output = self.forward(x)
        PRINTED_FORWARDS = False
        print("</END OF Printed forwards>")
        print("")
        return output

class DAGBlock(nn.Module):
    def __init__(
        self,
        inputs: dict[str, nn.Module],
        merge: str,
        post_input: nn.Module or None,
        middle: nn.Module,
        outputs: dict[str, nn.Module],
    ):
        super(DAGBlock, self).__init__()

        self.input_keys = list(inputs.keys())
        self.input_ops = nn.ModuleList(list(inputs.values()))
        self.merge = merge
        self.post_input = post_input

        self.middle = middle

        self.output_keys = list(outputs.keys())
        self.output_ops = nn.ModuleList(list(outputs.values()))

        # Note: self refers to variables set in SegHead
        # Custom upscale setting
        if self.upscale_active:
            self.upscale1 = UpscaleBlock(in_channels=self.upscale1_in,
                                        out_channels=self.upscale1_out,
                                        num_layers=self.upscale_sublayers)
            if self.verbose:
                print("\  [Upscale1] Upscale in/out=",self.upscale1_in,self.upscale1_out)

            if self.upscale_layers > 1:
                self.upscale2 = UpscaleBlock(in_channels=self.upscale2_in,
                                             out_channels=self.upscale2_out,
                                             num_layers=self.upscale_sublayers)
                if self.verbose:
                    print("\  [Upscale2] Upscale in/out=",self.upscale2_in,self.upscale2_out)

    def forward(self, feature_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if PRINTED_FORWARDS:
            print("Decoder/SegHead gets inputs", list(feature_dict.keys()), "and uses only", self.input_keys)

        feat = [op(feature_dict[key]) for key, op in zip(self.input_keys, self.input_ops)]
        if PRINTED_FORWARDS:
            for key, f, op in zip(self.input_keys, feat, self.input_ops):
                if INTENSE_PRINT:
                    print(key, "uses op:")
                    try:
                        print(op.op_list)
                    except:
                        print(op)
                    print("outputs as =>", shape_format(f.shape))
                else:
                    print(key, "outputs as =>", shape_format(f.shape))

        if self.merge == "add":
            feat = list_sum(feat)
        elif self.merge == "cat":
            feat = torch.concat(feat, dim=1)
        else:
            raise NotImplementedError

        if PRINTED_FORWARDS:
            print("SegHead merged with", self.merge, "as:", shape_format(feat.shape))

        if self.post_input is not None:
            if PRINTED_FORWARDS:
                print("using post_input:", self.post_input)

            feat = self.post_input(feat)
        feat = self.middle(feat)
        if PRINTED_FORWARDS:
            print("SegHead after middle:", shape_format(feat.shape))

        # Notes on architecture:
        # in Head the middle was: one MBConv block (inverted conv, depth conv and point cont)
        # ... so similar like the MLP layer in SegFormer which is after the concat ...
        # ... so place Upscale layer here
        if self.upscale_active:
            feat = self.upscale1(feat)
            if PRINTED_FORWARDS:
                print("+ Upscale1 =>", shape_format(feat.shape))

            if self.upscale_layers > 1:
                feat = self.upscale2(feat)
                if PRINTED_FORWARDS:
                    print("+ Upscale2 =>", shape_format(feat.shape))

        # Output operations
        for key, op in zip(self.output_keys, self.output_ops):
            feature_dict[key] = op(feat)

        if PRINTED_FORWARDS:
            print("SegHead output keys are:", feature_dict.keys())
            print("SegHead main output shape {Batch, NClasses, Res, Res}:", shape_format(feature_dict["segout"].shape))

        return feature_dict

class SegHead(DAGBlock):
    def __init__(
        self,
        fid_list: list[str],
        in_channel_list: list[int],
        stride_list: list[int],
        # head_stride: int,
        head_width: int,
        head_depth: int,
        expand_ratio: float,
        middle_op: str,
        final_expand: float or None,
        n_classes: int,
        dropout=0,
        norm="bn2d",
        act_func="hswish",
        verbose=False,
        custom_config="None",
    ):
        self.custom_config = custom_config
        self.verbose = verbose

        # Custom stride setting
        head_stride = 8 # default value
        if custom_config != "None" and custom_config.head_stride != head_stride:
            if verbose: print("With custom stride =",custom_config.head_stride)
            head_stride = custom_config.head_stride

        inputs = {}
        if verbose: print("SegHead START =")
        for fid, in_channel, stride in zip(fid_list, in_channel_list, stride_list):

            factor = stride // head_stride
            if factor == 1:
                if verbose:
                    print("\ ", fid, "Drop + Conv2D (in:", in_channel, "out:", head_width, "kernel=", 1, ") + Norm" ) # in, out, kernel size
                inputs[fid] = ConvLayer(in_channel, head_width, 1, norm=norm, act_func=None)
            else:
                if verbose:
                    print("\ ", fid, "Drop + Conv2D (in:", in_channel, "out:", head_width, "kernel=", 1, ") + Norm + UpBicubic", str(factor)+"x" ) # in, out, kernel size
                inputs[fid] = OpSequential(
                    [
                        ConvLayer(in_channel, head_width, 1, norm=norm, act_func=None),
                        UpSampleLayer(factor=factor),
                    ]
                )

        if verbose: print("SegHead MIDDLE =")

        middle = []
        for _ in range(head_depth):
            if middle_op == "mbconv":
                block = MBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, act_func, None),
                    verbose=verbose
                )
            elif middle_op == "fmbconv":
                block = FusedMBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, None),
                    verbose=verbose
                )
            else:
                raise NotImplementedError

            if verbose: print("\  ResidualBlock")
            middle.append(ResidualBlock(block, IdentityLayer()))
        middle = OpSequential(middle)

        # Custom upscale setting
        classifier_in_dim = head_width
        self.upscale_active = False

        if custom_config != "None" and custom_config.upscale_layers > 0:
            decimate_channels_ratio = 2 # good def
            self.upscale_sublayers = 2  # good def
            self.upscale1_in = head_width
            self.upscale1_out = int(self.upscale1_in / decimate_channels_ratio)
            classifier_in_dim = self.upscale1_out

            if custom_config.upscale_layers > 1:
                self.upscale2_in = self.upscale1_out
                self.upscale2_out = int(self.upscale2_in / decimate_channels_ratio)
                classifier_in_dim = self.upscale2_out

            # Inserting an upscale layer before the classifier
            self.upscale_active = True
            self.upscale_layers = custom_config.upscale_layers

        if verbose:
            print("SegHead OUTPUT =")
            if final_expand is None:
                print("no extra operation")
            else:
                print("\  Drop + Conv2D (in:", classifier_in_dim, "out:", classifier_in_dim * final_expand, "kernel=", 1, ") + Norm + Act " + act_func )
                print("\  Drop + Conv2D (in:", classifier_in_dim * (final_expand or 1), "out:", n_classes, "kernel=", 1, ")")

        outputs = {
            "segout": OpSequential(
                [
                    None
                    if final_expand is None
                    else ConvLayer(classifier_in_dim, classifier_in_dim * final_expand, 1, norm=norm, act_func=act_func),
                    ConvLayer(
                        classifier_in_dim * (final_expand or 1),
                        n_classes,
                        1,
                        use_bias=True,
                        dropout=dropout,
                        norm=None,
                        act_func=None,
                    ),
                ]
            )
        }

        super(SegHead, self).__init__(inputs, "add", None, middle=middle, outputs=outputs)


### UTILS ###

def build_kwargs_from_config(config: dict, target_func: callable) -> dict[str, any]:
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs

def list_sum(x: list) -> any:
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])



# register activation function here
REGISTERED_ACT_DICT: dict[str, type] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu": nn.SiLU,
    "gelu": partial(nn.GELU, approximate="tanh"),
}


def build_act(name: str, **kwargs) -> nn.Module or None:
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        args = build_kwargs_from_config(kwargs, act_cls)
        return act_cls(**args)
    else:
        return None


def val2list(x: list or tuple or any, repeat_time=1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def get_same_padding(kernel_size: int or tuple[int, ...]) -> int or tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


# register normalization function here
REGISTERED_NORM_DICT: dict[str, type] = {
    "bn2d": nn.BatchNorm2d,
    "ln": nn.LayerNorm,
    "ln2d": LayerNorm2d,
}


def build_norm(name="bn2d", num_features=None, **kwargs) -> nn.Module or None:
    if name in ["ln", "ln2d"]:
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    else:
        return None

def set_norm_eps(model: nn.Module, eps: float or None = None) -> None:
    for m in model.modules():
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm, _BatchNorm)):
            if eps is not None:
                m.eps = eps

def resize(
    x: torch.Tensor,
    size: any or None = None,
    scale_factor: list[float] or None = None,
    mode: str = "bicubic",
    align_corners: bool or None = False,
) -> torch.Tensor:
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    elif mode in {"nearest", "area"}:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")


### BACKBONE ###

class EfficientViTBackbone(nn.Module):
    def __init__(
        self,
        width_list: list[int],
        depth_list: list[int],
        in_channels=3,
        dim=32,
        expand_ratio=4,
        norm="bn2d",
        act_func="hswish",
        verbose=False,
        custom_config="None"
    ) -> None:
        super().__init__()

        # Custom settings:
        self.custom_config = custom_config
        self.conv1x1_active = False

        if custom_config != "None" and custom_config.conv1x1:
            self.conv1x1_active = True

        self.width_list = []
        # input stem

        if verbose:
            print("EfficientViTBackbone builder:")
            print("input_stem:")

        if self.conv1x1_active:
            if verbose:
                print("\  [Conv1x1] Conv1x1 in/out=", in_channels, in_channels)
            self.input_stem_proj_1x1 = nn.Conv2d(
                in_channels, in_channels, # same
                kernel_size=1, # to get 1x1
                # we want the spatial resolution to be kept the same ... set stride and padding accordingly ...
                stride=1, padding=0,
            )

        if verbose:
            print("\  ConvLayer in/out= ",in_channels,width_list[0])
        self.input_stem = [
            ConvLayer(
                in_channels=in_channels,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
                verbose=verbose
            )
            self.input_stem.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.input_stem = OpSequential(self.input_stem)
        self.width_list.append(in_channels)

        # stages
        if verbose:
            print("stages:")

        # does 16, 32
        self.stages = []
        if self.conv1x1_active:
            self.stages_conv1x1 = []
        for w, d in zip(width_list[1:3], depth_list[1:3]):
            stage = []

            # Add Conv1x1 #
            if self.conv1x1_active:
                conv1x1_for_this_stage = nn.Conv2d(
                    in_channels, in_channels, kernel_size=1, stride=1, padding=0
                )
                if verbose:
                    print("\  [Conv1x1] Conv1x1 in/out=", in_channels, in_channels)
                self.stages_conv1x1.append(conv1x1_for_this_stage)
            # \Add Conv1x1 #

            for i in range(d):
                stride = 2 if i == 0 else 1
                block = self.build_local_block(
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=act_func,
                    verbose=verbose
                )
                block = ResidualBlock(block, IdentityLayer() if stride == 1 else None)
                stage.append(block)
                in_channels = w
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        # does 64, 128
        for w, d in zip(width_list[3:], depth_list[3:]):
            stage = []

            # Add Conv1x1 #
            if self.conv1x1_active:
                conv1x1_for_this_stage = nn.Conv2d(
                    in_channels, in_channels, kernel_size=1, stride=1, padding=0
                )
                if verbose:
                    print("\  [Conv1x1] Conv1x1 in/out=", in_channels, in_channels)
                self.stages_conv1x1.append(conv1x1_for_this_stage)
            # \Add Conv1x1 #

            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
                verbose=verbose
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w

            for _ in range(d):
                stage.append(
                    EfficientViTBlock(
                        in_channels=in_channels,
                        dim=dim,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=act_func,
                    )
                )
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)
        if self.conv1x1_active:
            self.stages_conv1x1 = nn.ModuleList(self.stages_conv1x1)

    @staticmethod
    def build_local_block(
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
        verbose=False
    ) -> nn.Module:
        if expand_ratio == 1:
            if verbose:
                print("\  DSConv in/out= ", in_channels, out_channels)
            block = DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        else:
            if verbose:
                print("\  MBConv in/out= ", in_channels, out_channels)
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        return block

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:

        if PRINTED_FORWARDS:
            print("Encoder/Backbone input", shape_format(x.shape))
            before_shape = shape_format(x.shape)

        output_dict = {"input": x}

        if self.conv1x1_active:
            x = self.input_stem_proj_1x1(x)
            if PRINTED_FORWARDS:
                print("+ Conv1x1", before_shape, "=>", shape_format(x.shape))
                before_shape = shape_format(x.shape)

        output_dict["stage0"] = x = self.input_stem(x)

        if PRINTED_FORWARDS:
            print("backbone stage0:", before_shape, "=>", shape_format(x.shape))
            before_shape = shape_format(x.shape)



        for stage_id, stage in enumerate(self.stages, 1):

            if self.conv1x1_active:
                conv1x1_layer = self.stages_conv1x1[stage_id-1] # because of stage0 it started from 1, we have them separate though
                x = conv1x1_layer(x)
                if PRINTED_FORWARDS:
                    print("+ Conv1x1", before_shape, "=>", shape_format(x.shape))
                    before_shape = shape_format(x.shape)

            output_dict["stage%d" % stage_id] = x = stage(x)

            if PRINTED_FORWARDS:
                print("backbone", "stage%d:" % stage_id, before_shape, "=>", shape_format(x.shape))
                before_shape = shape_format(x.shape)

        output_dict["stage_final"] = x
        return output_dict


def efficientvit_backbone_b0(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[8, 16, 32, 64, 128],
        depth_list=[1, 2, 2, 2, 2],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


def efficientvit_backbone_b1(verbose=False,**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        verbose=verbose,
        width_list=[16, 32, 64, 128, 256],
        depth_list=[1, 2, 3, 3, 4],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


def efficientvit_backbone_b2(verbose=False,**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        verbose=verbose,
        width_list=[24, 48, 96, 192, 384],
        depth_list=[1, 3, 4, 4, 6],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


def efficientvit_backbone_b3(verbose=False,**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        verbose=verbose,
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 4, 6, 6, 9],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone

#################################################################################
#                             Basic Layers                                      #
#################################################################################

class EfficientViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        heads_ratio: float = 1.0,
        dim=32,
        expand_ratio: float = 4,
        scales=(5,),
        norm="bn2d",
        act_func="hswish",
    ):
        super(EfficientViTBlock, self).__init__()
        self.context_module = ResidualBlock(
            LiteMLA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=dim,
                norm=(None, norm),
                scales=scales,
            ),
            IdentityLayer(),
        )
        local_module = MBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            expand_ratio=expand_ratio,
            use_bias=(True, True, False),
            norm=(None, None, norm),
            act_func=(act_func, act_func, None),
        )
        self.local_module = ResidualBlock(local_module, IdentityLayer())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.local_module(x)
        return x




class DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super(DSConv, self).__init__()

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.depth_conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.point_conv = ConvLayer(
            in_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm=("bn2d", "bn2d", "bn2d"),
        act_func=("relu6", "relu6", None),
        verbose = False,
    ):
        super(MBConv, self).__init__()

        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)
        if verbose: print("\ ", "MBConv (in:", in_channels, "mid:", mid_channels, "out:", out_channels, ")")

        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
            use_bias=use_bias[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class FusedMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        groups=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
        verbose=False,
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = mid_channels or round(in_channels * expand_ratio)
        if verbose: print("\ ", "FusedMBConv (in:", in_channels, "mid:", mid_channels, "out:", out_channels, ")")

        self.spatial_conv = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            groups=groups,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.conv1 = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.conv2 = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LiteMLA(nn.Module):
    r"""Lightweight multi-scale linear attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int or None = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales: tuple[int, ...] = (5,),
        eps=1.0e-15,
    ):
        super(LiteMLA, self).__init__()
        self.eps = eps
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    @autocast(enabled=False)
    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        if qkv.dtype == torch.float16:
            qkv = qkv.float()

        qkv = torch.reshape(
            qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        qkv = torch.transpose(qkv, -1, -2)
        q, k, v = (
            qkv[..., 0 : self.dim],
            qkv[..., self.dim : 2 * self.dim],
            qkv[..., 2 * self.dim :],
        )

        # lightweight linear attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        # linear matmul
        trans_k = k.transpose(-1, -2)

        # Original code:
        v = F.pad(v, (0, 1), mode="constant", value=1)
        # # Note: "v = F.pad(v, (0, 1), mode="constant", value=1)" likely causes problems with ONNX
        # # Constant folding - Only steps=1 can be constant folded for opset >= 10 onnx::Slice op. Constant folding not applied.
        # # We can temporarily replace the functionality with:
        # v_padding = torch.ones_like(v[:, :, :, [-1]])
        # v = torch.cat([v, v_padding], dim=-1)

        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + self.eps)

        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)

        out = self.relu_linear_att(multi_scale_qkv)
        out = self.proj(out)

        return out


class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class UpSampleLayer(nn.Module):
    def __init__(
        self,
        mode="bicubic",
        size: int or tuple[int, int] or list[int] or None = None,
        factor=2,
        align_corners=False,
    ):
        super(UpSampleLayer, self).__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self.size is not None and tuple(x.shape[-2:]) == self.size) or self.factor == 1:
            return x
        return resize(x, self.size, self.factor, self.mode, self.align_corners)


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0,
        norm="bn2d",
        act_func="relu",
    ):
        super(ConvLayer, self).__init__()

        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x



class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: nn.Module or None,
        shortcut: nn.Module or None,
        post_act=None,
        pre_norm: nn.Module or None = None,
    ):
        super(ResidualBlock, self).__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
        return res


class OpSequential(nn.Module):
    def __init__(self, op_list: list[nn.Module or None]):
        super(OpSequential, self).__init__()
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.ModuleList(valid_op_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x
