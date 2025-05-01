# Special small efficient models exploration
# - LinkNet models used in "Optimizing Methane Detection Onboard Satellites: Speed, Accuracy, and Low-Power Solutions
#                           for Resource-Constrained Hardware"

import torch.nn
import segmentation_models_pytorch as smp
from hyper.models.model_utils import num_of_params

from hyper.models.hyperstarcop_module import HyperStarcopModule

class LinkNetModule(HyperStarcopModule):

    def __init__(self, settings, visualiser):
        super().__init__(settings, visualiser)
        self.model_desc = "LinkNet"

    def create_network(self):
        pretrained = None if self.hyperstarcop_settings.pretrained == 'None' else self.hyperstarcop_settings.pretrained
        encoder_name=self.hyperstarcop_settings.backbone
        model = smp.Linknet(encoder_name=encoder_name, encoder_weights=pretrained, in_channels=self.num_channels, classes=self.num_classes)
        return model

    def summarise(self):
        pretrained = "(not pretrained)" if self.hyperstarcop_settings.pretrained == 'None' else "(pretrained on "+self.hyperstarcop_settings.pretrained+")"

        params = num_of_params(self.network)
        M_params = round(params / 1000 / 1000, 2)
        print("[Model] LinkNet model with the", self.hyperstarcop_settings.backbone, "backbone", pretrained,
              "with", str(M_params)+"M parameters.")
        print("- Input num channels =", self.num_channels, ", Output num classes=", self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # simplified, no current epoch as with the custom unet...
        return self.network(x)

    def printed_forward(self, x: torch.Tensor) -> torch.Tensor:
        # fallback to simple:
        return self.forward(x)

