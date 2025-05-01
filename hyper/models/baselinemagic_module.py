import torch
import torch.nn
import numpy as np
import pytorch_lightning as pl
from kornia.morphology import dilation as kornia_dilation
from kornia.morphology import erosion as kornia_erosion

def binary_opening(x:torch.Tensor, kernel:torch.Tensor)-> torch.Tensor:
    eroded = torch.clamp(kornia_erosion(x.float(), kernel), 0, 1) > 0
    return torch.clamp(kornia_dilation(eroded.float(), kernel), 0, 1) > 0

class BaselineMagicModule(pl.LightningModule):

    def __init__(self, settings, visualiser, mag1c_band = 0):
        super().__init__()

        self.visualiser = visualiser
        self.settings = settings

        self.task = settings.model.task # only segmentation makes sense here
        self.mag1c_threshold = int(settings.model.classical_baseline.threshold) # 500

        # locate the mag1c band:
        # possible_magic_bands = ["mag1c", "B_magic30_tile", "B_rmf"]
        # self.band_mag1c = settings.dataset.input_products.specific_products.index("mag1c")
        self.band_mag1c = mag1c_band

        self.element_stronger = torch.nn.Parameter(torch.from_numpy(np.array([[0,1,0],
                                                                              [1,1,1],
                                                                              [0,1,0]])).float(),requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mag1c = x[:, self.band_mag1c:(self.band_mag1c+1)]
        mag1c_thresholded = (mag1c > self.mag1c_threshold)
        mag1c_opened = binary_opening(mag1c_thresholded, self.element_stronger).long()

        # in eval we compare against 0 with <=, so here do
        eps = 0.5
        return 2*(mag1c_opened - eps) # as the values are 0 or 1 otherwise

    def summarise(self):
        print("Model: Baseline magic using the mag1c threshold of", self.mag1c_threshold, "and looking for the mag1c band at index", self.band_mag1c)