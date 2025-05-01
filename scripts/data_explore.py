# Creates data loaders using all the other modules and correct nesting, presents basic visualisation of the data
# - works as a standalone example of how to use the data in this project
#   later same scripts will be used with models and overall training process

import matplotlib
from hyper.utils import check_headless
if check_headless():
    matplotlib.use('agg')

import hydra
from omegaconf import DictConfig
from hyper.data.datamodule import HyperDataModule

@hydra.main(version_base=None, config_path=".", config_name="settings")
def main(settings : DictConfig) -> None:
    # load larger area, so that with rotations we don't have no-data corners
    settings.dataset.augment_rotation_load_surrounding_area = 0.4

    data_module = HyperDataModule(settings)
    data_module.prepare_data()
    data_module.summarise()

    # Choose one:
    train_dataloader = data_module.train_dataloader(batch_size=4, num_workers=1)
    train_dataset = data_module.train_dataset
    test_dataloader = data_module.test_dataloader(batch_size=4, num_workers=1)
    test_dataset = data_module.test_dataset

    print("TRAIN DATASET")
    for batch_data in train_dataloader:
        print("plume!, batch size", len(batch_data), batch_data.keys(), "x", batch_data["x"].shape, "y",
              batch_data["y"].shape)
        if "qplume_fulltile" in batch_data.keys():
            print("qplume values:", batch_data["qplume_fulltile"])
            print("ids:", batch_data["id"])
            train_dataset.show_item_from_data(batch_data)
        break

    print("TEST DATASET")
    for batch_data in test_dataloader:
        print("plume!, batch size", len(batch_data), batch_data.keys(), "x", batch_data["x"].shape, "y",
              batch_data["y"].shape)
        if "qplume_fulltile" in batch_data.keys():
            print("qplume values:", batch_data["qplume_fulltile"])
            print("ids:", batch_data["id"])
            test_dataset.show_item_from_data(batch_data)
        break


if __name__ == "__main__":
    main()