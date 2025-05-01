# HyperSTARCOP model
# - info: https://www.nature.com/articles/s41598-023-44918-6
# - input: RGB+MF
# - model: U-NET

# Note: for these commands only the RGB and mag1c bands are needed

## EMIT dataset:
python -m scripts.train dataset.input_products.specific_products=[641,551,462,"B_magic30_tile"] \
          dataset.format="EMIT" \
          model.architecture="unet" model.hyperstarcop.custom_config="None" \
          model.positive_weight=1 experiment_name="HyperSTARCOP_magic_rgb_EMIT" \
          dataloader.num_workers=8 training.val_check_interval=0.5 training.max_epochs=50 \
          model.multiply_loss_by_mag1c=False model.weighted_random_sampler=True \
          training.visualiser.bands=['rgb','labelbinary','mag1c','prediction','differences'] \
          dataset.root_folder="/DATASETS/OxHyperSyntheticCH4" \
          dataset.train_csv="train_filtered_v2.csv" \
          dataset.val_csv="val_filtered_v2.csv" \
          dataset.test_csv="test_filtered_v2.csv" \
          dataset.tiler.tile_size=64 dataset.tiler.tile_overlap=32

# AVIRIS dataset:
python -m scripts.train dataset.input_products.specific_products=[640,550,460,"mag1c"] \
          dataset.format="AVIRIS" \
          model.architecture="unet" model.hyperstarcop.custom_config="None" \
          model.positive_weight=1 experiment_name="HyperSTARCOP_magic_rgb" \
          dataloader.num_workers=4 training.val_check_interval=0.5 training.max_epochs=50 \
          model.multiply_loss_by_mag1c=True model.weighted_random_sampler=True \
          training.visualiser.bands=['rgb','labelbinary','mag1c','prediction','differences'] \
          dataset.root_folder="/DATASETS/STARCOP_allbands" \
          dataset.train_csv="train.csv" \
          dataset.test_csv="test.csv" \
