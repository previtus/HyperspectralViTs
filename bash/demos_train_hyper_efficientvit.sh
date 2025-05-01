# HyperEfficientViT model
# - info: HyperEfficientViT https://doi.org/10.1109/JSTARS.2025.3557527
# - input: 86 bands of data
# - model: HyperEfficientViT with adjustments (here Conv + Up)

echo "Remember to change paths for CODEBASE and DATASETS!"

python -m scripts.train dataset.input_products.specific_products=[] \
          dataset.input_products.band_ranges=[[400,2500]] \
          model.architecture="efficientvit" model.multiply_loss_by_mag1c=False \
          dataset.feature_folder="/CODEBASE/HyperspectralViTs/parameters" \
          dataset.normalisation.mode_input="from_data" dataset.normalisation.save_load_from="normaliser_computed_emit_5perc_400_2500_full" \
          dataset.normalisation.max_style="max_outliers" dataset.normalisation.max_outlier_percentile=5 \
          model.positive_weight=1 experiment_name="HyperEffViT_ConvUp_EMIT" \
          dataloader.num_workers=8 dataloader.train_batch_size=16 dataloader.val_batch_size=16  \
          training.val_check_interval=0.5 training.max_epochs=50 \
          dataset.format="EMIT" model.weighted_random_sampler=True \
          training.visualiser.bands=['rgb','labelbinary','prediction','differences'] \
          dataset.train_csv="train_filtered_v2.csv" dataset.val_csv="val_filtered_v2.csv" dataset.test_csv="test_filtered_v2.csv" \
          dataset.root_folder="/DATASETS/OxHyperSyntheticCH4" \
          dataset.tiler.tile_size=64 dataset.tiler.tile_overlap=32 \
          model.efficientvit.backbone="b1" \
          model.efficientvit.custom_config.conv1x1=True \
          model.efficientvit.custom_config.upscale_layers=1 \
          model.lr=0.00006 \
          model.log_train_images_every_batch_i=5000 \
          model.auto_continue=True
