# HyperSegFormer model
# - info: HyperSegFormer https://doi.org/10.1109/JSTARS.2025.3557527
# - input: 86 bands of data
# - model: HyperSegFormer with adjustments (here Conv + Up + Stride)

echo "Remember to change paths for CODEBASE, MODELS and DATASET!"

python -m scripts.evaluate dataset.input_products.specific_products=[] \
          dataset.input_products.band_ranges=[[400,2500]] \
          model.architecture="segformer" model.multiply_loss_by_mag1c=False \
          dataset.feature_folder="/CODEBASE/HyperspectralViTs/parameters" \
          dataset.normalisation.mode_input="from_data" dataset.normalisation.save_load_from="normaliser_computed_emit_5perc_400_2500_full" \
          dataset.normalisation.max_style="max_outliers" dataset.normalisation.max_outlier_percentile=5 \
          model.positive_weight=1 experiment_name="SegF_ConvUpStride_EMIT_b0_64x64_batch16_L_BCE___DEMO_MINI_EVAL" \
          dataloader.num_workers=8 dataloader.train_batch_size=16 dataloader.val_batch_size=16  \
          training.val_check_interval=0.5 \
          dataset.format="EMIT" model.weighted_random_sampler=True \
          training.visualiser.bands=['rgb','labelbinary','prediction','differences'] \
          dataset.train_csv="synth_train_mini5.csv" dataset.test_csv="synth_test_mini5.csv" \
          dataset.root_folder="/DATASETS/OxHyperSyntheticCH4_MINI" \
          dataset.tiler.tile_size=64 dataset.tiler.tile_overlap=32 \
          model.transformer.backbone="nvidia/mit-b0" \
          model.transformer.pretrained=False \
          model.transformer.custom_config_features.conv1x1=True \
          model.transformer.custom_config_features.upscale=True \
          model.transformer.custom_config_features.stride=True \
          model.log_train_images_every_batch_i=5000 \
          model.auto_continue=True \
          model.transformer.custom_config.loss_overrides.multilabel_override=True \
          model.transformer.custom_config.loss_overrides.loss_override_to="BCEWithLogitsLoss" \
          model.load_path="/MODELS/SegF_ConvUpStride_EMITv2_b0_64x64_batch16_L_BCE_R1/final_checkpoint_model_50ep.ckpt" \
          evaluation.plot_save=True evaluation.plot_show=True

# HyperSegFormer
# Stride ON
# Up ON
# 1x1 Conv ON
