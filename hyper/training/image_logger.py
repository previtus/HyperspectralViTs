# Simple implementation of image logging
# ... will be called from each model module during training_step (if (batch_idx % 100) == 0)
#     and validation_step (at the end)
# calls visualisation of few random samples, showing all input, output and used helpful products

# One version for pytorch_lightning available in
# https://github.com/spaceml-org/STARCOP/blob/main/starcop/data/data_logger.py

import wandb

def log_images_from_batch(mode, batch, predictions, batch_idx, model):
    settings = model.settings
    visualiser = model.visualiser

    plt, fig = visualiser.plot_batch(batch, predictions)

    logging_target = settings.training.visualiser.target
    if logging_target == 'wandb':
        wandb.log({mode+'_batch': [wandb.Image(fig)]})
    else:
        assert False, "Logging not implemented yet for logging_target="+logging_target
    plt.close()
