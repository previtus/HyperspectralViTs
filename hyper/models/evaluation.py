import os
import torch
import torchmetrics
from tqdm import tqdm
from hyper.utils import to_device, CustomJSONEncoder
from hyper.training import metrics
import json
import numpy as np
from hyper.data.data_utils import mkdir

EASY_HARD_THRESHOLD = 1000
NUMBER_OF_PIXELS_PRED = 10 # ~ val we used in Starcop

def save_results(results, path):
    with open(path, "w") as fh:
        json.dump(results, fh, cls=CustomJSONEncoder)
    print("... saved results to:",path)

@torch.no_grad()
def evaluate_datamodule(model, data_module, settings, plotting=False, add_to_name=""):
    experiment_path = settings["experiment_path"]
    # note: do we need to maintain the batch size of 1?

    has_validation = False  # Then it just uses train + test
    if settings.dataset.val_csv != "":
        has_validation = True  # Will also use val

    results_test = "results_" + str(settings.dataset.test_csv).replace(".csv", "") + add_to_name + ".json"
    results_val = "results_" + str(settings.dataset.val_csv).replace(".csv", "") + add_to_name + ".json"
    results_train = "results_" + str(settings.dataset.train_csv).replace(".csv", "") + add_to_name + ".json"

    save_plots_to = ""
    plot_show = settings.evaluation.plot_show
    plot_save = settings.evaluation.plot_save

    if settings.evaluation.test:
        if plot_save:
            save_plots_to = "plots_test"
        print("Running evaluation on the test dataset:")
        test_dataloader = data_module.test_dataloader(batch_size=1)
        # or run_evaluation_masked
        run_evaluation(model, test_dataloader, settings, plotting=plotting,
                                        save_results_to=os.path.join(experiment_path, results_test), plot_show=plot_show, save_plots_to=save_plots_to)

    if has_validation and settings.evaluation.val:
        if plot_save:
            save_plots_to = "plots_val"
        print("Running evaluation on the val dataset:")
        val_dataloader = data_module.val_dataloader(batch_size=1)
        # or run_evaluation_masked
        run_evaluation(model, val_dataloader, settings, plotting=plotting,
                                        save_results_to=os.path.join(experiment_path, results_val), plot_show=plot_show, save_plots_to=save_plots_to)

    if settings.evaluation.train:
        if plot_save:
            save_plots_to = "plots_train"
        print("Running evaluation on the training dataset:")
        data_module.weighted_random_sampler = False
        train_dataloader = data_module.train_dataloader(batch_size=1, shuffle=False)
        # or run_evaluation_masked
        run_evaluation(model, train_dataloader, settings, plotting=plotting,
                                        save_results_to=os.path.join(experiment_path, results_train), plot_show=plot_show, save_plots_to=save_plots_to)


@torch.no_grad()
def run_evaluation(model, dataloader, settings, plotting=True, save_results_to="", save_plots_to="", plot_show=True, override_device = None, override_threshold = None):
    model.eval() # !
    visualiser = model.visualiser
    device = model.device

    outputs = []
    labels = []

    task = settings.model.task # "segmentation" / "regression"

    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = to_device(batch, device)
        x, y = batch["x"], batch["y"]
        predictions = model.forward(x) # (B, 1, H, C)
        y_long = y.long() # (B, 1, H, C)

        outputs.append(predictions.detach())

        if task == "segmentation":
            labels.append(y_long.detach())
        elif task == "regression":
            labels.append(y.detach())

        if plotting:
            plt, fig = visualiser.plot_batch(batch, predictions)

            if save_plots_to != "":
                name_string = ""
                if "id" in batch.keys():
                    name_string += "_id_" + "-".join(batch["id"])
                name_string += "_batch_"+str(idx).zfill(3)

                mkdir(save_plots_to)
                plt.savefig(os.path.join(save_plots_to, "plot_"+name_string+".png"), dpi=600)
            if plot_show:
                plt.show()
            plt.close()

    all_preds = torch.cat(outputs, dim=0)
    all_y = torch.cat(labels, dim=0)

    if override_device is not None:
        print("Now moving everything to the device", override_device)
        device = override_device
        all_preds = all_preds.to(device)
        all_y = all_y.to(device)

    if task == "segmentation":
        metric_functions = metrics.METRICS_CONFUSION_MATRIX + [metrics.TP, metrics.TN, metrics.FP, metrics.FN, metrics.iou,
                                                               metrics.balanced_accuracy, metrics.cohen_kappa, "AUPRC"]
        for value in settings.model.extra_metrics: metric_functions.append(value)

        metrics_iter = evaluation_metrics_segmentation(all_y, all_preds, device, metric_functions = metric_functions, override_threshold = override_threshold)

    # elif task == "regression":
    #     metric_functions = ["MSE", "MAE"]
    #     metrics_iter = evaluation_metrics_regression(all_y, all_preds, device, metric_functions = metric_functions)

    print(metrics_iter)
    print("Final metrics:")
    for key in metrics_iter.keys():
        print(key,":", metrics_iter[key])

    # free memory
    outputs.clear()
    labels.clear()

    # Save results
    if save_results_to != "":
        save_results(metrics_iter, save_results_to)

def to_cpu_if_needed(x):
    if "cuda" in str(x.device):
        return x.detach().cpu()
    return x

def torch_where_logical(cond, x_1, x_2):
    return (cond * x_1) + (torch.logical_not(cond) * x_2)

def erode_k_times_cpu(k, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    from kornia.morphology import erosion as kornia_erosion

    for i in range(k):
        x = torch.clamp(kornia_erosion(x.float(), kernel), 0, 1) > 0

    # to replace:
    x = to_cpu_if_needed(x)
    np_inter = np.where(x == True, 1, 0)
    x = torch.from_numpy(np_inter)
    return x

def erode_k_times_gpu(k, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    from kornia.morphology import erosion as kornia_erosion

    for i in range(k):
        x = torch.clamp(kornia_erosion(x.float(), kernel), 0, 1) > 0

    x = torch_where_logical(x == True, 1, 0)
    return x


@torch.no_grad()
def run_evaluation_masked(model, dataloader, settings, plotting=True, erode_mask=0, save_results_to="", save_plots_to="", override_threshold = None):
    model.eval() # !
    visualiser = model.visualiser
    device = model.device

    outputs = []
    labels = []

    task = settings.model.task # "segmentation" / "regression"

    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = to_device(batch, device)
        x, y = batch["x"], batch["y"]

        predictions = model.forward(x) # (B, 1, H, C)
        y_long = y.long() # (B, 1, H, C)

        # mask out invalid areas ...
        valid_mask = batch["valid_mask"]

        # optionally, grow the area of invalid pixels
        if erode_mask > 0 and torch.min(valid_mask) == 0:
            element_stronger = torch.nn.Parameter(torch.from_numpy(np.array([[0, 1, 0],
                                                                             [1, 1, 1],
                                                                             [0, 1, 0]])).float(), requires_grad=False)
            element_stronger = element_stronger.to(valid_mask.device)
            valid_mask = erode_k_times_gpu(erode_mask, valid_mask, element_stronger)

        # we can't use nans - as the confusion matrix is confused
        nodata_y = 0 # no event
        if task == "segmentation":
            y_long = torch_where_logical(valid_mask == 0, nodata_y, y_long)  # w,h
            labels.append(y_long)

        elif task == "regression":
            y = torch_where_logical(valid_mask == 0, nodata_y, y)  # w,h
            labels.append(y)

        nodata_pred = -1 # no event
        predictions = torch_where_logical(valid_mask == 0, nodata_pred, predictions)  # w,h
        outputs.append(predictions)

        if plotting:
            plt, fig = visualiser.plot_batch(batch, predictions)
            if save_plots_to != "":
                plt.savefig(os.path.join(save_plots_to, "plot_batch_"+str(idx).zfill(3)+".png"), dpi=600)
            plt.show()
            plt.close()

    all_preds = torch.cat(outputs, dim=0)
    all_y = torch.cat(labels, dim=0)

    # move all on cpu - also more likely to fit ...
    all_preds = to_cpu_if_needed(all_preds)
    all_y = to_cpu_if_needed(all_y)
    cpu_device = torch.device("cpu")

    if task == "segmentation":
        metric_functions = metrics.METRICS_CONFUSION_MATRIX + [metrics.TP, metrics.TN, metrics.FP, metrics.FN, metrics.iou,
                                                               metrics.balanced_accuracy, metrics.cohen_kappa, "AUPRC"]
        for value in settings.model.extra_metrics: metric_functions.append(value)
        metrics_iter = evaluation_metrics_segmentation(all_y, all_preds, cpu_device, metric_functions = metric_functions, override_threshold = override_threshold)

    # elif task == "regression":
    #     metric_functions = ["MSE", "MAE"]
    #     metrics_iter = evaluation_metrics_regression(all_y, all_preds, cpu_device, metric_functions = metric_functions)

    print(metrics_iter)
    print("Final metrics:")
    for key in metrics_iter.keys():
        print(key,":", metrics_iter[key])

    # free memory
    outputs.clear()
    labels.clear()

    # Save results
    if save_results_to != "":
        save_results(metrics_iter, save_results_to)


@torch.no_grad()
def evaluation_metrics_segmentation(y_long, predictions, device, metric_functions = metrics.METRICS_CONFUSION_MATRIX, override_threshold = None):
    # to be called in on_validation_epoch_end() function
    # note: prediction threshold is at 0 here...
    if override_threshold is None:
        pred_binary = (predictions >= 0).long() # ~ (B, 1, H, C)
    else:
        print("Thresholding using a custom thr=",override_threshold,"(default one is 0)")
        pred_binary = (predictions >= override_threshold).long()  # ~ (B, 1, H, C)
    # y_long ~ (B, 1, H, C)

    # Per-pixel results
    confusion_matrix = torchmetrics.ConfusionMatrix(task='binary', num_classes=2)
    confusion_matrix.to(device)

    confusion_matrix_easy = torchmetrics.ConfusionMatrix(task='binary', num_classes=2)
    confusion_matrix_easy.to(device)

    confusion_matrix_hard = torchmetrics.ConfusionMatrix(task='binary', num_classes=2)
    confusion_matrix_hard.to(device)

    # Per-tile results
    confusion_matrix_per_tile = torchmetrics.ConfusionMatrix(task='binary', num_classes=2)
    confusion_matrix_per_tile.to(device)

    # note ~ doesn't like nan values ...
    confusion_matrix.update(pred_binary, y_long)

    # Re-compute labels of difficulty
    # We do this here, because then we can just use the info in y's.
    # In theory we could recompute these in the csv's, but then we'd still have to send it through...
    y_flat = torch.flatten(y_long, start_dim=1) # (B, H*C)
    label_pixels_plume = torch.sum(y_flat, 1) # (B)
    tile_has_plume = label_pixels_plume > 0
    difficulty = ["easy" if x > EASY_HARD_THRESHOLD else "hard" for x in label_pixels_plume]

    # This also includes only samples with plume!
    easy_idx = [i for i, x in enumerate(difficulty) if x == "easy" and tile_has_plume[i]]
    hard_idx = [i for i, x in enumerate(difficulty) if x == "hard" and tile_has_plume[i]]

    if 'easy' in difficulty:
        confusion_matrix_easy.update(pred_binary[easy_idx], y_long[easy_idx])
    if 'hard' in difficulty:
        confusion_matrix_hard.update(pred_binary[hard_idx], y_long[hard_idx])

    metrics_iter = {}

    # Extra metrics that require all the data:
    if "AUPRC" in metric_functions:
        # run AUPRC metric on all, on easy and on hard ...
        print("calculating AUPRC...")
        auprc_all = metrics.auprc(true_changes=y_long, pred_change_scores=predictions)
        metrics_iter["auprc"] = auprc_all

        auprc_easy = metrics.auprc(true_changes=y_long[easy_idx], pred_change_scores=predictions[easy_idx])
        metrics_iter["auprc_easy"] = auprc_easy

        auprc_hard = metrics.auprc(true_changes=y_long[hard_idx], pred_change_scores=predictions[hard_idx])
        metrics_iter["auprc_hard"] = auprc_hard

    # Extra metrics we made for multi-hot labels:
    if "multihot_f1score" in metric_functions:
        multihot_f1_perclass = metrics.multihot_f1score(y_long, pred_binary)
        for class_i in range(len(multihot_f1_perclass)):
            metrics_iter["multihot_f1_class_"+str(class_i)] = multihot_f1_perclass[class_i]
    if "multihot_precision" in metric_functions:
        multihot_precision_perclass = metrics.multihot_precision(y_long, pred_binary)
        for class_i in range(len(multihot_precision_perclass)):
            metrics_iter["multihot_precision_class_"+str(class_i)] = multihot_precision_perclass[class_i]
    if "multihot_recall" in metric_functions:
        multihot_recall_perclass = metrics.multihot_recall(y_long, pred_binary)
        for class_i in range(len(multihot_recall_perclass)):
            metrics_iter["multihot_recall_class_"+str(class_i)] = multihot_recall_perclass[class_i]

    # Tile classification
    pred_flat = torch.flatten(pred_binary, start_dim=1) # (B, H*C)
    pred_pixels_plume = torch.sum(pred_flat, 1) # (B)
    tile_pred_has_plume = pred_pixels_plume > NUMBER_OF_PIXELS_PRED
    confusion_matrix_per_tile.update(tile_pred_has_plume, tile_has_plume)


    cm = confusion_matrix.compute()

    cm_easy = confusion_matrix_easy.compute()
    cm_hard = confusion_matrix_hard.compute()
    for fun in metric_functions:
        if fun == "AUPRC": continue
        if fun == "multihot_f1score": continue
        if fun == "multihot_precision": continue
        if fun == "multihot_recall": continue

        metrics_iter[fun.__name__] = fun(cm).item()

        metrics_iter[fun.__name__ + "_easy"] = fun(cm_easy).item()
        metrics_iter[fun.__name__ + "_hard"] = fun(cm_hard).item()

    cm_per_tile_classification = confusion_matrix_per_tile.compute()
    metrics_iter["tile_FPR"] = metrics.FPR(cm_per_tile_classification).item()

    # print(metrics_iter)
    return metrics_iter

@torch.no_grad()
def evaluation_metrics_regression(y, predictions, device, metric_functions):
    # to be called in on_validation_epoch_end() function

    # predictions ~ (batch, channels, W, H)
    # y ~ (batch, channels, H, C)
    print("y.shape", y.shape)
    print("predictions.shape", predictions.shape)
    metrics_iter = {}

    # Extra metrics that require all the data:
    if "MSE" in metric_functions:
        print("calculating MSE/L2...")
        mse_all = metrics.mse(y=y, pred=predictions)
        metrics_iter["MSE_metric"] = mse_all

    if "MAE" in metric_functions:
        print("calculating MAE/L1...")
        mae_all = metrics.mae(y=y, pred=predictions)
        metrics_iter["MAE_metric"] = mae_all
    return metrics_iter
