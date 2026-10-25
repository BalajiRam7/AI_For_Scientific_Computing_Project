import numpy as np
import torch
from torch.utils.data import DataLoader

from src.common.visualization.visualization import (
    plot_sample_grid_1d,
    plot_sample_grid_2d,
    plot_sample_grid_2dvec,
)


@torch.no_grad()
def generate_random_sample_grid_1d(
    model,
    stream,
    num_samples_per_exp=5,
    seed=42,
    list_of_exp_indices=None,
):
    device = next(model.parameters()).device

    if list_of_exp_indices is None:
        exp_list = range(len(stream))
    else:
        exp_list = list_of_exp_indices

    samples = []
    for exp_idx in exp_list:
        exp = stream[exp_idx]
        generator = torch.Generator(torch.device("cpu"))
        generator.manual_seed(seed)
        dl = DataLoader(
            exp.test_dataset,
            batch_size=num_samples_per_exp,
            shuffle=True,
            generator=generator,
        )

        # Geenerate random samples
        batch = next(iter(dl))
        x, y = batch["input"], batch["output"]
        y = y[:, 0, :]  # TODO: Make sure this works for all datasets
        preds = model(x.to(device), aux_input=batch["aux_input"])
        preds = preds.detach().cpu().squeeze().numpy()
        gts = y.detach().cpu().squeeze().numpy()

        samples_i = []
        for j in range(num_samples_per_exp):
            sample_ij = {}

            x_values = list(range(len(preds[j]))) * 2
            y_values = np.concatenate([preds[j], gts[j]], axis=0).tolist()
            labels = ["Prediction"] * len(preds[j]) + ["Ground Truth"] * len(gts[j])

            sample_ij["x"] = x_values
            sample_ij["y"] = y_values
            sample_ij["Label"] = labels
            samples_i.append(sample_ij)

        samples.append(samples_i)

    sample_keys = {"x": "x", "y": "y", "hue": "Label"}

    return samples, sample_keys


@torch.no_grad()
def generate_random_sample_grid_2d(
    model,
    stream,
    num_samples_per_exp=5,
    seed=42,
    list_of_exp_indices=None,
):
    device = next(model.parameters()).device

    if list_of_exp_indices is None:
        exp_list = range(len(stream))
    else:
        exp_list = list_of_exp_indices

    samples = []
    for exp_idx in exp_list:
        exp = stream[exp_idx]
        generator = torch.Generator(device=torch.device("cpu"))
        generator.manual_seed(seed)
        dl = DataLoader(
            exp.test_dataset,
            batch_size=num_samples_per_exp,
            shuffle=True,
            generator=generator,
        )

        # Geenerate random samples
        batch = next(iter(dl))
        x, y = batch["input"], batch["output"]
        preds = model(x.to(device), aux_input=batch["aux_input"])
        gts = y.detach().cpu().squeeze().numpy()[:, 0, :, :]
        preds = preds.detach().cpu().squeeze().numpy().reshape(gts.shape)

        # if len(x.shape) is 3, it assumes the tensor has shape B x L x D
        samples_i = []

        # For each sample, we have a dictionary with the keys "GT" and "Prediction"
        for j in range(num_samples_per_exp):
            gt_j = gts[j]
            pred_j = preds[j]
            sample_ij = {}
            sample_ij["GT"] = gt_j
            sample_ij["Prediction"] = pred_j

            samples_i.append(sample_ij)

        samples.append(samples_i)

    sample_keys = ["GT", "Prediction"]

    return samples, sample_keys


def generate_and_plot_sample_grid(model, stream):
    ds_name = stream[0].train_dataset.name

    if ds_name in [
        "advdiff1d",
        "burgers1d",
    ]:
        samples, sample_keys = generate_random_sample_grid_1d(model, stream)
        fig = plot_sample_grid_1d(samples, sample_keys)

    elif ds_name in [
        "advdiff2d",
        "navierstokes2d",
    ]:
        samples, sample_keys = generate_random_sample_grid_2d(model, stream)
        stats = stream[0].train_dataset.normalizer.stats
        fig = plot_sample_grid_2d(samples, sample_keys, stats)

    elif ds_name in [
        "burgers2d",
    ]:
        samples, sample_keys = generate_random_sample_grid_2d(model, stream)
        stats = stream[0].train_dataset.normalizer.stats
        fig = plot_sample_grid_2dvec(samples, sample_keys, stats)

    else:
        raise NotImplementedError()

    return fig
