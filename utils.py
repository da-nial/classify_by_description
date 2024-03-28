import yaml
import torch
import torchmetrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import filters


def compare_predictions(dclip_predictions, clip_predictions, labels):
    n_classes = labels.size(0)
    class_counters = {
        "both_correct": torch.zeros(n_classes),
        "both_wrong": torch.zeros(n_classes),
        "only_clip_correct": torch.zeros(n_classes),
        "only_descr_correct": torch.zeros(n_classes),
        "intersection": torch.zeros(n_classes),
        "intersection_correct": torch.zeros(n_classes),
    }

    for class_idx in range(n_classes):
        class_mask = (labels == class_idx)

        class_counters["both_correct"][class_idx] += (clip_predictions == labels).logical_and(
            dclip_predictions == labels
        ).logical_and(class_mask).sum().item()

        class_counters["both_wrong"][class_idx] += (clip_predictions != labels).logical_and(
            dclip_predictions != labels
        ).logical_and(class_mask).sum().item()

        class_counters["only_clip_correct"][class_idx] += (clip_predictions == labels).logical_and(
            dclip_predictions != labels
        ).logical_and(class_mask).sum().item()

        class_counters["only_descr_correct"][class_idx] += (clip_predictions != labels).logical_and(
            dclip_predictions == labels
        ).logical_and(class_mask).sum().item()

        class_counters["intersection"][class_idx] += (clip_predictions == dclip_predictions).logical_and(
            class_mask
        ).sum().item()

        class_counters["intersection_correct"][class_idx] += (clip_predictions == dclip_predictions).logical_and(
            clip_predictions == labels
        ).logical_and(class_mask).sum().item()

    return class_counters


def get_metrics(n_classes):
    acc_metric = torchmetrics.Accuracy(num_classes=n_classes, task="multiclass")
    acc_metric_top5 = torchmetrics.Accuracy(num_classes=n_classes, task="multiclass", top_k=5)
    acc_metric_per_cls = torchmetrics.ClasswiseWrapper(
        torchmetrics.Accuracy(num_classes=n_classes, task="multiclass", average=None)
    )
    return [acc_metric, acc_metric_top5, acc_metric_per_cls]


def pretty_print(d: dict):
    print(yaml.dump(d, allow_unicode=True, default_flow_style=False))


def plot_acc_per_cls(clip_acc_per_cls: dict, dclip_acc_per_cls: dict):
    res_per_cls = {
        'clip_acc': [round(clip_acc_per_cls[cls_idx].item(), 3) for cls_idx in clip_acc_per_cls.keys()],
        'dclip_acc': [round(dclip_acc_per_cls[cls_idx].item(), 3) for cls_idx in dclip_acc_per_cls.keys()],
    }
    df = pd.DataFrame.from_dict(res_per_cls)
    df['diff'] = df['dclip_acc'] - df['clip_acc']

    plt.figure(figsize=(16, 9), dpi=900)
    plt.bar(df.index, df['diff'], color='blue')
    plt.title('D-CLIP Improvement Per Class')
    plt.xlabel('Index')
    plt.ylabel('D-CLIP Acc - CLIP Acc')
    plt.ylim(-1.1, 1.1)  # Adjusting the y-axis limits to ensure -1 to 1 range is visible

    plt.xticks(rotation=90)
    plt.show()


def plot_cam_similarity_over_acc_improvement(
        clip_acc_per_cls: dict,
        dclip_acc_per_cls: dict,
        cam_sims_per_cls: dict
):
    res_per_cls = {
        'clip_acc': [round(clip_acc_per_cls[cls_idx].item(), 3) for cls_idx in clip_acc_per_cls.keys()],
        'dclip_acc': [round(dclip_acc_per_cls[cls_idx].item(), 3) for cls_idx in dclip_acc_per_cls.keys()],
    }
    df = pd.DataFrame.from_dict(res_per_cls)
    df['acc_diff'] = df['dclip_acc'] - df['clip_acc']

    cam_sims_df = pd.DataFrame.from_dict(cam_sims_per_cls)
    df['cam_sim_mean'] = cam_sims_df.apply(lambda row: row.mean(), axis=1)

    plt.plot([-1, 1], [-1, 1], 'r--')

    for i, (x, y) in enumerate(zip(df['acc_diff'], df['cam_sim_mean'])):
        plt.text(x, y, i, fontsize=8)

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    plt.xlabel('Description Method Improvement')
    plt.ylabel('Cam Similarity (Average)')
    plt.title('Each point: Acc Improvement & Avg CAM Sim for that class')
    plt.show()


def normalize(x: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1].
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x


def get_att_map(img, attn_map, blur=True):
    """
    Modified from: https://github.com/salesforce/ALBEF/blob/main/visualization.ipynb
    """
    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02 * max(img.shape[:2]))
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap('jet')
    attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map = 1 * (1 - attn_map ** 0.7).reshape(attn_map.shape + (1,)) * img + \
               (attn_map ** 0.7).reshape(attn_map.shape + (1,)) * attn_map_c
    return attn_map


def display_cams(img, cams: list, blur=True):
    # img = np.transpose(img, (1, 2, 0))  # Rearrange dimensions to (224, 224, 3)
    img = normalize(img)
    cams = [normalize(cam) for cam in cams]
    num_plots = len(cams) + 1

    _, axes = plt.subplots(1, num_plots, figsize=(num_plots * 2, 3))
    axes[0].imshow(img)
    for i in range(num_plots - 1):
        axes[i + 1].imshow(get_att_map(img, cams[i], blur))

    for ax in axes:
        ax.axis("off")
    plt.show()

    plt.cla()
    plt.clf()
    plt.close()
