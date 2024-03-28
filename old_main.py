from load import *
import torchmetrics
from tqdm import tqdm
import random
import pandas as pd

from pytorch_grad_cam_wrapper import CAMWrapper
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

seed_everything(hparams['seed'])

BASE_PATH = f'run_{random.randint(0, 100000)}'
os.makedirs(BASE_PATH, exist_ok=True)
print(f'BASE_PATH: {BASE_PATH}')

bs = hparams['batch_size']
# dataloader = DataLoader(dataset, bs, shuffle=True, num_workers=16, pin_memory=True)
dataloader = DataLoader(dataset, bs, shuffle=True, num_workers=8, pin_memory=False)

# for class_name, class_index in dataset.datasets[0].class_to_idx.items():
for class_name, class_index in dataset.class_to_idx.items():
    print(f"{class_index} > {class_name}")

print("Loading model...")

device = torch.device(hparams['device'])
# load model
model, preprocess = clip.load(hparams['model_size'], device=device, jit=False)
model.eval()
model.requires_grad_(False)

print("Encoding descriptions...")

description_encodings = compute_description_encodings(model)
torch.save(description_encodings, f'{BASE_PATH}/description_encodings.pth')
# description_encodings = torch.load('description_encodings.pth')

label_encodings = compute_label_encodings(model)
torch.save(label_encodings, f'{BASE_PATH}/label_encodings.pth')
# label_encodings = torch.load('label_encodings.pth')

"""
description_encodings: dict[label_name -> torch.tensor(num_descriptions_for_label, 1024)]
label_encodings: torch.tensor(num_classes, 1024)
"""
n_classes = len(description_encodings)

print("Evaluating...")
lang_accuracy_metric = torchmetrics.Accuracy(num_classes=n_classes, task="multiclass").to(device)
lang_accuracy_metric_top5 = torchmetrics.Accuracy(num_classes=n_classes, task="multiclass", top_k=5).to(device)

clip_accuracy_metric = torchmetrics.Accuracy(num_classes=n_classes, task="multiclass").to(device)
clip_accuracy_metric_top5 = torchmetrics.Accuracy(num_classes=n_classes, task="multiclass", top_k=5).to(device)

lang_accuracy_metric_per_cls = torchmetrics.ClasswiseWrapper(
    torchmetrics.Accuracy(num_classes=n_classes, task="multiclass", average=None)
).to(device)
clip_accuracy_metric_per_cls = torchmetrics.ClasswiseWrapper(
    torchmetrics.Accuracy(num_classes=n_classes, task="multiclass", average=None)
).to(device)

class_counters = {
    # "class_name": list(dataset.datasets[0].class_to_idx.keys()),
    "class_name": list(dataset.class_to_idx.keys()),
    "both_correct": torch.zeros(n_classes),
    "both_wrong": torch.zeros(n_classes),
    "only_clip_correct": torch.zeros(n_classes),
    "only_descr_correct": torch.zeros(n_classes),
    "intersection": torch.zeros(n_classes),
    "intersection_correct": torch.zeros(n_classes),
}

for batch_number, batch in tqdm(enumerate(dataloader)):
    images, labels = batch
    """
    images: (batch_size, num_channels, height, width)
    labels: (batch_size)
    """
    images = images.to(device)
    labels = labels.to(device)

    image_encodings = model.encode_image(images)
    image_encodings = F.normalize(image_encodings)

    """
    image_encodings: batch_size, 1024
    labels: (batch_size) [Each cell is a number between 0 to num_classes-1]
    """

    # Clip Basic Method
    image_labels_similarity = image_encodings @ label_encodings.T
    clip_predictions = image_labels_similarity.argmax(dim=1)

    """
    image_labels_similarity: batch_size, num_classes
    """

    clip_acc = clip_accuracy_metric(image_labels_similarity, labels)
    clip_acc_top5 = clip_accuracy_metric_top5(image_labels_similarity, labels)
    clip_acc_per_cls = clip_accuracy_metric_per_cls(image_labels_similarity, labels)

    #  This Paper Method
    image_description_similarity = [None] * n_classes
    image_description_similarity_cumulative = [None] * n_classes

    # You can also vectorize this; it wasn't much faster for me
    for class_i, (class_name, class_descriptions_encodings) in enumerate(description_encodings.items()):
        # image_encodings: (batch_size, 1024) | class_descriptions_encodings: (num_descriptions_for_cls, 1024)
        dot_product_matrix = image_encodings @ class_descriptions_encodings.T
        # dot_product_matrix: (batch_size, num_descriptions_for_cls)

        image_description_similarity[class_i] = dot_product_matrix
        image_description_similarity_cumulative[class_i] = aggregate_similarity(
            image_description_similarity[class_i], aggregation_method='mean'
        )

    # create tensor of similarity means
    cumulative_tensor = torch.stack(image_description_similarity_cumulative, dim=1)

    descr_predictions = cumulative_tensor.argmax(dim=1)

    """
    cumulative_tensor: batch_size, num_classes
    descr_predictions: batch_size
    """
    # lang_acc = lang_accuracy_metric(cumulative_tensor.softmax(dim=-1), labels)
    # lang_acc_top5 = lang_accuracy_metric_top5(cumulative_tensor.softmax(dim=-1), labels)
    # lang_acc_per_cls = lang_accuracy_metric_per_cls(cumulative_tensor.softmax(dim=-1), labels)

    lang_acc = lang_accuracy_metric(cumulative_tensor, labels)
    lang_acc_top5 = lang_accuracy_metric_top5(cumulative_tensor, labels)
    lang_acc_per_cls = lang_accuracy_metric_per_cls(cumulative_tensor, labels)

    for class_idx in range(n_classes):
        class_mask = (labels == class_idx)

        class_counters["both_correct"][class_idx] += (clip_predictions == labels).logical_and(
            descr_predictions == labels
        ).logical_and(class_mask).sum().item()

        class_counters["both_wrong"][class_idx] += (clip_predictions != labels).logical_and(
            descr_predictions != labels).logical_and(class_mask).sum().item()

        class_counters["only_clip_correct"][class_idx] += (clip_predictions == labels).logical_and(
            descr_predictions != labels).logical_and(class_mask).sum().item()

        class_counters["only_descr_correct"][class_idx] += (clip_predictions != labels).logical_and(
            descr_predictions == labels).logical_and(class_mask).sum().item()

        class_counters["intersection"][class_idx] += (clip_predictions == descr_predictions).logical_and(
            class_mask).sum().item()

        class_counters["intersection_correct"][class_idx] += (clip_predictions == descr_predictions).logical_and(
            clip_predictions == labels).logical_and(class_mask).sum().item()

# Calculate final metrics

print("\n")

accuracy_logs = {}
accuracy_logs["Total Description-based Top-1 Accuracy: "] = 100 * lang_accuracy_metric.compute().item()
accuracy_logs["Total Description-based Top-5 Accuracy: "] = 100 * lang_accuracy_metric_top5.compute().item()

accuracy_logs["Total CLIP-Standard Top-1 Accuracy: "] = 100 * clip_accuracy_metric.compute().item()
accuracy_logs["Total CLIP-Standard Top-5 Accuracy: "] = 100 * clip_accuracy_metric_top5.compute().item()

clip_res_per_cls = clip_accuracy_metric_per_cls.compute()
lang_res_per_cls = lang_accuracy_metric_per_cls.compute()

# print the dictionary
print("\n")
for key, value in accuracy_logs.items():
    print(key, value)

print(f'\n')
print(clip_res_per_cls)
assert clip_res_per_cls.keys() == lang_res_per_cls.keys()
res_per_cls = {
    # 'class': list(dataset.datasets[0].class_to_idx.keys()),
    'class': list(dataset.class_to_idx.keys()),
    'clip acc': [round(clip_res_per_cls[cls_idx].item(), 3) for cls_idx in clip_res_per_cls.keys()],
    'descr acc': [round(lang_res_per_cls[cls_idx].item(), 3) for cls_idx in lang_res_per_cls.keys()],
}

pd.DataFrame(res_per_cls).to_csv('res_per_cls.csv', index=False)
pd.DataFrame(class_counters).to_csv('class_counters.csv', index=False)
