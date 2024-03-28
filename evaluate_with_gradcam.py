import torch
from torch.nn import functional as F

from tqdm import tqdm

from load import aggregate_similarity
from scratch_grad_cam.gradcam import get_cams
from utils import pretty_print, get_metrics


def evaluate(
        model,
        dataloader,
        label_encodings,
        description_encodings,
        device
):
    n_classes = label_encodings.size(0)

    clip_metrics = [metric.to(device) for metric in get_metrics(n_classes)]
    dclip_metrics = [metric.to(device) for metric in get_metrics(n_classes)]

    cam_sims_per_cls = [[] for _ in range(n_classes)]

    for batch_number, batch in tqdm(enumerate(dataloader)):
        images, labels = batch
        images = images.to(device)  # (batch_size, num_channels, height, width)
        labels = labels.to(device)  # (batch_size) [Each cell is a number between 0 to num_classes-1]

        image_encodings = model.encode_image(images)  # batch_size, 1024
        image_encodings = F.normalize(image_encodings)

        # Clip Basic Method
        image_labels_similarity = image_encodings @ label_encodings.T  # batch_size, num_classes
        # clip_predictions = image_labels_similarity.argmax(dim=1)  # batch_size

        for metric in clip_metrics:
            metric(image_labels_similarity, labels)

        #  DClip Method
        image_description_similarity_cumulative = [None] * n_classes

        # You can also vectorize this; it wasn't much faster for me
        for class_i, (class_name, class_descriptions_encodings) in enumerate(description_encodings.items()):
            # image_encodings: (batch_size, 1024) | class_descriptions_encodings: (num_descriptions_for_cls, 1024)
            descr_cams_tensor, dot_product_matrix = get_cams(class_descriptions_encodings, images, model)

            descr_cams_tensor = descr_cams_tensor.view(
                descr_cams_tensor.size(0), descr_cams_tensor.size(1), -1
            )  # Flatten cams tensor (bs, num_descr_for_cls, h, w) -> (bs, num_descr_for_cls, h * w)
            cos_sim_matrix = F.cosine_similarity(
                descr_cams_tensor[:, :, None, :], descr_cams_tensor[:, None, :, :], dim=-1
            )
            cos_sim_matrix = cos_sim_matrix - torch.eye(cos_sim_matrix.size(1)).to(cos_sim_matrix.device)
            average_cos_sim = cos_sim_matrix.mean(dim=[1, 2])
            cam_sims_per_cls[class_i].extend(average_cos_sim.tolist())

            image_description_similarity_cumulative[class_i] = aggregate_similarity(
                dot_product_matrix, aggregation_method='mean'
            )
        # create tensor of similarity means
        cumulative_tensor = torch.stack(image_description_similarity_cumulative, dim=1)  # batch_size, num_classes
        # descr_predictions = cumulative_tensor.argmax(dim=1)  # batch_size

        for metric in dclip_metrics:
            metric(cumulative_tensor, labels)
            metric(cumulative_tensor.softmax(dim=-1), labels)

    # Calculate final metrics
    clip_acc_metric, clip_acc_metric_top5, clip_acc_metric_per_cls = clip_metrics
    dclip_acc_metric, dclip_acc_metric_top5, dclip_acc_metric_per_cls = dclip_metrics

    accuracy_logs = {
        "Total Description-based Top-1 Accuracy: ": 100 * dclip_acc_metric.compute().item(),
        "Total Description-based Top-5 Accuracy: ": 100 * dclip_acc_metric_top5.compute().item(),
        "Total CLIP-Standard Top-1 Accuracy: ": 100 * clip_acc_metric.compute().item(),
        "Total CLIP-Standard Top-5 Accuracy: ": 100 * clip_acc_metric_top5.compute().item()
    }
    print("\n")
    pretty_print(accuracy_logs)

    clip_acc_per_cls = clip_acc_metric_per_cls.compute()
    # print(f'CLIP-Standard Accuracy Per Class')
    # print(clip_acc_per_cls)

    dclip_acc_per_cls = dclip_acc_metric_per_cls.compute()
    # print(f'Description-based Accuracy Per Class')
    # print(dclip_acc_per_cls)

    # TODO
    #  s

    return clip_acc_per_cls, dclip_acc_per_cls, cam_sims_per_cls
