from typing import OrderedDict as OrderedDictType
from collections import OrderedDict
import pathlib

from torch.nn import functional as F
from torchvision.datasets import ImageNet, ImageFolder, Places365
from imagenetv2_pytorch import ImageNetV2Dataset as ImageNetV2
from PIL import Image
import clip

from datasets import _transform, CUBDataset
from descriptor_strings import *  # label_to_classname, wordify, modify_descriptor,
from loading_helpers import *


# hparams = {}
# gpt_descriptions, unmodify_dict, label_to_classname = None, None, None


# hparams['label_after_text'] = ' which is a type of bird.'
# hparams['label_after_text'] = ' which is a pet animal.'
# hparams['label_after_text'] = ' which is a type of animal, a cub, a young carnivorous mammal.'
# hparams['label_after_text'] = ' which is a type of bird, Aves.'
# hparams['label_after_text'] = ' which is a type of Aves, a taxonomic class of birds.'
# hparams['label_after_text'] = ' which is a texture of a surface.'


def load_hparams(custom_hparams: dict = None):
    hparams = {}
    if custom_hparams is None:
        custom_hparams = {}

    hparams.update({
        # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        'model_size': 'RN50',
        # ['imagenet', 'imagenetv2', 'cub', 'eurosat', 'places365', 'food101', 'pets', 'dtd']
        'dataset': 'imagenet',
        'batch_size': 128,
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'category_name_inclusion': 'prepend',
        'apply_descriptor_modification': True,
        'verbose': False,
        'image_size': 224,
        'before_text': '',
        # DON'T FORGET THE TRAILING SPACE
        'label_before_text': '',  # ["this is an image of a ", ]
        'between_text': ', ',  # ['', ' ', ', ']
        'after_text': '',  # ['.', ' which is a type of bird.', ],
        # DON'T FORGET THE LEADING SPACE
        'label_after_text': '',
        'unmodify': True,
        'seed': 1,
    })

    # custom hparams
    hparams.update(custom_hparams)

    # image size validation
    model_to_image_size_mapping = {
        'ViT-L/14@336px': 336,
        'RN50x4': 288,
        'RN50x16': 384,
        'RN50x64': 448
    }
    correct_image_size = model_to_image_size_mapping.get(hparams['model_size'], 224)
    if hparams['image_size'] != correct_image_size:
        print(f'Model size is {hparams["model_size"]}'
              f' but image size is {hparams["image_size"]}. '
              f'Setting image size to {correct_image_size}.')
        hparams['image_size'] = correct_image_size

    # data dir
    dataset_to_dir_mapping = {
        'imagenet': './data/ImageNet/',
        'imagenetv2': './data/ImageNetV2/',
        'cub': './data/cub',
        'eurosat': './data/eurosat',
        'food101': './data/food101',
        'pets': './data/pets',
        'dtd': './data/dtd',
        'places365': './data/places365'
    }
    hparams['data_dir'] = pathlib.Path(dataset_to_dir_mapping[hparams['dataset']])

    # descriptors
    hparams['descriptor_fname'] = f"./descriptors/descriptors_{hparams['dataset']}"

    if hparams['dataset'] == 'imagenet':
        hparams['after_text'] = hparams['label_after_text'] = '.'

    return hparams


def load_dataset(hparams):
    tfms = _transform(hparams['image_size'])

    if hparams['dataset'] == 'imagenet':
        dataset = ImageNet(hparams['data_dir'], split='val', transform=tfms)
        classes_to_load = None
    elif hparams['dataset'] == 'imagenetv2':
        dataset = ImageNetV2(location=hparams['data_dir'], transform=tfms)
        classes_to_load = openai_imagenet_classes
        dataset.classes = classes_to_load
    elif hparams['dataset'] == 'cub':
        dataset = CUBDataset(hparams['data_dir'], train=False, transform=tfms)
        classes_to_load = None  # dataset.classes
    # I recommend using VISSL https://github.com/facebookresearch/vissl/blob/main/extra_scripts/README.md to download
    elif hparams['dataset'] == 'eurosat':
        from extra_datasets.patching.eurosat import EuroSATVal
        dataset = EuroSATVal(location=hparams['data_dir'], preprocess=tfms)
        dataset = dataset.test_dataset
        hparams['descriptor_fname'] = 'descriptors_eurosat'
        classes_to_load = None
    elif hparams['dataset'] == 'places365':
        dataset = ImageFolder(hparams['data_dir'] / 'val', transform=tfms)
        classes_to_load = None
    elif hparams['dataset'] == 'food101':
        dataset = ImageFolder(hparams['data_dir'] / 'test', transform=tfms)
        classes_to_load = None
    elif hparams['dataset'] == 'pets':
        dataset = ImageFolder(hparams['data_dir'] / 'test', transform=tfms)
        # train_dataset = ImageFolder(hparams['data_dir'] / 'train', transform=tfms)
        # test_dataset = ImageFolder(hparams['data_dir'] / 'test', transform=tfms)
        # dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        classes_to_load = None
    elif hparams['dataset'] == 'dtd':
        dataset = ImageFolder(hparams['data_dir'] / 'val', transform=tfms)
        classes_to_load = None
    else:
        raise ValueError('Unknown dataset: {}'.format(hparams['dataset']))

    return dataset, classes_to_load


def load_gpt_descriptions_wrapper(hparams, classes_to_load):
    print("Creating descriptors...")
    gpt_descriptions, unmodify_dict = load_gpt_descriptions(hparams, classes_to_load)
    label_to_classname = list(gpt_descriptions.keys())
    hparams['n_classes'] = len(list(gpt_descriptions.keys()))
    return gpt_descriptions, unmodify_dict, label_to_classname


def compute_description_encodings(hparams, model, gpt_descriptions) -> OrderedDictType[str, torch.Tensor]:
    """
    Computes and returns an ordered dictionary mapping label names to their corresponding tensor encodings.

    Each tensor encoding represents the descriptions for a specific label,
     with dimensions (num_descriptions_for_label, 1024).

    Args:
        model: The model used to compute the description encodings.

    Returns:
        An OrderedDict where each key is a label name, and the corresponding value is a torch.Tensor containing the encodings for that label.
    """
    description_encodings = OrderedDict()
    for k, v in gpt_descriptions.items():
        tokens = clip.tokenize(v).to(hparams['device'])
        description_encodings[k] = F.normalize(model.encode_text(tokens))
    return description_encodings


def compute_label_encodings(hparams, model, labels):
    """
        Computes and returns a tensor of label encodings for a given model.

        This function processes the labels by concatenating a prefix and suffix to them,
         tokenizing and encoding them using the provided model.
         The labels are then normalized to ensure they are in a consistent format.

        Parameters:
        - model: The model used to encode the labels.
        - labels: A list of labels

        Returns:
        - label_encodings: A torch.Tensor of shape (num_classes, 1024) containing the normalized encodings for each label.
        """
    label_encodings = F.normalize(
        model.encode_text(
            clip.tokenize(labels).to(hparams['device'])
        )
    )
    return label_encodings


def get_formatted_labels(hparams, label_to_classname):
    return [
        hparams['label_before_text'] + wordify(l) + hparams['label_after_text'] for l in label_to_classname
    ]


def aggregate_similarity(similarity_matrix_chunk, aggregation_method='mean'):
    if aggregation_method == 'max':
        return similarity_matrix_chunk.max(dim=1)[0]
    elif aggregation_method == 'sum':
        return similarity_matrix_chunk.sum(dim=1)
    elif aggregation_method == 'mean':
        return similarity_matrix_chunk.mean(dim=1)
    else:
        raise ValueError("Unknown aggregate_similarity")


def show_from_indices(
        indices,
        images,
        labels=None,
        predictions=None,
        predictions2=None,
        n=None,
        image_description_similarity=None,
        image_labels_similarity=None
):
    if indices is None or (len(indices) == 0):
        print("No indices provided")
        return

    if n is not None:
        indices = indices[:n]

    for index in indices:
        show_single_image(images[index])
        print(f"Index: {index}")
        if labels is not None:
            true_label = labels[index]
            true_label_name = label_to_classname[true_label]
            print(f"True label: {true_label_name}")
        if predictions is not None:
            predicted_label = predictions[index]
            predicted_label_name = label_to_classname[predicted_label]
            print(f"Predicted label (ours): {predicted_label_name}")
        if predictions2 is not None:
            predicted_label2 = predictions2[index]
            predicted_label_name2 = label_to_classname[predicted_label2]
            print(f"Predicted label 2 (CLIP): {predicted_label_name2}")

        print("\n")

        if image_labels_similarity is not None:
            if labels is not None:
                print(
                    f"Total similarity to {true_label_name} (true label) labels: {image_labels_similarity[index][true_label].item()}")
            if predictions is not None:
                if labels is not None and true_label_name == predicted_label_name:
                    print("Predicted label (ours) matches true label")
                else:
                    print(
                        f"Total similarity to {predicted_label_name} (predicted label) labels: {image_labels_similarity[index][predicted_label].item()}")
            if predictions2 is not None:
                if labels is not None and true_label_name == predicted_label_name2:
                    print("Predicted label 2 (CLIP) matches true label")
                elif predictions is not None and predicted_label_name == predicted_label_name2:
                    print("Predicted label 2 (CLIP) matches predicted label 1")
                else:
                    print(
                        f"Total similarity to {predicted_label_name2} (predicted label 2) labels: {image_labels_similarity[index][predicted_label2].item()}")

            print("\n")

        if image_description_similarity is not None:
            if labels is not None:
                print_descriptor_similarity(image_description_similarity, index, true_label, true_label_name, "true")
                print("\n")
            if predictions is not None:
                if labels is not None and true_label_name == predicted_label_name:
                    print("Predicted label (ours) same as true label")
                    # continue
                else:
                    print_descriptor_similarity(image_description_similarity, index, predicted_label,
                                                predicted_label_name, "descriptor")
                print("\n")
            if predictions2 is not None:
                if labels is not None and true_label_name == predicted_label_name2:
                    print("Predicted label 2 (CLIP) same as true label")
                    # continue
                elif predictions is not None and predicted_label_name == predicted_label_name2:
                    print("Predicted label 2 (CLIP) matches predicted label 1")
                else:
                    print_descriptor_similarity(image_description_similarity, index, predicted_label2,
                                                predicted_label_name2, "CLIP")
            print("\n")


def print_descriptor_similarity(
        image_description_similarity,
        index,
        label,
        label_name,
        label_type="provided"
):
    # print(f"Total similarity to {label_name} ({label_type} label) descriptors: {aggregate_similarity(image_description_similarity[label][index].unsqueeze(0)).item()}")
    print(f"Total similarity to {label_name} ({label_type} label) descriptors:")
    print(f"Average:\t\t{100. * aggregate_similarity(image_description_similarity[label][index].unsqueeze(0)).item()}")
    label_descriptors = gpt_descriptions[label_name]
    for k, v in sorted(zip(label_descriptors, image_description_similarity[label][index]), key=lambda x: x[1],
                       reverse=True):
        k = unmodify_dict[label_name][k]
        # print("\t" + f"matched \"{k}\" with score: {v}")
        print(f"{k}\t{100. * v}")


def print_max_descriptor_similarity(
        image_description_similarity,
        index,
        label,
        label_name
):
    max_similarity, argmax = image_description_similarity[label][index].max(dim=0)
    label_descriptors = gpt_descriptions[label_name]
    print(
        f"I saw a {label_name} because I saw {unmodify_dict[label_name][label_descriptors[argmax.item()]]} with score: {max_similarity.item()}")


def show_misclassified_images(
        images,
        labels,
        predictions,
        n=None,
        image_description_similarity=None,
        image_labels_similarity=None,
        true_label_to_consider: int = None,
        predicted_label_to_consider: int = None
):
    misclassified_indices = yield_misclassified_indices(
        images,
        labels=labels,
        predictions=predictions,
        true_label_to_consider=true_label_to_consider,
        predicted_label_to_consider=predicted_label_to_consider
    )
    if misclassified_indices is None:
        return

    show_from_indices(
        misclassified_indices,
        images, labels, predictions,
        n=n,
        image_description_similarity=image_description_similarity,
        image_labels_similarity=image_labels_similarity
    )


def yield_misclassified_indices(
        images,
        labels,
        predictions,
        true_label_to_consider=None,
        predicted_label_to_consider=None
):
    misclassified_indicators = (predictions.cpu() != labels.cpu())

    if true_label_to_consider is not None:
        misclassified_indicators = misclassified_indicators & (labels.cpu() == true_label_to_consider)

    if predicted_label_to_consider is not None:
        misclassified_indicators = misclassified_indicators & (predictions.cpu() == predicted_label_to_consider)

    if misclassified_indicators.sum() == 0:
        output_string = 'No misclassified images found'
        if true_label_to_consider is not None:
            output_string += f' with true label {label_to_classname[true_label_to_consider]}'
        if predicted_label_to_consider is not None:
            output_string += f' with predicted label {label_to_classname[predicted_label_to_consider]}'
        print(output_string + '.')

        return

    misclassified_indices = torch.arange(images.shape[0])[misclassified_indicators]
    return misclassified_indices


def predict_and_show_explanations(
        hparams,
        images,
        model,
        gpt_descriptions,
        labels=None,
        description_encodings=None,
        label_encodings=None,
        device=None,
):
    tfms = _transform(hparams['image_size'])
    if type(images) == Image:
        images = tfms(images)

    if images.device != device:
        images = images.to(device)
        labels = labels.to(device)

    image_encodings = model.encode_image(images)
    image_encodings = F.normalize(image_encodings)

    image_labels_similarity = image_encodings @ label_encodings.T
    clip_predictions = image_labels_similarity.argmax(dim=1)

    n_classes = len(description_encodings)
    image_description_similarity = [None] * n_classes
    image_description_similarity_cumulative = [None] * n_classes
    for i, (k, v) in enumerate(
            description_encodings.items()):  # You can also vectorize this; it wasn't much faster for me

        dot_product_matrix = image_encodings @ v.T

        image_description_similarity[i] = dot_product_matrix
        image_description_similarity_cumulative[i] = aggregate_similarity(image_description_similarity[i])

    # create tensor of similarity means
    cumulative_tensor = torch.stack(image_description_similarity_cumulative, dim=1)

    descr_predictions = cumulative_tensor.argmax(dim=1)

    show_from_indices(torch.arange(images.shape[0]), images, labels, descr_predictions, clip_predictions,
                      image_description_similarity=image_description_similarity,
                      image_labels_similarity=image_labels_similarity)
