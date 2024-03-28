import torch
from torch.utils.data import DataLoader
import clip

from load import (
    seed_everything,
    load_hparams,
    load_dataset,
    load_gpt_descriptions_wrapper,
    compute_description_encodings,
    compute_label_encodings,
    get_formatted_labels,
)
from evaluate import evaluate
from evaluate_with_gradcam import evaluate as evaluate_with_gradcam
from evaluate_with_scorecam import evaluate as evaluate_with_scorecam


def main(custom_hparams=None):
    hparams = load_hparams(custom_hparams)
    print(hparams)

    seed_everything(hparams['seed'])

    dataset, classes_to_load = load_dataset(hparams)
    gpt_descriptions, unmodify_dict, label_to_classname = load_gpt_descriptions_wrapper(hparams, classes_to_load)

    # dataloader = DataLoader(dataset, bs, shuffle=True, num_workers=16, pin_memory=True)
    dataloader = DataLoader(dataset, hparams['batch_size'], shuffle=True, num_workers=8, pin_memory=False)
    print(list(dataset.class_to_idx.items())[:10])

    print("Loading model...")
    device = torch.device(hparams['device'])
    model, preprocess = clip.load(hparams['model_size'], device=device, jit=False)
    model.eval()
    model.requires_grad_(False)

    print("Encoding descriptions...")
    description_encodings = compute_description_encodings(hparams, model, gpt_descriptions)
    print("Encoding labels...")
    class_labels = get_formatted_labels(hparams, label_to_classname)
    label_encodings = compute_label_encodings(hparams, model, class_labels)

    print("Evaluating...")
    if hparams.get('eval') is None:
        return evaluate(
            model, dataloader, label_encodings, description_encodings, device
        )
    elif hparams['eval'] == 'with_gradcam':
        return evaluate_with_gradcam(
            model, dataloader, label_encodings, description_encodings, device
        )
    elif hparams['eval'] == 'with_scorecam':
        return evaluate_with_scorecam(
            model, dataloader, label_encodings, description_encodings, device, hparams, class_labels
        )


if __name__ == '__main__':
    main()
