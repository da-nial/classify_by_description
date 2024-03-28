import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple


class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)

    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()

    @property
    def activation(self) -> torch.Tensor:
        return self.data

    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad


# Reference: https://arxiv.org/abs/1610.02391
def gradCAM(
        model: nn.Module,
        input: torch.Tensor,
        target: torch.Tensor,
        layer: nn.Module
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()

    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)

    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)
    with Hook(layer) as hook:
        # Do a forward and backward pass.
        output = model(input)
        output.backward(target)

        grad = hook.gradient.float()
        act = hook.activation.float()

        # Global average pool gradient across spatial dimension to obtain importance weights.
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel dimension.
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we clamp any negative ones.
        gradcam = torch.clamp(gradcam, min=0)

    # Resize gradcam to input resolution.
    gradcam = F.interpolate(
        gradcam,
        input.shape[2:],
        mode='bicubic',
        align_corners=False
    )

    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])

    return gradcam, output


def get_cams(text_encodings, image_tensors, model, saliency_layer='layer4'):
    cams_list = []
    sims_list = []

    for image_tensor in image_tensors:
        cams_for_image = []
        sims_for_image = []

        for text_encoding in text_encodings:
            cam, sim = get_cam(text_encoding, image_tensor, model, saliency_layer)
            cams_for_image.append(cam.squeeze())
            sims_for_image.append(sim.squeeze())

        cams_list.append(torch.stack(cams_for_image))
        sims_list.append(torch.stack(sims_for_image))

    cams_tensor = torch.stack(cams_list)
    sims_tensor = torch.stack(sims_list)

    return cams_tensor, sims_tensor


def get_cam(text_encoding, image_tensor, model, saliency_layer='layer4'):
    # image_input = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    # # image_np = load_image(image_path, model.visual.input_resolution)

    # text_input = clip.tokenize([caption]).to(device)
    # text_encoding = model.encode_text(text_input).float()

    attn_map, image_encoding = gradCAM(
        model.visual,
        image_tensor.unsqueeze(0),
        text_encoding.unsqueeze(0),
        getattr(model.visual, saliency_layer)
    )
    # attn_map = attn_map.squeeze().detach().cpu().numpy()

    text_encoding = text_encoding.unsqueeze(0)
    # image_encoding = model.encode_image(image_input).float()
    sim = (image_encoding @ text_encoding.T) / (
            image_encoding.norm(dim=1, p=2) * text_encoding.norm(dim=1, p=2)
    )

    return attn_map, sim
