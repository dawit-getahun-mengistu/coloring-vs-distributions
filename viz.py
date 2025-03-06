import torch
import numpy as np
import matplotlib.pyplot as plt
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict


def remove_all_hooks(model):
    """Remove all hooks from model"""
    for module in model.modules():
        if hasattr(module, '_forward_hooks'):
            module._forward_hooks.clear()
        if hasattr(module, '_backward_hooks'):
            module._backward_hooks.clear()
        if hasattr(module, '_forward_pre_hooks'):
            module._forward_pre_hooks.clear()
        if hasattr(module, '_backward_pre_hooks'):
            module._backward_pre_hooks.clear()

    # Also check for hook_handles attribute that torchcam might have added
    if hasattr(model, 'hook_handles'):
        for handle in model.hook_handles:
            handle.remove()
        model.hook_handles = []


class ActivationHook:
    """Class to capture layer activations"""

    def __init__(self, layer):
        self.activation = None
        self.hook = layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.activation = output.detach()

    def remove(self):
        self.hook.remove()


def get_model_layers(model):
    """Automatically find convolution and linear layers in model"""
    layers = OrderedDict()
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            layers[name] = module
    return layers


def collect_samples(model, test_loader, device, num_classes=10):
    if hasattr(model, '_hooks_enabled'):
        model._hooks_enabled = False

    samples = {i: {'correct': [], 'misclassified': []}
               for i in range(num_classes)}

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Collecting samples"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            for img, label, pred in zip(images, labels, preds):
                class_samples = samples[label.item()]
                status = 'correct' if pred == label else 'misclassified'

                if len(class_samples[status]) < 2:
                    class_samples[status].append({
                        'tensor': img.cpu(),
                        'true_label': label.item(),
                        'pred_label': pred.item()
                    })

                # Early exit if all classes have enough samples
                complete = all(
                    len(samples[c][s]) >= 2
                    for c in samples
                    for s in ['correct', 'misclassified']
                )
                if complete:
                    return samples
    return samples


def visualize_results(samples, model, device):
    model.eval()
    for param in model.parameters():
        param.requires_grad = True

    # Get convolutional layers
    def get_conv_layers(model):
        layers = OrderedDict()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                layers[name] = module
        return layers

    model_layers = get_conv_layers(model)
    if not model_layers:
        raise ValueError("No convolutional layers found in the model!")

    # Use last conv layer for GradCAM
    target_layer_name = list(model_layers.keys())[-1]
    target_layer = model_layers[target_layer_name]

    # Use first two conv layers for activation visualization
    activation_layers = list(model_layers.items())[:2]

    # Set up hooks
    hooks = [ActivationHook(layer) for _, layer in activation_layers]

    # Create CAM extractor
    cam_extractor = GradCAM(model, target_layer=target_layer)

    # Process each class
    for class_idx in range(10):
        class_samples = samples[class_idx]
        for status in ['correct', 'misclassified']:
            for sample in class_samples[status][:2]:
                cam_extractor = GradCAM(model, target_layer=target_layer)

                img_tensor = sample['tensor'].unsqueeze(0).to(device)
                true_label = sample['true_label']
                pred_label = sample['pred_label']

                # Enable gradients only for GradCAM
                with torch.set_grad_enabled(True):  # Enable gradients

                    img_tensor.requires_grad = True

                    # model.zero_grad()  # Clear any existing gradients
                    output = model(img_tensor)  # Forward pass
                    # cam = cam_extractor(pred_label, output)  # Compute CAM
                    # model.zero_grad()  # Clear gradients again
                    try:
                        cam = cam_extractor(
                            class_idx=pred_label, scores=output)
                    except Exception as e:
                        print(f"Error computing CAM: {e}")
                        continue

                # Disable gradients for the rest of the operations
                with torch.no_grad():
                    # Process activations
                    activation_maps = []
                    for hook in hooks:
                        act = hook.activation.squeeze().cpu().numpy()
                        act = act.mean(axis=0)  # Average across channels
                        act = (act - act.min()) / \
                            (act.max() - act.min() + 1e-8)
                        activation_maps.append(Image.fromarray(
                            (act * 255).astype(np.uint8)))

                    img_np = img_tensor.squeeze(
                        0).cpu().permute(1, 2, 0).numpy()
                    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))

                    # Process CAM
                    # cam_img = overlay_mask(img_pil, cam[0].squeeze().cpu().numpy(), alpha=0.5)
                    # Process CAM
                    cam_numpy = cam[0].squeeze().cpu().numpy()
                    # Normalize to 0-1 range
                    cam_numpy = (cam_numpy - cam_numpy.min()) / \
                        (cam_numpy.max() - cam_numpy.min() + 1e-8)
                    # Convert to 0-255 range and to a PIL Image
                    cam_pil = Image.fromarray(
                        (cam_numpy * 255).astype(np.uint8))
                    # Now apply the overlay
                    cam_img = overlay_mask(img_pil, cam_pil, alpha=0.5)

                    # Plotting
                    fig, axs = plt.subplots(
                        1, 1 + len(activation_layers) + 1, figsize=(20, 5))
                    titles = [
                        f"Original ({'Correct' if status == 'correct' else 'Wrong'})"]
                    titles += [f"Layer {i+1}" for i in range(
                        len(activation_layers))]
                    titles += [f"CAM (Pred: {pred_label})"]

                    images = [img_pil] + activation_maps + [cam_img]

                    for ax, img, title in zip(axs, images, titles):
                        ax.imshow(img if isinstance(img, Image.Image) else img,
                                  cmap='jet' if 'Layer' in title else None)
                        ax.set_title(title)
                        ax.axis('off')

                    plt.tight_layout()
                    plt.show()

                cam_extractor.remove_hooks()

    # Cleanup hooks
    for hook in hooks:
        hook.remove()
