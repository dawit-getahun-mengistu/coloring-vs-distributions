import torchvision.utils as vutils
import torch
from torch.utils.data import DataLoader
from torchvision import utils

import matplotlib.pyplot as plt


def show_images_from_loader(loader: DataLoader, classes: list[str], n_images: int = 64):

    # Get a batch of images and labels
    images, labels = next(iter(loader))

    # Select n_images from the batch
    images, labels = images[:n_images], labels[:n_images]

    # Create a grid of images
    grid = utils.make_grid(images, nrow=int(
        n_images**0.5), padding=2, normalize=True)

    plt.figure(figsize=(8, 8))
    # Convert from (C, H, W) to (H, W, C) for plotting
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title("Sample Images from Dataset")

    for i, label in enumerate(labels):
        row, col = divmod(i, int(n_images**0.5))
        plt.text(col * (grid.size(2) / n_images**0.5) + 5,
                 row * (grid.size(1) / n_images**0.5) + 5,
                 classes[label.item()],
                 color='white', fontsize=9, ha='center', bbox=dict(facecolor='black', alpha=0.6))
    plt.show()


def show_images_from_loader_of_class(
    loader,
    classes: list[str],
    target_class_idx: int,
    n_images: int = 64
):
    """
    Show 'n_images' examples of a specific class (identified by target_class_idx)
    from a DataLoader. Iterates over all batches until enough images are found.
    """

    collected_images = []
    collected_labels = []

    # Iterate through the DataLoader
    for images, labels in loader:
        for img, lbl in zip(images, labels):
            if lbl.item() == target_class_idx:
                collected_images.append(img)
                collected_labels.append(lbl)
            if len(collected_images) >= n_images:
                break  # Stop if we have enough
        if len(collected_images) >= n_images:
            break

    # If we couldn't find enough images, show as many as we have
    if len(collected_images) == 0:
        print(f"No images found for class index {target_class_idx}.")
        return

    # Slice the collected images to exactly n_images if we have more than needed
    collected_images = collected_images[:n_images]
    collected_labels = collected_labels[:n_images]

    # Stack the images into a single tensor
    images_tensor = torch.stack(collected_images)

    # Make a grid of images
    grid = vutils.make_grid(
        images_tensor,
        nrow=int(n_images**0.5),  # or set nrow=8 (etc.) directly
        padding=2,
        normalize=True
    )

    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title(f"Sample Images from Class: {classes[target_class_idx]}")

    # Optionally, you can annotate each image.
    # However, since they're all the same class, we might not need to.
    # If you do want text, you'd do something similar to your original function:
    for i, label in enumerate(collected_labels):
        row, col = divmod(i, int(n_images**0.5))
        plt.text(
            col * (grid.size(2) / (n_images**0.5)) + 5,
            row * (grid.size(1) / (n_images**0.5)) + 5,
            classes[label.item()],
            color='white',
            fontsize=9,
            ha='center',
            bbox=dict(facecolor='black', alpha=0.6)
        )

    plt.show()
