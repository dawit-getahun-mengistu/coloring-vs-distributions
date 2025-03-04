import random
import torch
from torch.utils.data import Dataset

from colors import generate_colors
import colorsys


class ForEachInstanceRandomlyColorsDatasetGenerator(Dataset):
    def __init__(self, dataset, transform=None, saturation: float = 1.0, brightness: float = 1.0):
        self.dataset = dataset
        self.transform = transform
        self.saturation = saturation
        self.brightness = brightness

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)

        # For reproducibility, seed random with the index so each instance always gets the same color
        random.seed(idx)
        hue = random.random()  # random hue in [0,1)
        r, g, b = colorsys.hsv_to_rgb(hue, self.saturation, self.brightness)
        color_tensor = torch.tensor([r, g, b], dtype=torch.float32)

        colored_image = image * color_tensor.view(3, 1, 1)
        return colored_image, label


class SingleColorsForAllButOneDatasetGenerator(Dataset):
    def __init__(self, dataset, outlier_cls_idx: int, num_outlier_colors: int, transform=None, num_classes: int = 10):

        self.dataset = dataset
        self.outlier_class = outlier_cls_idx
        self.transform = transform
        self.num_classes = num_classes
        self.num_outlier_colors = num_outlier_colors

        self.colors = {i: v for i, v in enumerate(
            generate_colors(self.num_classes))}

        self.outlier_colors = generate_colors(self.num_outlier_colors)
        self.outlier_coloring_idx = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        if self.transform:
            image = self.transform(image)

        if label == self.outlier_class:
            colored_image, label = self.color_outlier(image, label, idx)
            return colored_image, label

        colored_image, label = self.colorize(image, label)
        return colored_image, label

    def colorize(self, image, label):
        if label != self.outlier_class:
            _color = torch.tensor(self.colors[label], dtype=torch.float32)
            colored_image = image * _color.view(3, 1, 1)
            return colored_image, label

        raise Exception("Invalid Class Index")

    def color_outlier(self, image, label, idx):
        if label != self.outlier_class:
            raise Exception("Invalid Class Index")

        # Use modulo to ensure the index wraps around the available colors
        color_index = self.outlier_coloring_idx % len(self.outlier_colors)

        _color = torch.tensor(
            self.outlier_colors[color_index], dtype=torch.float32)
        colored_image = image * _color.view(3, 1, 1)

        self.outlier_coloring_idx += 1

        return colored_image, label
