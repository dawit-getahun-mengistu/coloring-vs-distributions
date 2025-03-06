import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from cfg import NormalizationType


# initialize with xavier
def init_cnn(module):
    """Initialize weights for CNNs."""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)


class LeNetBasedModel(nn.Module):
    """LeNet based model"""

    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.network = nn.Sequential(
            # Input: (in_channels, 32, 32)
            nn.Conv2d(in_channels, 6, kernel_size=5),    # -> (6, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # -> (6, 14, 14)

            nn.Conv2d(6, 16, kernel_size=5),              # -> (16, 10, 10)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # -> (16, 5, 5)

            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

        # Apply weight initialization
        self.apply(init_cnn)

    def forward(self, x):
        return self.network(x)


class LeNetBasedModelWithNormalization(nn.Module):
    """LeNet based model with nomalization"""

    def __init__(self, in_channels=3, num_classes=10, normalization_type: NormalizationType = NormalizationType.batch):

        super().__init__()

        self.normalization_type = normalization_type

        self.conv1 = nn.Conv2d(
            in_channels, 6, kernel_size=5)
        self.norm1 = self._get_norm_layer(6)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.norm2 = self._get_norm_layer(16)

        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        # if self.normalization_type == NormalizationType.layer:

        #     self.layer_norm1 = nn.LayerNorm([6, 24, 24])
        #     self.layer_norm2 = nn.LayerNorm([16, 8, 8])

        # Apply weight initialization
        self.apply(init_cnn)

    def _get_norm_layer(self, num_features):
        if self.normalization_type == NormalizationType.batch:
            return nn.BatchNorm2d(num_features)
        elif self.normalization_type == NormalizationType.instance:
            return nn.InstanceNorm2d(num_features)
        elif self.normalization_type == NormalizationType.layer:
            return None  # applied dynamically
        else:
            raise ValueError(
                f"Unsupported Normalization type: {self.normalization_type}")

    def forward(self, x):
        # layer 1
        x = self.conv1(x)
        if self.normalization_type == NormalizationType.layer:
            _lnorm = nn.LayerNorm(x.size()[1:]).to(x.device)
            x = _lnorm(x)
        else:
            x = self.norm1(x)
        x = self.pool(torch.relu(x))

        # layer 2
        x = self.conv2(x)
        if self.normalization_type == NormalizationType.layer:
            _lnorm = nn.LayerNorm(x.size()[1:]).to(x.device)
            x = _lnorm(x)
        else:
            x = self.norm2(x)
        x = self.pool(torch.relu(x))

        # fc1,2,3
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class VGGInspiredCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(VGGInspiredCNN, self).__init__()
        # Convolutional block 1
        self.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        # Convolutional block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        # Convolutional block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Max pooling layer with a 2x2 kernel (applied after each conv block)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Assuming input size of 32x32, after three poolings: 32 -> 16 -> 8 -> 4.
        # The flattened feature dimension is then 256 * 4 * 4 = 4096.
        self.fc1 = nn.Linear(256 * 4 * 4, 512)  # first dense layer
        self.fc2 = nn.Linear(512, 128)          # second dense layer
        self.fc3 = nn.Linear(128, num_classes)  # decision layer

    def forward(self, x):
        # Convolutional Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Convolutional Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Convolutional Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten the feature maps for the fully connected layers
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Softmax activation for classification
        x = F.softmax(x, dim=1)
        return x


if __name__ == '__main__':
    base_model = LeNetBasedModel()

    models = [copy.deepcopy(base_model) for _ in range(2)]
    print(torch.equal(models[0].network[0].weight,
          models[1].network[0].weight))

    torch.manual_seed(42)
    model1 = LeNetBasedModel()

    torch.manual_seed(42)
    model2 = LeNetBasedModelWithNormalization(
        normalization_type=NormalizationType.batch)

    torch.manual_seed(42)
    model3 = LeNetBasedModelWithNormalization(
        normalization_type=NormalizationType.layer)

    torch.manual_seed(42)
    model4 = LeNetBasedModelWithNormalization(
        normalization_type=NormalizationType.instance)

    print(torch.equal(model1.network[0].weight, model2.conv1.weight), torch.equal(
        model1.network[3].weight, model2.conv2.weight), torch.equal(model1.network[7].weight, model2.fc1.weight), torch.equal(model1.network[9].weight, model2.fc2.weight), torch.equal(model1.network[11].weight, model2.fc3.weight))
    print(torch.equal(model2.conv1.weight, model3.conv1.weight), torch.equal(model2.conv2.weight, model3.conv2.weight), torch.equal(
        model2.fc1.weight, model3.fc1.weight), torch.equal(model2.fc2.weight, model3.fc2.weight), torch.equal(model2.fc3.weight, model3.fc3.weight))
    print(torch.equal(model3.conv1.weight, model4.conv1.weight), torch.equal(model3.conv2.weight, model4.conv2.weight), torch.equal(
        model3.fc1.weight, model4.fc1.weight), torch.equal(model3.fc2.weight, model4.fc2.weight), torch.equal(model3.fc3.weight, model4.fc3.weight))
